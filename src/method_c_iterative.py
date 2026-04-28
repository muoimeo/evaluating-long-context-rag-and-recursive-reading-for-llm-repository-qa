import argparse
import json
import os
import re
import time
from typing import Any, Dict, List, Set

from openai import OpenAI

from config import API_KEY, INDEX_PATH, MODEL_NAME, OLLAMA_BASE_URL, TOP_K_RETRIEVAL
from method_c_state import (
    bounded_text as _bounded_text,
    build_answer_plan,
    covered_facet_ids as _covered_facet_ids,
    covers_required_subparts,
    dataset_of_path as _dataset_of_path,
    evidence_files_for_facts as _evidence_files_for_facts,
    evidence_limit_for_mode as _evidence_limit_for_mode,
    fact_datasets as _fact_datasets,
    fact_key as _fact_key,
    fact_matches_facet as _fact_matches_facet,
    facts_by_support_type as _facts_by_support_type,
    has_comparison_relation_fact as _has_comparison_relation_fact,
    infer_required_facets,
    make_initial_state,
    merge_search_agenda as _merge_search_agenda,
    prune_resolved_agenda as _prune_resolved_agenda,
    question_mode as _question_mode,
    should_stop,
    trim_verified_facts as _trim_verified_facts,
)
from method_c_tooling import (
    attribute_terms_from_text as _attribute_terms_from_text,
    definition_score_for_doc as _definition_score_for_doc,
    dedupe_text as _dedupe,
    doc_to_result as _doc_to_result,
    extract_structured_candidates,
    find_definition,
    find_references,
    is_specific_symbol as _is_specific_symbol,
    load_index_docs as _load_index_docs,
    matched_definition_symbols as _matched_definition_symbols,
    normalize_tokens as _normalize_tokens,
    read_neighbors,
    search_attribute_reads,
    search_constructor_calls,
    symbol_terms_from_text as _symbol_terms_from_text,
)
from reranker import is_docs_noise, rerank_and_select
from vector_store import VectorStore

client = OpenAI(base_url=OLLAMA_BASE_URL, api_key=API_KEY)

# Bounded, local-model-compatible structure-aware reader. This is not an
# autonomous agent; it is deterministic iterative retrieval with evidence-bound state.
MAX_VERIFIED_FACTS = 18
MAX_AGENDA_ITEMS = 8
MAX_CANDIDATES = 16
MAX_EVIDENCE_SNIPPETS = 5
MIN_FACTS_TO_STOP = 2
FOLLOWUP_ROUNDS = 1
FOLLOWUP_QUERY_LIMIT = 3
FOLLOWUP_RETRIEVAL_K = 3
FOLLOWUP_READ_K = 3
MAX_EXACT_LOOKUP_CHUNKS = 5
MAX_CHUNK_FACTS = 4
MAX_TOOL_ACTIONS = 6
MAX_TOOL_RESULTS_PER_ACTION = 3
NEIGHBOR_WINDOW = 1
QUICK_ROLE_MAX_CHARS = 2200
QUICK_ROLE_MAX_TOKENS = 10
FINAL_EVIDENCE_LIMITS = {"extract": 5, "trace": 7, "compare": 8}

VALID_ROLES = {"direct", "bridge", "noise"}
VALID_PRIORITIES = {"high", "medium", "low"}
VALID_SUPPORT_TYPES = {"answer", "bridge", "context"}
PRIORITY_RANK = {"high": 0, "medium": 1, "low": 2}


def _merge_candidates(state: Dict[str, Any], key: str, values: List[str]) -> None:
    state[key] = _dedupe(state.get(key, []) + values, MAX_CANDIDATES)


def _merge_tool_actions(state: Dict[str, Any], actions: List[Dict[str, Any]]) -> None:
    existing = {
        (item.get("tool"), str(item.get("argument", "")).lower(), item.get("file"), item.get("line_start"))
        for item in state.get("tool_actions", [])
    }
    merged = list(state.get("tool_actions", []))
    for action in actions:
        key = (
            action.get("tool"),
            str(action.get("argument", "")).lower(),
            action.get("file"),
            action.get("line_start"),
        )
        if action.get("tool") and key not in existing:
            existing.add(key)
            merged.append(action)
    state["tool_actions"] = merged[:MAX_TOOL_ACTIONS]


def _is_docs_noise(file_path: str) -> bool:
    return is_docs_noise(file_path)


def _lookup_exact_chunks(state: Dict[str, Any], visited_chunk_ids: Set[str]) -> List[Dict[str, Any]]:
    docs = _load_index_docs()
    file_terms = [t.lower() for t in state.get("candidate_files", []) + state.get("candidate_paths", [])]
    symbol_terms = [t.lower().rstrip("()") for t in state.get("candidate_symbols", []) if len(t) >= 3]
    raw_symbol_terms = [t.rstrip("()") for t in state.get("candidate_symbols", []) if len(t) >= 3]
    exact, seen = [], set()
    definition_hits_by_symbol: Dict[str, Dict[str, Any]] = {}
    other_hits: List[Dict[str, Any]] = []
    for doc in docs:
        doc_id = f"{doc['file']}_{doc.get('chunk_index', 0)}"
        if doc_id in visited_chunk_ids or doc_id in seen:
            continue
        file_lower = doc["file"].lower()
        content_lower = doc.get("content", "").lower()
        file_match = any(t and (file_lower == t or file_lower.endswith(t) or t in file_lower) for t in file_terms)
        symbol_match = any(t and re.search(rf"\b{re.escape(t)}\b", content_lower) for t in symbol_terms[:MAX_CANDIDATES])
        definition_score = _definition_score_for_doc(doc, raw_symbol_terms)
        if file_match or symbol_match:
            seen.add(doc_id)
            result = _doc_to_result(doc, score=definition_score if definition_score is not None else 0.01 if symbol_match else 0.03)
            if definition_score is not None:
                for symbol in _matched_definition_symbols(doc, raw_symbol_terms):
                    old = definition_hits_by_symbol.get(symbol)
                    if old is None or result["score"] < old["score"]:
                        definition_hits_by_symbol[symbol] = result
            else:
                other_hits.append(result)
    ordered_definition_hits = []
    for symbol in raw_symbol_terms:
        hit = definition_hits_by_symbol.get(symbol)
        if hit and hit["id"] not in {item["id"] for item in ordered_definition_hits}:
            ordered_definition_hits.append(hit)
    remaining_definition_hits = sorted(
        [hit for hit in definition_hits_by_symbol.values() if hit["id"] not in {item["id"] for item in ordered_definition_hits}],
        key=lambda item: item["score"],
    )
    exact = ordered_definition_hits + remaining_definition_hits + other_hits
    return exact[:MAX_EXACT_LOOKUP_CHUNKS]


def build_followup_queries(question: str, state: Dict[str, Any]) -> List[str]:
    queries: List[str] = []
    agenda = sorted(state.get("search_agenda", []), key=lambda x: PRIORITY_RANK.get(x.get("priority", "low"), 2))
    for item in agenda:
        relation = item.get("missing_relation", "")
        if relation:
            queries.append(f"{question}\nFind evidence for missing relation: {relation}")
    for symbol in state.get("candidate_symbols", [])[:6]:
        queries.append(f"{question}\nDefinition of {symbol}")
        queries.append(f"{question}\nWhere is {symbol} called or registered?")
    for path in state.get("candidate_paths", [])[:4]:
        queries.append(f"{question}\nModule or import path {path}")
    for file_path in state.get("candidate_files", [])[:4]:
        queries.append(f"{question}\nRelevant file {file_path}")
    return _dedupe(queries, FOLLOWUP_QUERY_LIMIT)


def build_followup_tool_actions(question: str, state: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Create deterministic repo-navigation actions from current evidence state."""
    actions: List[Dict[str, Any]] = []
    facts_text = " ".join(
        f"{fact.get('fact', '')} {fact.get('supports_question_part', '')}"
        for fact in state.get("verified_facts", [])
    )
    symbols = _dedupe(state.get("candidate_symbols", []) + _symbol_terms_from_text(facts_text), MAX_CANDIDATES)
    secondary_symbol_actions: List[Dict[str, Any]] = []
    for symbol in symbols:
        clean = symbol.rstrip("()")
        if "." in clean:
            clean = clean.rsplit(".", 1)[-1]
        if not _is_specific_symbol(clean):
            continue
        actions.append({"tool": "find_definition", "argument": clean})
        if clean[0].isupper():
            secondary_symbol_actions.append({"tool": "search_constructor_calls", "argument": clean})
        else:
            secondary_symbol_actions.append({"tool": "find_references", "argument": clean})

    # Attribute reads are high-variance. Prefer attributes grounded in facts;
    # fall back to the question only before facts exist.
    attr_source = facts_text if facts_text.strip() else question
    attr_terms = _attribute_terms_from_text(attr_source)
    for attr in attr_terms[:4]:
        actions.append({"tool": "search_attribute_reads", "argument": attr})

    actions.extend(secondary_symbol_actions)

    # Neighbor reads are cheap and help when chunking cuts a definition before
    # assignments or the next call in a chain.
    for loc in state.get("read_locations", [])[-3:]:
        actions.append({
            "tool": "read_neighbors",
            "argument": loc.get("file"),
            "file": loc.get("file"),
            "line_start": loc.get("line_start", 1),
            "window": NEIGHBOR_WINDOW,
        })

    executed = {
        (item.get("tool"), str(item.get("argument", "")).lower(), item.get("file"), item.get("line_start"))
        for item in state.get("executed_tool_actions", [])
    }
    unique = []
    seen = set()
    for action in actions:
        key = (
            action.get("tool"),
            str(action.get("argument", "")).lower(),
            action.get("file"),
            action.get("line_start"),
        )
        if key in seen or key in executed:
            continue
        seen.add(key)
        unique.append(action)
        if len(unique) >= MAX_TOOL_ACTIONS:
            break
    _merge_tool_actions(state, unique)
    return unique


def execute_tool_action(action: Dict[str, Any]) -> List[Dict[str, Any]]:
    tool = action.get("tool")
    argument = str(action.get("argument", "") or "")
    if tool == "find_definition":
        return find_definition(argument)
    if tool == "find_references":
        return find_references(argument)
    if tool == "search_attribute_reads":
        return search_attribute_reads(argument)
    if tool == "search_constructor_calls":
        return search_constructor_calls(argument)
    if tool == "read_neighbors":
        file_path = str(action.get("file") or argument)
        return read_neighbors(file_path, int(action.get("line_start", 1)), int(action.get("window", NEIGHBOR_WINDOW)))
    return []


def _rank_tool_hits_for_state(hits: List[Dict[str, Any]], state: Dict[str, Any]) -> List[Dict[str, Any]]:
    preferred_files = [loc.get("file") for loc in state.get("read_locations", [])[-4:]]
    preferred_files.extend(state.get("candidate_files", [])[:6])
    preferred_files = _dedupe([fp for fp in preferred_files if fp])

    def key(item: Dict[str, Any]) -> tuple:
        file_path = item["metadata"]["file"]
        try:
            file_rank = preferred_files.index(file_path)
        except ValueError:
            file_rank = len(preferred_files) + 1
        docs_penalty = 1 if _is_docs_noise(file_path) else 0
        return (file_rank, docs_penalty, item["score"], file_path, item["metadata"]["chunk_index"])

    return sorted(hits, key=key)


def run_followup_tools(question: str, state: Dict[str, Any], visited_chunk_ids: Set[str]) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []
    for action in build_followup_tool_actions(question, state):
        state.setdefault("executed_tool_actions", []).append(action)
        hits = _rank_tool_hits_for_state(execute_tool_action(action), state)[:MAX_TOOL_RESULTS_PER_ACTION]
        if hits:
            print(f"  tool {action['tool']}({action.get('argument')}) -> {len(hits)} hits")
        for hit in hits:
            if hit["id"] not in visited_chunk_ids:
                results.append(hit)
    return results


def _normalize_read_result(result: Dict[str, Any]) -> Dict[str, Any]:
    role = str(result.get("evidence_role", "noise")).lower().strip()
    if role not in VALID_ROLES:
        role = "noise"
    facts = []
    for item in result.get("verified_facts", []) or []:
        fact = item if isinstance(item, str) else item.get("fact", "")
        part = "unspecified" if isinstance(item, str) else item.get("supports_question_part", "unspecified")
        support_type = "answer" if role == "direct" else "bridge" if role == "bridge" else "context"
        if not isinstance(item, str):
            support_type = str(item.get("support_type", support_type)).lower().strip()
        if support_type not in VALID_SUPPORT_TYPES:
            support_type = "context"
        fact = _bounded_text(fact)
        if fact:
            facts.append({
                "fact": fact,
                "supports_question_part": _bounded_text(part, 160) or "unspecified",
                "support_type": support_type,
            })
        if len(facts) >= MAX_CHUNK_FACTS:
            break
    agenda = []
    for item in result.get("search_agenda", []) or []:
        relation = item if isinstance(item, str) else item.get("missing_relation", "")
        priority = "medium" if isinstance(item, str) else str(item.get("priority", "medium")).lower()
        relation = _bounded_text(relation, 220)
        if relation:
            agenda.append({"missing_relation": relation, "priority": priority if priority in VALID_PRIORITIES else "medium"})
        if len(agenda) >= 4:
            break
    return {
        "evidence_role": role,
        "verified_facts": facts,
        "search_agenda": agenda,
        "candidate_symbols": _dedupe(result.get("candidate_symbols", []) or [], MAX_CANDIDATES),
        "candidate_files": _dedupe(result.get("candidate_files", []) or [], MAX_CANDIDATES),
        "candidate_paths": _dedupe(result.get("candidate_paths", []) or [], MAX_CANDIDATES),
        "answer_ready": bool(result.get("answer_ready", False)),
        "tiny_reasoning": _bounded_text(result.get("tiny_reasoning", ""), 240),
        "tokens_in": result.get("tokens_in", 0),
        "tokens_out": result.get("tokens_out", 0),
    }


def _is_annotation_heavy_low_signal_chunk(question: str, chunk_text: str) -> bool:
    """Generic guard against treating parameter docs as decisive implementation evidence."""
    annotation_count = len(re.findall(r"\b(?:Doc|Annotated)\s*\(", chunk_text))
    if annotation_count < 3:
        return False
    # Annotated/Doc often appears in real FastAPI implementation signatures, not
    # only docs. Require genuinely weak implementation signal before warning.
    implementation_signal = len(re.findall(
        r"\b(?:async\s+def|def|class|return|yield|raise|await)\b|"
        r"self\.|=\s*|->|:\s*Annotated\[|"
        r"\.(?:append|extend|get|setdefault|update|add_route|add_api_route)\s*\(",
        chunk_text,
    ))
    question_overlap = len(_normalize_tokens(question) & _normalize_tokens(chunk_text))
    return implementation_signal < 3 and question_overlap < 5


def quick_chunk_role(question: str, question_type: str, file_path: str, chunk_text: str,
                     line_start: int, line_end: int) -> Dict[str, Any]:
    """Cheap first-pass role classifier.

    The goal is not to reason. It only avoids running the expensive structured
    reader on obvious noise. Ambiguous chunks become "bridge" so recall is
    favored over speed.
    """
    snippet = chunk_text[:QUICK_ROLE_MAX_CHARS]
    prompt = f"""Classify this repository chunk for answering the question.

Return exactly one word: direct, bridge, or noise.

direct = contains direct answer evidence.
bridge = contains definitions, calls, symbols, config keys, or file paths that could help find the answer.
noise = unrelated or only generic tutorial/documentation text.
For compare/hybrid questions, do not mark a chunk as noise if it appears to support one side of the comparison.
Policy/config/IAM JSON/YAML chunks with Action, Effect, Resource, Statement, permission, or service:Action fields are direct or bridge evidence for permission/config questions.
If unsure, return bridge.

QUESTION TYPE: {question_type}
QUESTION: {question}
CHUNK: {file_path} lines {line_start}-{line_end}
SNIPPET:
{snippet}
"""
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=QUICK_ROLE_MAX_TOKENS,
        )
        content = response.choices[0].message.content.strip().lower()
        match = re.search(r"\b(direct|bridge|noise)\b", content)
        role = match.group(1) if match else "bridge"
        return {
            "role": role,
            "tokens_in": response.usage.prompt_tokens if response.usage else 0,
            "tokens_out": response.usage.completion_tokens if response.usage else 0,
            "model_calls": 1,
        }
    except Exception as exc:
        return {
            "role": "bridge",
            "tokens_in": 0,
            "tokens_out": 0,
            "model_calls": 1,
            "error": str(exc),
        }


def _side_signal(question: str, file_path: str, chunk_text: str) -> str:
    blob = f"{file_path}\n{chunk_text[:1200]}".lower()
    q = question.lower()
    if "fastapi" in q and ("fastapi/" in file_path or "fastapi" in blob):
        return "fastapi"
    if any(term in q for term in ["aws", "lambda", "api gateway", "iam", "sam"]):
        if any(term in blob for term in [
            "sample-apps/", "iam-policies/", "template.yaml", "template.yml",
            "lambda:", "apigateway", "dynamodb", "s3:", "kinesis",
        ]):
            return "lambda"
    return ""


def _state_has_side(state: Dict[str, Any], side: str) -> bool:
    if not side:
        return False
    for fact in state.get("verified_facts", []):
        if side in str(fact.get("supports_question_part", "")).lower():
            return True
        for evidence_id in fact.get("evidence_ids", []):
            if side == "fastapi" and "fastapi/" in evidence_id:
                return True
            if side == "lambda" and any(term in evidence_id for term in ["sample-apps/", "iam-policies/", "templates/"]):
                return True
    return False


def _is_policy_or_config_evidence(question: str, file_path: str, chunk_text: str) -> bool:
    q = question.lower()
    if not any(term in q for term in ["permission", "policy", "iam", "config", "template", "resource", "api gateway", "sam"]):
        return False
    if not file_path.endswith((".json", ".yml", ".yaml")) and "iam-policies/" not in file_path:
        return False
    return bool(re.search(
        r'"(?:Action|Effect|Resource|Statement|Sid)"\s*:|'
        r"\b(?:Action|Effect|Resource|Statement)\s*:|"
        r"\b[a-z0-9-]+:[A-Za-z*]+",
        chunk_text,
        flags=re.I,
    ))


def _strong_path_or_symbol_overlap(question: str, file_path: str, chunk_text: str) -> bool:
    q_tokens = _normalize_tokens(question)
    path_tokens = _normalize_tokens(file_path.replace("/", " ").replace(".", " "))
    if len(q_tokens & path_tokens) >= 2:
        return True
    signature_hits = re.findall(r"\b(?:async\s+def|def|class)\s+([A-Za-z_][A-Za-z0-9_]*)", chunk_text)
    return any(_normalize_tokens(symbol) & q_tokens for symbol in signature_hits)


def override_quick_role(quick_role: str, question: str, state: Dict[str, Any],
                        file_path: str, chunk_text: str) -> str:
    """Prevent the cheap classifier from dropping high-recall evidence.

    Stage 1 is allowed to save time only on obvious noise. Compare/hybrid sides
    and policy/config artifacts are too important to skip based on a tiny prompt.
    """
    if quick_role != "noise":
        return quick_role
    if _is_policy_or_config_evidence(question, file_path, chunk_text):
        return "direct"
    mode = state.get("question_type", _question_mode(question))
    if mode == "compare":
        side = _side_signal(question, file_path, chunk_text)
        if side and not _state_has_side(state, side):
            return "bridge"
    if _strong_path_or_symbol_overlap(question, file_path, chunk_text):
        return "bridge"
    return quick_role


def read_and_reason(question: str, state: Dict[str, Any],
                    file_path: str, chunk_text: str,
                    line_start: int, line_end: int,
                    chunk_num: int, total_chunks: int) -> Dict[str, Any]:
    numbered = "\n".join(f"{line_start + i}: {line}" for i, line in enumerate(chunk_text.split("\n")))
    compact_state = {
        "verified_facts": state.get("verified_facts", [])[-8:],
        "search_agenda": state.get("search_agenda", [])[:MAX_AGENDA_ITEMS],
        "candidate_symbols": state.get("candidate_symbols", [])[:8],
        "candidate_files": state.get("candidate_files", [])[:8],
        "candidate_paths": state.get("candidate_paths", [])[:8],
        "question_type": state.get("question_type", _question_mode(question)),
        "required_facets": state.get("required_facets", []),
    }
    doc_noise_hint = ""
    if _is_annotation_heavy_low_signal_chunk(question, chunk_text):
        doc_noise_hint = (
            "\nIMPORTANT: This chunk appears annotation/documentation-heavy and has little "
            "implementation signal. Be skeptical, but still extract facts if the chunk shows "
            "concrete control flow, data flow, registration, storage, calls, return values, "
            "schema/config keys, or definitions.\n"
        )

    prompt = f"""You are reading one repository chunk for a bounded iterative QA system.

QUESTION:
{question}
{doc_noise_hint}

CURRENT STRUCTURED STATE:
{json.dumps(compact_state, indent=2)}

NEW CHUNK: {file_path} (Lines {line_start}-{line_end}), chunk {chunk_num}/{total_chunks}
{numbered}

TASK:
1. Classify this chunk as "direct", "bridge", or "noise".
2. Extract only facts grounded in this chunk. Do not summarize generic framework behavior.
3. Each fact must say which part of the question it supports.
4. Label each fact with "support_type":
   - "answer": directly answers a requested implementation detail or comparison dimension.
   - "bridge": connects symbols/files/callers/callees needed to form a trace.
   - "context": useful background but not decisive support.
5. Identify missing relations still needed to answer fully.
6. Emit candidate symbols/files/paths visible in this chunk for follow-up retrieval.
7. Return strict JSON only.

EXTRACTION RULES:
- Preserve exact symbol names, method names, field names, config keys, IAM/action names, return values, status codes, and literal values visible in the chunk.
- Prefer concrete facts of the form "symbol/field/action -> effect/result".
- For call chains, write "A() -> B() -> C()" when the chunk directly shows the calls.
- For policy/config files, include all locally relevant actions/keys in the same block; do not select only one subsection if the question asks for multiple categories.
- For comparison questions, say which side the fact supports: fastapi, lambda, or both.

JSON schema:
{{
  "evidence_role": "direct|bridge|noise",
  "verified_facts": [
    {{
      "fact": "concise chunk-grounded implementation fact",
      "supports_question_part": "sub-question, trace step, side, or comparison dimension supported",
      "support_type": "answer|bridge|context"
    }}
  ],
  "search_agenda": [
    {{"missing_relation": "specific unresolved relation, caller, callee, registration, config consumer, or compared side", "priority": "high|medium|low"}}
  ],
  "candidate_symbols": ["function/class/method/config key/decorator names"],
  "candidate_files": ["file paths or filenames"],
  "candidate_paths": ["import/module/dotted paths"],
  "answer_ready": false,
  "tiny_reasoning": "optional one sentence, max 30 words"
}}"""
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=600,
        )
        content = response.choices[0].message.content.strip()
        try:
            parsed = json.loads(content)
        except json.JSONDecodeError:
            if "```" not in content:
                parsed = {"evidence_role": "noise"}
            else:
                json_str = content.split("```")[1]
                parsed = json.loads(json_str[4:].strip() if json_str.startswith("json") else json_str.strip())
        parsed["tokens_in"] = response.usage.prompt_tokens if response.usage else 0
        parsed["tokens_out"] = response.usage.completion_tokens if response.usage else 0
        normalized = _normalize_read_result(parsed)
        return normalized
    except Exception as exc:
        return {
            "evidence_role": "noise",
            "verified_facts": [],
            "search_agenda": [],
            "candidate_symbols": [],
            "candidate_files": [],
            "candidate_paths": [],
            "answer_ready": False,
            "tiny_reasoning": "",
            "tokens_in": 0,
            "tokens_out": 0,
            "error": str(exc),
        }


def update_state_from_step(state: Dict[str, Any], step_result: Dict[str, Any],
                           file_path: str, chunk_text: str,
                           chunk_id: str | None = None) -> int:
    chunk_id = chunk_id or file_path
    normalized = _normalize_read_result(step_result)
    state.setdefault("evidence_roles", {})[chunk_id] = normalized["evidence_role"]
    existing = {_fact_key(fact): fact for fact in state.get("verified_facts", [])}
    new_count = 0
    fact_texts = []
    new_fact_objects = []
    for fact in normalized["verified_facts"]:
        text = fact["fact"]
        fact_texts.append(text)
        key = _fact_key(fact)
        if key in existing:
            ids = existing[key].setdefault("evidence_ids", [])
            if chunk_id not in ids:
                ids.append(chunk_id)
            continue
        fact_object = {
            "fact": text,
            "supports_question_part": fact["supports_question_part"],
            "support_type": fact["support_type"],
            "evidence_ids": [chunk_id],
        }
        fact_object["covered_facets"] = _covered_facet_ids(
            fact_object,
            state.get("required_facets", []),
            file_path=file_path,
        )
        state.setdefault("verified_facts", []).append(fact_object)
        new_fact_objects.append(fact_object)
        new_count += 1
    _trim_verified_facts(state)
    _merge_search_agenda(state, normalized["search_agenda"])
    _prune_resolved_agenda(state, new_fact_objects)
    candidates = extract_structured_candidates(file_path, chunk_text, fact_texts)
    _merge_candidates(state, "candidate_symbols", normalized["candidate_symbols"] + candidates["candidate_symbols"])
    _merge_candidates(state, "candidate_files", normalized["candidate_files"] + candidates["candidate_files"])
    _merge_candidates(state, "candidate_paths", normalized["candidate_paths"] + candidates["candidate_paths"])
    if normalized.get("tiny_reasoning"):
        state["tiny_reasoning"] = normalized["tiny_reasoning"]
    state["answer_ready"] = bool(state.get("answer_ready", False) or normalized.get("answer_ready", False))
    return new_count


def score_evidence_relevance(evidence_item: Dict[str, Any], question: str) -> float:
    q_tokens = _normalize_tokens(question)
    if not q_tokens:
        return 0.0
    score = len(q_tokens & _normalize_tokens(evidence_item["text"])) / len(q_tokens)
    if q_tokens & _normalize_tokens(evidence_item["file"]):
        score += 0.2
    if evidence_item.get("evidence_role") == "direct":
        score += 0.35
    elif evidence_item.get("evidence_role") == "bridge":
        score += 0.15
    if _is_docs_noise(evidence_item["file"]):
        score *= 0.5
    return score


def select_final_evidence(gathered_evidence: List[Dict[str, Any]], state: Dict[str, Any],
                          question: str, limit: int = MAX_EVIDENCE_SNIPPETS) -> List[Dict[str, Any]]:
    if not gathered_evidence:
        return []
    by_id = {item["id"]: item for item in gathered_evidence}
    selected, seen = [], set()
    for fact in state.get("verified_facts", []):
        for evidence_id in fact.get("evidence_ids", []):
            item = by_id.get(evidence_id)
            if item and item["id"] not in seen:
                selected.append(item)
                seen.add(item["id"])
            if len(selected) >= limit:
                return selected
    scored = sorted(
        ((score_evidence_relevance(item, question), item) for item in gathered_evidence),
        key=lambda pair: pair[0],
        reverse=True,
    )
    if state.get("question_type") == "compare":
        for dataset_name in ("fastapi", "lambda"):
            for _, item in scored:
                if item["id"] not in seen and _dataset_of_path(item["file"]) == dataset_name:
                    selected.append(item)
                    seen.add(item["id"])
                    break
    for _, item in scored:
        if item["id"] in seen:
            continue
        selected.append(item)
        seen.add(item["id"])
        if len(selected) >= limit:
            break
    return selected[:limit]


def format_selected_evidence(raw_evidence: List[Dict[str, Any]], limit: int | None = None) -> str:
    if not raw_evidence:
        return "(none gathered)"
    blocks = []
    effective_limit = limit if limit is not None else len(raw_evidence)
    for item in raw_evidence[:effective_limit]:
        blocks.append(
            f"--- {item['id']} | {item['file']} (Lines {item['line_start']}-{item['line_end']}) ---\n"
            f"{item['text']}"
        )
    return "\n\n".join(blocks)


def format_answer_plan(plan: Dict[str, Any]) -> str:
    lines = []
    for facet in plan.get("facets", []):
        facts = facet.get("facts", [])
        status = "covered" if facts else "missing"
        lines.append(f"- {facet.get('label', facet.get('id'))}: {status}")
        for fact in facts[:4]:
            evidence = ", ".join(fact.get("evidence_ids", []))
            lines.append(f"  fact: {fact.get('fact')} [evidence: {evidence}]")
    missing = plan.get("missing_facets", [])
    if missing:
        lines.append("Missing facets: " + "; ".join(missing))
    return "\n".join(lines) if lines else "(no answer plan)"


def format_final_context(state: Dict[str, Any], selected_raw_evidence: List[Dict[str, Any]],
                         evidence_limit: int | None = None) -> str:
    selected_by_id = {item["id"]: item for item in selected_raw_evidence}
    answer_plan = build_answer_plan(state, selected_by_id)
    facts = []
    for idx, fact in enumerate(state.get("verified_facts", []), start=1):
        facts.append(
            f"{idx}. {fact['fact']} "
            f"[type: {fact.get('support_type', 'context')}; "
            f"supports: {fact.get('supports_question_part', 'unspecified')}; "
            f"facets: {', '.join(fact.get('covered_facets', [])) or 'unmapped'}; "
            f"evidence: {', '.join(fact.get('evidence_ids', []))}]"
        )
    agenda = [
        f"- {item.get('missing_relation')} ({item.get('priority', 'medium')})"
        for item in state.get("search_agenda", [])
    ]
    return (
        "=== VERIFIED FACTS ===\n"
        + ("\n".join(facts) if facts else "(none)")
        + "\n\n=== STRUCTURED ANSWER PLAN ===\n"
        + format_answer_plan(answer_plan)
        + "\n\n=== UNRESOLVED EVIDENCE GAPS ===\n"
        + ("\n".join(agenda) if agenda else "(none)")
        + "\n\n=== SELECTED RAW EVIDENCE ===\n"
        + format_selected_evidence(selected_raw_evidence, evidence_limit)
    )


def final_answer(question: str, state: Dict[str, Any], citations: List[Dict[str, Any]],
                 selected_raw_evidence: List[Dict[str, Any]]) -> Dict[str, Any]:
    mode = state.get("question_type", _question_mode(question))
    evidence_limit = _evidence_limit_for_mode(mode)
    context = format_final_context(state, selected_raw_evidence, evidence_limit)
    task_instruction = {
        "compare": (
            "This is a comparison question. Cover both sides explicitly. "
            "Cover side A, side B, and the comparison relation. "
            "If one side lacks evidence, say which side is unsupported."
        ),
        "trace": (
            "This is a trace question. Give a short ordered flow and name the concrete "
            "files/functions/classes involved. Do not skip intermediate steps visible in evidence."
        ),
        "extract": (
            "This is an implementation-detail question. Answer with exact symbols, files, "
            "return values, config keys, or handlers visible in the evidence."
        ),
    }[mode]
    system_prompt = (
        "You synthesize a repository-QA answer from verified facts and raw evidence.\n"
        "PRIORITY: concrete implementation details from evidence. Use exact symbol names, "
        "method/function names, file paths, config keys, action names, return values, "
        "status codes, and literal values visible in the evidence.\n"
        "Use STRUCTURED ANSWER PLAN as the outline; use verified facts as support; "
        "use raw evidence to recover exact syntax and missing local details.\n"
        "For multi-part questions, answer each requested part explicitly.\n"
        "For implementation questions, describe the concrete call/registration/storage/validation chain visible in evidence.\n"
        "If UNRESOLVED EVIDENCE GAPS is not '(none)', do not pretend those gaps are resolved. "
        "Answer the supported part and name the missing relation instead of filling it from general knowledge.\n"
        "Keep the answer concise; avoid code fences unless the question explicitly asks for code.\n"
        f"{task_instruction}"
    )
    user_prompt = f"QUESTION:\n{question}\n\nCONTEXT:\n{context}\n\nWrite a concise grounded answer."
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
            temperature=0.0,
            max_tokens=1000,
        )
        return {
            "answer": response.choices[0].message.content.strip(),
            "evidence": citations,
            "input_tokens": response.usage.prompt_tokens if response.usage else 0,
            "output_tokens": response.usage.completion_tokens if response.usage else 0,
        }
    except Exception as exc:
        return {"answer": f"Error in final synthesis: {exc}", "evidence": citations, "input_tokens": 0, "output_tokens": 0}


def read_chunk_sequence(question: str, state: Dict[str, Any], chunks: List[Dict[str, Any]],
                        grounded_citations: List[Dict[str, Any]], gathered_evidence: List[Dict[str, Any]],
                        visited_chunk_ids: Set[str], round_label: str) -> Dict[str, Any]:
    tokens_in = tokens_out = model_calls = new_facts = useful_chunks = 0
    if not chunks:
        return {"tokens_in": 0, "tokens_out": 0, "model_calls": 0, "new_facts": 0, "useful_chunks": 0}
    print(f"\n{round_label}: reading {len(chunks)} chunks...")
    for i, res in enumerate(chunks, start=1):
        meta = res["metadata"]
        file_path = meta["file"]
        line_start = meta.get("line_start", 1)
        line_end = meta.get("line_end", 1)
        chunk_text = res["document"]
        chunk_id = res["id"]
        visited_chunk_ids.add(chunk_id)
        if file_path not in state.setdefault("visited_files", []):
            state["visited_files"].append(file_path)
        state.setdefault("read_locations", []).append({
            "file": file_path,
            "line_start": line_start,
            "line_end": line_end,
        })
        state["read_locations"] = state["read_locations"][-12:]
        print(f"  [{i}/{len(chunks)}] Reading {file_path} (lines {line_start}-{line_end})...", end=" ")
        quick = quick_chunk_role(
            question=question,
            question_type=state.get("question_type", _question_mode(question)),
            file_path=file_path,
            chunk_text=chunk_text,
            line_start=line_start,
            line_end=line_end,
        )
        model_calls += quick.get("model_calls", 1)
        tokens_in += quick.get("tokens_in", 0)
        tokens_out += quick.get("tokens_out", 0)
        quick_role = override_quick_role(
            quick.get("role", "bridge"),
            question,
            state,
            file_path,
            chunk_text,
        )
        if quick_role == "noise":
            state.setdefault("evidence_roles", {})[chunk_id] = "noise"
            print("QUICK_NOISE | skipped full read")
            continue

        step = read_and_reason(question, state, file_path, chunk_text, line_start, line_end, i, len(chunks))
        model_calls += 1
        tokens_in += step.get("tokens_in", 0)
        tokens_out += step.get("tokens_out", 0)
        before = len(state.get("verified_facts", []))
        added = update_state_from_step(state, step, file_path, chunk_text, chunk_id)
        new_facts += max(added, len(state.get("verified_facts", [])) - before)
        role = state.get("evidence_roles", {}).get(chunk_id, "noise")
        useful = role in {"direct", "bridge"} or added > 0
        if useful:
            useful_chunks += 1
            grounded_citations.append({"file": file_path, "line_start": line_start, "line_end": line_end})
            gathered_evidence.append({
                "id": chunk_id,
                "file": file_path,
                "line_start": line_start,
                "line_end": line_end,
                "text": chunk_text,
                "evidence_role": role,
            })
        print(f"{role.upper()} | +{added} facts ({len(state.get('verified_facts', []))} total)")
        if useful and role == "direct":
            evidence_by_id = {item["id"]: item for item in gathered_evidence}
            if should_stop(state, evidence_by_id, no_progress_rounds=0):
                print(f"  Early stop inside {round_label}: typed evidence coverage is sufficient.")
                break
    return {
        "tokens_in": tokens_in,
        "tokens_out": tokens_out,
        "model_calls": model_calls,
        "new_facts": new_facts,
        "useful_chunks": useful_chunks,
    }


def expand_retrieval(vs: VectorStore, question: str, state: Dict[str, Any], visited_chunk_ids: Set[str]) -> List[Dict[str, Any]]:
    queries = build_followup_queries(question, state)
    # Tool hits are deterministic code-navigation reads. Vector retrieval remains
    # a fallback for semantic gaps that tools cannot express.
    retrieved = list(run_followup_tools(question, state, visited_chunk_ids))
    retrieved.extend(_lookup_exact_chunks(state, visited_chunk_ids))
    for query in queries[:FOLLOWUP_QUERY_LIMIT]:
        retrieved.extend(vs.retrieve(query, top_k=FOLLOWUP_RETRIEVAL_K))
    unique, seen = [], set()
    for item in retrieved:
        if item["id"] in seen or item["id"] in visited_chunk_ids:
            continue
        seen.add(item["id"])
        unique.append(item)
    if not unique:
        return []
    reranked = rerank_and_select(unique, top_k=max(FOLLOWUP_READ_K * 2, FOLLOWUP_READ_K), query=question)
    source_first = [item for item in reranked if not _is_docs_noise(item["metadata"]["file"])]
    return (source_first or reranked)[:FOLLOWUP_READ_K]


def dedupe_citations(citations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen, unique = set(), []
    for item in citations:
        key = (item["file"], item["line_start"], item["line_end"])
        if key in seen:
            continue
        seen.add(key)
        unique.append(item)
    return unique


def _dedupe_evidence_items(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen, unique = set(), []
    for item in items:
        if item["id"] not in seen:
            seen.add(item["id"])
            unique.append(item)
    return unique


def _citation_from_result(res: Dict[str, Any]) -> Dict[str, Any]:
    meta = res["metadata"]
    return {"file": meta["file"], "line_start": meta.get("line_start", 1), "line_end": meta.get("line_end", 1)}


def _citation_from_evidence(item: Dict[str, Any]) -> Dict[str, Any]:
    return {"file": item["file"], "line_start": item.get("line_start", 1), "line_end": item.get("line_end", 1)}


def _evidence_from_result(res: Dict[str, Any], role: str = "fallback") -> Dict[str, Any]:
    meta = res["metadata"]
    return {
        "id": res["id"],
        "file": meta["file"],
        "line_start": meta.get("line_start", 1),
        "line_end": meta.get("line_end", 1),
        "text": res["document"],
        "evidence_role": role,
    }


def run_method_c(question: str, top_k: int = TOP_K_RETRIEVAL) -> Dict[str, Any]:
    print("=" * 50)
    print("Method C: Structure-Aware Iterative Reader")
    print("=" * 50)
    start_time = time.time()
    total_tokens_in = total_tokens_out = model_calls = 0

    print(f"\nStep 1: Retrieving top-{top_k} candidates from {top_k * 3} pool...")
    vs = VectorStore()
    vs.validate_manifest(INDEX_PATH, raise_on_mismatch=True)
    raw_results = vs.retrieve(question, top_k=top_k * 3)
    if not raw_results:
        return {
            "success": False,
            "error": "No chunks retrieved.",
            "latency": time.time() - start_time,
            "input_tokens": 0,
            "output_tokens": 0,
            "model_calls": 0,
        }
    reranked = rerank_and_select(raw_results, top_k=top_k * 2, query=question)
    source_first = [res for res in reranked if not _is_docs_noise(res["metadata"]["file"])]
    source_chunks = (source_first or reranked)[:top_k]
    print(f"  {len(source_chunks)} chunks selected ({len(reranked) - len(source_first)} docs filtered out before top-k)")

    state = make_initial_state(question)
    grounded_citations: List[Dict[str, Any]] = []
    gathered_evidence: List[Dict[str, Any]] = []
    visited_chunk_ids: Set[str] = set()

    stats = read_chunk_sequence(question, state, source_chunks, grounded_citations, gathered_evidence, visited_chunk_ids, "Initial read")
    total_tokens_in += stats["tokens_in"]
    total_tokens_out += stats["tokens_out"]
    model_calls += stats["model_calls"]
    evidence_by_id = {item["id"]: item for item in gathered_evidence}
    no_progress_rounds = 0

    for round_idx in range(FOLLOWUP_ROUNDS):
        if should_stop(state, evidence_by_id, no_progress_rounds):
            print("\nStopping: evidence coverage is sufficient or no-progress rule fired.")
            break
        print(f"\nStep 2.{round_idx + 1}: Structured follow-up retrieval...")
        followup_chunks = expand_retrieval(vs, question, state, visited_chunk_ids)
        if not followup_chunks:
            print("  No new high-relevance follow-up chunks.")
            if should_stop(state, evidence_by_id, no_progress_rounds, retrieval_empty=True):
                break
            no_progress_rounds += 1
            continue
        stats = read_chunk_sequence(
            question, state, followup_chunks, grounded_citations, gathered_evidence,
            visited_chunk_ids, f"Follow-up round {round_idx + 1}"
        )
        total_tokens_in += stats["tokens_in"]
        total_tokens_out += stats["tokens_out"]
        model_calls += stats["model_calls"]
        gathered_evidence = _dedupe_evidence_items(gathered_evidence)
        evidence_by_id = {item["id"]: item for item in gathered_evidence}
        no_progress_rounds = no_progress_rounds + 1 if stats["new_facts"] <= 0 else 0

    grounded_citations = dedupe_citations(grounded_citations)
    source_citations = [item for item in grounded_citations if not _is_docs_noise(item["file"])]
    if source_citations:
        grounded_citations = source_citations
    if not gathered_evidence:
        gathered_evidence = [_evidence_from_result(res) for res in source_chunks[:MAX_EVIDENCE_SNIPPETS]]
        grounded_citations = [_citation_from_result(res) for res in source_chunks[:MAX_EVIDENCE_SNIPPETS]]
    if not state.get("verified_facts"):
        state["search_agenda"] = [{
            "missing_relation": "No verified facts were extracted; final answer must state what is unsupported.",
            "priority": "high",
        }]

    print("\nStep 3: Generating final answer from verified facts + selected raw evidence...")
    source_evidence = [item for item in gathered_evidence if not _is_docs_noise(item["file"])] or gathered_evidence
    final_evidence_limit = _evidence_limit_for_mode(state.get("question_type", "extract"))
    selected = select_final_evidence(source_evidence, state, question, limit=final_evidence_limit)
    selected_citations = dedupe_citations([_citation_from_evidence(item) for item in selected]) or grounded_citations
    answer_result = final_answer(question, state, selected_citations, selected)
    model_calls += 1
    total_tokens_in += answer_result.get("input_tokens", 0)
    total_tokens_out += answer_result.get("output_tokens", 0)
    return {
        "success": True,
        "answer": answer_result.get("answer", ""),
        "evidence": selected_citations,
        "latency": time.time() - start_time,
        "input_tokens": total_tokens_in,
        "output_tokens": total_tokens_out,
        "model_calls": model_calls,
    }


def main():
    parser = argparse.ArgumentParser(description="Run Method C: Structure-Aware Iterative Reader")
    parser.add_argument("--query", type=str, required=True, help="Question to ask")
    parser.add_argument("--top-k", type=int, default=TOP_K_RETRIEVAL, help="Number of initial candidate chunks to read")
    args = parser.parse_args()
    result = run_method_c(args.query, top_k=args.top_k)
    if result.get("success"):
        print("\n" + "=" * 50)
        print("FINAL ANSWER:")
        print(result["answer"])
        print("\nEVIDENCE:")
        for ev in result["evidence"]:
            print(f"  - {ev.get('file')} (Lines {ev.get('line_start')} to {ev.get('line_end')})")
        print("=" * 50)
        print(
            f"Metrics: Latency={result['latency']:.2f}s | "
            f"Tokens(in)={result.get('input_tokens')} | "
            f"Tokens(out)={result.get('output_tokens')} | "
            f"Model Calls={result.get('model_calls')}"
        )
    else:
        print(f"\nError occurred: {result.get('error')}")


if __name__ == "__main__":
    main()
