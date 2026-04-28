from __future__ import annotations

import re
from typing import Any, Dict, List

from method_c_tooling import dedupe_text, normalize_tokens

DEFAULT_MAX_VERIFIED_FACTS = 18
DEFAULT_MAX_EVIDENCE_SNIPPETS = 5
DEFAULT_FINAL_EVIDENCE_LIMITS = {"extract": 5, "trace": 7, "compare": 8}


def dataset_of_path(file_path: str) -> str:
    fastapi_root_files = {"pyproject.toml", "README.md", "CONTRIBUTING.md"}
    if (
        file_path.startswith("fastapi/")
        or "docs/" in file_path
        or file_path.startswith("docs_src/")
        or file_path in fastapi_root_files
    ):
        return "fastapi"
    return "lambda"


def question_mode(question: str) -> str:
    q = question.lower()
    if any(t in q for t in ["compare", "both ", "difference", "differences", "versus", "vs "]):
        return "compare"
    if any(t in q for t in ["trace", "flow", "through to", "from the", "call chain", "path"]):
        return "trace"
    dataflow_verbs = ["extract", "propagate", "derive", "generate", "register", "convert", "map", "store"]
    if q.startswith("how does") and any(v in q for v in dataflow_verbs):
        return "trace"
    return "extract"


def bounded_text(value: Any, max_chars: int = 300) -> str:
    text = str(value or "").strip()
    return text if len(text) <= max_chars else text[: max_chars - 3].rstrip() + "..."


def evidence_limit_for_mode(
    mode: str,
    final_limits: Dict[str, int] | None = None,
    default_limit: int = DEFAULT_MAX_EVIDENCE_SNIPPETS,
) -> int:
    final_limits = final_limits or DEFAULT_FINAL_EVIDENCE_LIMITS
    return final_limits.get(mode, default_limit)


def _facet_id(label: str) -> str:
    cleaned = re.sub(r"[^a-z0-9]+", "_", label.lower()).strip("_")
    return cleaned[:48] or "main_answer"


def _facet_keywords(label: str) -> List[str]:
    return sorted(normalize_tokens(label))


def _clean_facet_label(text: str) -> str:
    text = re.sub(r"^\s*(?:what|which|how\s+does|how\s+do|how|in)\s+", "", text, flags=re.I)
    text = re.sub(r"^\s*(?:the|a|an)\s+", "", text, flags=re.I)
    text = re.split(
        r"\b(?:are|is|does|do|work together|works together|from|inside|in)\b",
        text,
        maxsplit=1,
        flags=re.I,
    )[0]
    return re.sub(r"\s+", " ", text.strip(" ?.,:;`'\""))


def infer_required_facets(question: str, mode: str) -> List[Dict[str, Any]]:
    q = question.strip()
    q_lower = q.lower()

    if mode == "compare":
        facets: List[Dict[str, Any]] = []
        if "fastapi" in q_lower:
            facets.append({"id": "side_fastapi", "label": "FastAPI side", "keywords": ["fastapi"]})
        if any(term in q_lower for term in ["lambda", "aws", "sam", "api gateway"]):
            facets.append({"id": "side_lambda", "label": "AWS/Lambda side", "keywords": ["aws", "lambda"]})
        facets.append({
            "id": "comparison_relation",
            "label": "comparison relation",
            "keywords": ["compare", "difference", "relation", "versus", "whereas"],
        })
        return facets

    labels: List[str] = []
    if re.search(r"\s+and\s+", q, flags=re.I):
        parts = re.split(r"\s+and\s+", q, flags=re.I)
        if 2 <= len(parts) <= 4:
            cleaned_parts = [_clean_facet_label(part) for part in parts]
            tail_tokens = normalize_tokens(cleaned_parts[-1] if cleaned_parts else "")
            head_noun = next(
                (term for term in ["permissions", "actions", "routes", "methods", "fields", "attributes", "resources", "endpoints", "values"]
                 if term in tail_tokens),
                "",
            )
            for part in cleaned_parts:
                if not part:
                    continue
                if len(normalize_tokens(part)) == 1 and head_noun and head_noun not in part.lower():
                    part = f"{part} {head_noun}"
                labels.append(part)

    if not labels:
        labels = ["ordered implementation flow"] if mode == "trace" else ["main requested answer"]

    facets = []
    seen = set()
    for label in labels[:4]:
        fid = _facet_id(label)
        if fid in seen:
            continue
        seen.add(fid)
        facets.append({"id": fid, "label": label, "keywords": _facet_keywords(label)})
    return facets


def make_initial_state(question: str) -> Dict[str, Any]:
    q_type = question_mode(question)
    return {
        "verified_facts": [],
        "search_agenda": [{"missing_relation": question, "priority": "high"}],
        "candidate_symbols": [],
        "candidate_files": [],
        "candidate_paths": [],
        "tool_actions": [],
        "executed_tool_actions": [],
        "question_type": q_type,
        "required_facets": infer_required_facets(question, q_type),
        "answer_ready": False,
        "visited_files": [],
        "read_locations": [],
        "evidence_roles": {},
        "tiny_reasoning": "",
    }


def merge_search_agenda(state: Dict[str, Any], new_items: List[Dict[str, str]], max_items: int = 8) -> None:
    merged = list(state.get("search_agenda", []))
    seen = {
        (
            item.get("missing_relation", "").strip().lower(),
            item.get("priority", "medium"),
        )
        for item in merged
    }
    for item in new_items:
        relation = str(item.get("missing_relation", "")).strip()
        if not relation:
            continue
        priority = str(item.get("priority", "medium")).lower().strip()
        if priority not in {"high", "medium", "low"}:
            priority = "medium"
        key = (relation.lower(), priority)
        if key not in seen:
            seen.add(key)
            merged.append({"missing_relation": relation, "priority": priority})
    priority_rank = {"high": 0, "medium": 1, "low": 2}
    merged.sort(key=lambda item: (priority_rank.get(item.get("priority", "medium"), 1), item.get("missing_relation", "")))
    state["search_agenda"] = merged[:max_items]


def fact_key(fact: Dict[str, Any]) -> tuple:
    return (
        fact.get("fact", "").strip().lower(),
        fact.get("supports_question_part", "").strip().lower(),
        fact.get("support_type", "context").strip().lower(),
    )


def fact_resolves_relation(fact: Dict[str, Any], relation: str) -> bool:
    relation_tokens = normalize_tokens(relation)
    if not relation_tokens:
        return False
    fact_blob = f"{fact.get('fact', '')} {fact.get('supports_question_part', '')}"
    fact_tokens = normalize_tokens(fact_blob)
    overlap = len(relation_tokens & fact_tokens)
    return overlap >= max(1, min(2, len(relation_tokens)))


def prune_resolved_agenda(state: Dict[str, Any], new_facts: List[Dict[str, Any]]) -> None:
    if not new_facts:
        return
    remaining = []
    for item in state.get("search_agenda", []):
        relation = item.get("missing_relation", "")
        if any(fact_resolves_relation(fact, relation) for fact in new_facts):
            continue
        remaining.append(item)
    state["search_agenda"] = remaining


def fact_matches_facet(
    fact: Dict[str, Any],
    facet: Dict[str, Any],
    evidence_by_id: Dict[str, Dict[str, Any]] | None = None,
    file_path: str | None = None,
) -> bool:
    blob = f"{fact.get('fact', '')} {fact.get('supports_question_part', '')}"
    blob_tokens = normalize_tokens(blob)
    facet_id = facet.get("id")

    if facet_id == "side_fastapi":
        evidence_files = [
            (evidence_by_id or {}).get(evidence_id, {}).get("file", "")
            for evidence_id in fact.get("evidence_ids", [])
        ]
        if file_path and dataset_of_path(file_path) == "fastapi":
            return True
        return "fastapi" in blob.lower() or any(dataset_of_path(path) == "fastapi" for path in evidence_files)
    if facet_id == "side_lambda":
        evidence_files = [
            (evidence_by_id or {}).get(evidence_id, {}).get("file", "")
            for evidence_id in fact.get("evidence_ids", [])
        ]
        if file_path and dataset_of_path(file_path) == "lambda":
            return True
        return any(term in blob.lower() for term in ["aws", "lambda", "sam"]) or any(
            dataset_of_path(path) == "lambda" for path in evidence_files
        )
    if facet_id == "comparison_relation":
        return has_comparison_relation_fact([fact])
    if facet_id == "main_answer":
        return fact.get("support_type") in {"answer", "bridge"}
    if facet_id == "ordered_implementation_flow":
        return fact.get("support_type") in {"answer", "bridge"}

    keywords = set(facet.get("keywords", []) or [])
    if not keywords:
        return False
    overlap = len(keywords & blob_tokens)
    return overlap >= max(1, min(2, len(keywords)))


def covered_facet_ids(
    fact: Dict[str, Any],
    facets: List[Dict[str, Any]],
    evidence_by_id: Dict[str, Dict[str, Any]] | None = None,
    file_path: str | None = None,
) -> List[str]:
    return [
        facet["id"]
        for facet in facets
        if fact_matches_facet(fact, facet, evidence_by_id=evidence_by_id, file_path=file_path)
    ]


def trim_verified_facts(state: Dict[str, Any], max_verified_facts: int = DEFAULT_MAX_VERIFIED_FACTS) -> None:
    facts = state.get("verified_facts", [])
    if len(facts) <= max_verified_facts:
        return

    support_rank = {"answer": 0, "bridge": 1, "context": 2}
    role_rank = {"direct": 0, "bridge": 1, "fallback": 2, "noise": 3}
    evidence_roles = state.get("evidence_roles", {})

    ranked = []
    for idx, fact in enumerate(facts):
        evidence_ids = fact.get("evidence_ids", []) or []
        best_role = min(
            (role_rank.get(evidence_roles.get(evidence_id, "noise"), 3) for evidence_id in evidence_ids),
            default=3,
        )
        ranked.append((
            support_rank.get(fact.get("support_type", "context"), 2),
            best_role,
            -len(evidence_ids),
            -idx,
            idx,
            fact,
        ))

    selected_indexes = set()
    facets = state.get("required_facets", [])
    for facet in facets:
        facet_candidates = [
            item for item in ranked if facet.get("id") in (item[5].get("covered_facets") or [])
        ]
        if facet_candidates:
            selected_indexes.add(sorted(facet_candidates)[0][4])

    for item in sorted(ranked):
        if len(selected_indexes) >= max_verified_facts:
            break
        selected_indexes.add(item[4])

    kept = [facts[idx] for idx in sorted(selected_indexes)]
    state["verified_facts"] = kept[:max_verified_facts]


def build_answer_plan(
    state: Dict[str, Any],
    evidence_by_id: Dict[str, Dict[str, Any]],
    max_facts_per_facet: int = 4,
) -> Dict[str, Any]:
    facets = state.get("required_facets", []) or [{"id": "main_answer", "label": "main requested answer", "keywords": []}]
    plan_facets = []
    missing = []
    for facet in facets:
        matches = []
        for fact in state.get("verified_facts", []):
            if facet.get("id") in (fact.get("covered_facets") or []) or fact_matches_facet(
                fact, facet, evidence_by_id=evidence_by_id
            ):
                matches.append(fact)
        if not matches:
            missing.append(facet.get("label", facet.get("id")))
        plan_facets.append({
            "id": facet.get("id"),
            "label": facet.get("label", facet.get("id")),
            "facts": matches[:max_facts_per_facet],
        })
    return {"facets": plan_facets, "missing_facets": missing}


def evidence_files_for_facts(state: Dict[str, Any], evidence_by_id: Dict[str, Dict[str, Any]]) -> List[str]:
    files = []
    for fact in state.get("verified_facts", []):
        for evidence_id in fact.get("evidence_ids", []):
            item = evidence_by_id.get(evidence_id)
            if item:
                files.append(item["file"])
    return dedupe_text(files)


def facts_by_support_type(state: Dict[str, Any], *support_types: str) -> List[Dict[str, Any]]:
    wanted = set(support_types)
    return [fact for fact in state.get("verified_facts", []) if fact.get("support_type", "context") in wanted]


def fact_datasets(facts: List[Dict[str, Any]], evidence_by_id: Dict[str, Dict[str, Any]]) -> set:
    datasets = set()
    for fact in facts:
        for evidence_id in fact.get("evidence_ids", []):
            item = evidence_by_id.get(evidence_id)
            if item:
                datasets.add(dataset_of_path(item["file"]))
    return datasets


def has_comparison_relation_fact(facts: List[Dict[str, Any]]) -> bool:
    relation_terms = {
        "compare", "comparison", "difference", "different", "whereas", "while",
        "but", "unlike", "both", "side", "versus", "vs", "relation",
    }
    for fact in facts:
        blob = f"{fact.get('fact', '')} {fact.get('supports_question_part', '')}".lower()
        if relation_terms & set(blob.replace("/", " ").replace("-", " ").split()):
            return True
    return False


def covers_required_subparts(state: Dict[str, Any], evidence_by_id: Dict[str, Dict[str, Any]]) -> bool:
    answer_facts = facts_by_support_type(state, "answer")
    bridge_facts = facts_by_support_type(state, "bridge")
    decisive_facts = answer_facts + bridge_facts
    if not decisive_facts:
        return False
    mode = state.get("question_type", "extract")
    answer_plan = build_answer_plan(state, evidence_by_id)
    if answer_plan["missing_facets"]:
        return False
    evidence_files = evidence_files_for_facts({"verified_facts": decisive_facts}, evidence_by_id)
    if mode == "compare":
        return (
            len(fact_datasets(decisive_facts, evidence_by_id)) >= 2
            and len(answer_facts) >= 2
            and has_comparison_relation_fact(answer_facts)
        )
    if mode == "trace":
        return len(decisive_facts) >= 2 and (len(evidence_files) >= 2 or len(bridge_facts) >= 2)
    required_facets = state.get("required_facets", [])
    if len(required_facets) <= 1:
        return len(answer_facts) >= 1
    return all(
        any(
            facet.get("id") in (fact.get("covered_facets") or [])
            or fact_matches_facet(fact, facet, evidence_by_id=evidence_by_id)
            for fact in answer_facts
        )
        for facet in required_facets
    )


def should_stop(
    state: Dict[str, Any],
    evidence_by_id: Dict[str, Dict[str, Any]],
    no_progress_rounds: int,
    retrieval_empty: bool = False,
) -> bool:
    coverage_ready = covers_required_subparts(state, evidence_by_id)
    unresolved_high = any(item.get("priority") == "high" for item in state.get("search_agenda", []))
    if coverage_ready and (state.get("answer_ready") or not unresolved_high):
        return True
    if no_progress_rounds >= 2:
        return True
    if retrieval_empty and state.get("verified_facts"):
        return True
    return False
