import argparse
import json
import os
import re
import time
from typing import Any, Dict, List, Set

from openai import OpenAI

from config import API_KEY, INDEX_PATH, MODEL_NAME, OLLAMA_BASE_URL, TOP_K_RETRIEVAL
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
COMMON_SYMBOL_NOISE = {
    "app", "api", "body", "call", "class", "code", "data", "dict", "error",
    "event", "field", "file", "func", "function", "handler", "item", "items",
    "key", "list", "method", "methods", "model", "name", "none", "object",
    "param", "params", "path", "request", "response", "result", "route",
    "routes", "self", "type", "value", "values",
}
COMMON_ATTRIBUTE_NOISE = {
    "api", "app", "body", "class", "code", "data", "event", "field", "file",
    "function", "handler", "item", "items", "key", "method", "model", "name",
    "object", "open", "openapi", "param", "params", "path", "request",
    "response", "result", "route", "routes", "type", "value", "values",
}

VALID_ROLES = {"direct", "bridge", "noise"}
VALID_PRIORITIES = {"high", "medium", "low"}
VALID_SUPPORT_TYPES = {"answer", "bridge", "context"}
PRIORITY_RANK = {"high": 0, "medium": 1, "low": 2}
_INDEX_DOCS_CACHE: List[Dict[str, Any]] | None = None


def _dataset_of_path(file_path: str) -> str:
    fastapi_root_files = {"pyproject.toml", "README.md", "CONTRIBUTING.md"}
    if (
        file_path.startswith("fastapi/")
        or "docs/" in file_path
        or file_path.startswith("docs_src/")
        or file_path in fastapi_root_files
    ):
        return "fastapi"
    return "lambda"


def _question_mode(question: str) -> str:
    q = question.lower()
    if any(t in q for t in ["compare", "both ", "difference", "differences", "versus", "vs "]):
        return "compare"
    if any(t in q for t in ["trace", "flow", "through to", "from the", "call chain", "path"]):
        return "trace"
    # Some questions are nominally extractive but ask how information moves
    # through implementation steps. Treat those as trace-like without seeding
    # repo-specific files or symbols.
    dataflow_verbs = ["extract", "propagate", "derive", "generate", "register", "convert", "map", "store"]
    if q.startswith("how does") and any(v in q for v in dataflow_verbs):
        return "trace"
    return "extract"


def make_initial_state(question: str) -> Dict[str, Any]:
    question_type = _question_mode(question)
    return {
        "verified_facts": [],
        "search_agenda": [{"missing_relation": question, "priority": "high"}],
        "candidate_symbols": [],
        "candidate_files": [],
        "candidate_paths": [],
        "tool_actions": [],
        "executed_tool_actions": [],
        "question_type": question_type,
        "required_facets": infer_required_facets(question, question_type),
        "answer_ready": False,
        "visited_files": [],
        "read_locations": [],
        "evidence_roles": {},
        "tiny_reasoning": "",
    }


def _dedupe(items: List[Any], limit: int | None = None) -> List[str]:
    out, seen = [], set()
    for item in items:
        value = str(item or "").strip().strip("`'\"")
        if not value or value.lower() in {"true", "false", "none", "null"}:
            continue
        key = value.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(value)
        if limit is not None and len(out) >= limit:
            break
    return out


def _bounded_text(value: Any, max_chars: int = 300) -> str:
    text = str(value or "").strip()
    return text if len(text) <= max_chars else text[: max_chars - 3].rstrip() + "..."


def _evidence_limit_for_mode(mode: str) -> int:
    return FINAL_EVIDENCE_LIMITS.get(mode, MAX_EVIDENCE_SNIPPETS)


def _facet_id(label: str) -> str:
    cleaned = re.sub(r"[^a-z0-9]+", "_", label.lower()).strip("_")
    return cleaned[:48] or "main_answer"


def _facet_keywords(label: str) -> List[str]:
    return sorted(_normalize_tokens(label))


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
    """Infer coarse answer facets from the question without using benchmark labels."""
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
            tail_tokens = _normalize_tokens(cleaned_parts[-1] if cleaned_parts else "")
            head_noun = next(
                (term for term in ["permissions", "actions", "routes", "methods", "fields", "attributes", "resources", "endpoints", "values"]
                 if term in tail_tokens),
                "",
            )
            for part in cleaned_parts:
                if not part:
                    continue
                if len(_normalize_tokens(part)) == 1 and head_noun and head_noun not in part.lower():
                    part = f"{part} {head_noun}"
                labels.append(part)

    if not labels:
        if mode == "trace":
            labels = ["ordered implementation flow"]
        else:
            labels = ["main requested answer"]

    facets = []
    seen = set()
    for label in labels[:4]:
        fid = _facet_id(label)
        if fid in seen:
            continue
        seen.add(fid)
        facets.append({"id": fid, "label": label, "keywords": _facet_keywords(label)})
    return facets


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


def _normalize_tokens(text: str) -> set:
    stop = frozenset({
        "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
        "have", "has", "had", "do", "does", "did", "will", "would", "could",
        "should", "may", "might", "can", "shall", "to", "of", "in", "for",
        "on", "with", "at", "by", "from", "as", "into", "through", "during",
        "before", "after", "and", "but", "or", "nor", "not", "so", "yet",
        "both", "either", "each", "every", "all", "any", "few", "more", "most",
        "other", "some", "such", "no", "only", "own", "same", "than", "too",
        "very", "just", "about", "above", "how", "what", "which", "who",
        "whom", "this", "that", "these", "those", "it", "its",
    })
    text = re.sub(r"([a-z])([A-Z])", r"\1 \2", text)
    return set(text.replace("_", " ").replace("-", " ").lower().split()) - stop


def _load_index_docs() -> List[Dict[str, Any]]:
    global _INDEX_DOCS_CACHE
    if _INDEX_DOCS_CACHE is not None:
        return _INDEX_DOCS_CACHE
    if not os.path.exists(INDEX_PATH):
        _INDEX_DOCS_CACHE = []
        return _INDEX_DOCS_CACHE
    with open(INDEX_PATH, "r", encoding="utf-8") as f:
        payload = json.load(f)
    _INDEX_DOCS_CACHE = payload if isinstance(payload, list) else []
    return _INDEX_DOCS_CACHE


def _doc_to_result(doc: Dict[str, Any], score: float = 0.01) -> Dict[str, Any]:
    return {
        "id": f"{doc['file']}_{doc['chunk_index']}",
        "score": score,
        "document": doc.get("content", ""),
        "metadata": {
            "file": doc["file"],
            "chunk_index": doc.get("chunk_index", 0),
            "total_chunks": doc.get("total_chunks", 1),
            "language": doc.get("language", ""),
            "line_start": doc.get("line_start", 1),
            "line_end": doc.get("line_end", 1),
        },
    }


def _is_probably_source_file(file_path: str) -> bool:
    return file_path.endswith((".py", ".js", ".ts", ".go", ".java", ".cs", ".yml", ".yaml", ".json"))


def _is_specific_symbol(symbol: str) -> bool:
    clean = symbol.rstrip("()").strip()
    if len(clean) < 3 or clean.lower() in COMMON_SYMBOL_NOISE:
        return False
    if "." in clean:
        return True
    if clean[0].isupper():
        return True
    return "_" in clean or clean.endswith(("er", "or", "ion", "ity"))


def _reference_usage_score(content: str, symbol: str) -> float | None:
    """Rank references by likely code usage: call > attribute/member > argument > plain."""
    sym = re.escape(symbol)
    if re.search(rf"\b(?:self\.)?{sym}\s*\(", content):
        return 0.0015
    if re.search(rf"\.{sym}\b|\b{sym}\.", content):
        return 0.002
    if re.search(rf"\b{sym}\s*=", content) or re.search(rf"=\s*{sym}\b", content):
        return 0.0025
    if re.search(rf"\b{sym}\b", content):
        return 0.005
    return None


def _definition_score_for_doc(doc: Dict[str, Any], symbol_terms: List[str]) -> float | None:
    content = doc.get("content", "")
    best: float | None = None
    for symbol in symbol_terms[:MAX_CANDIDATES]:
        symbol = symbol.rstrip("()")
        if not symbol:
            continue
        if re.search(rf"\b(?:async\s+def|def)\s+{re.escape(symbol)}\s*\(", content):
            score = 0.001
            best = score if best is None else min(best, score)
        elif re.search(rf"\bclass\s+{re.escape(symbol)}\b", content):
            score = 0.001
            best = score if best is None else min(best, score)
    return best


def _matched_definition_symbols(doc: Dict[str, Any], symbol_terms: List[str]) -> List[str]:
    content = doc.get("content", "")
    matched = []
    for symbol in symbol_terms[:MAX_CANDIDATES]:
        symbol = symbol.rstrip("()")
        if not symbol:
            continue
        if re.search(rf"\b(?:async\s+def|def)\s+{re.escape(symbol)}\s*\(", content):
            matched.append(symbol)
        elif re.search(rf"\bclass\s+{re.escape(symbol)}\b", content):
            matched.append(symbol)
    return _dedupe(matched)


def _tool_result(doc: Dict[str, Any], tool_name: str, score: float) -> Dict[str, Any]:
    result = _doc_to_result(doc, score=score)
    result.setdefault("metadata", {})["tool"] = tool_name
    return result


def find_definition(symbol: str, limit: int = MAX_TOOL_RESULTS_PER_ACTION) -> List[Dict[str, Any]]:
    """Find chunks defining a function, async function, or class by exact symbol name."""
    symbol = symbol.rstrip("()").strip()
    if len(symbol) < 3:
        return []
    hits = []
    for doc in _load_index_docs():
        content = doc.get("content", "")
        if re.search(rf"\b(?:async\s+def|def)\s+{re.escape(symbol)}\s*\(", content):
            hits.append(_tool_result(doc, "find_definition", 0.001))
        elif re.search(rf"\bclass\s+{re.escape(symbol)}\b", content):
            hits.append(_tool_result(doc, "find_definition", 0.001))
    hits.sort(key=lambda item: (item["score"], item["metadata"]["file"], item["metadata"]["chunk_index"]))
    return hits[:limit]


def find_references(symbol: str, limit: int = MAX_TOOL_RESULTS_PER_ACTION) -> List[Dict[str, Any]]:
    """Find chunks that reference a symbol without being its definition."""
    symbol = symbol.rstrip("()").strip()
    if not _is_specific_symbol(symbol):
        return []
    definition_pattern = rf"\b(?:async\s+def|def|class)\s+{re.escape(symbol)}\b"
    hits = []
    for doc in _load_index_docs():
        file_path = doc.get("file", "")
        content = doc.get("content", "")
        if not _is_probably_source_file(file_path):
            continue
        usage_score = _reference_usage_score(content, symbol)
        if usage_score is not None and not re.search(definition_pattern, content):
            score = usage_score + (0.004 if _is_docs_noise(file_path) else 0.0)
            hits.append(_tool_result(doc, "find_references", score))
    hits.sort(key=lambda item: (item["score"], item["metadata"]["file"], item["metadata"]["chunk_index"]))
    return hits[:limit]


def read_neighbors(file_path: str, line_start: int, window: int = NEIGHBOR_WINDOW) -> List[Dict[str, Any]]:
    """Return adjacent indexed chunks around a known file/line position."""
    docs = [doc for doc in _load_index_docs() if doc.get("file") == file_path]
    if not docs:
        return []
    docs.sort(key=lambda doc: doc.get("line_start", 1))
    center = None
    for idx, doc in enumerate(docs):
        if doc.get("line_start", 1) <= line_start <= doc.get("line_end", doc.get("line_start", 1)):
            center = idx
            break
    if center is None:
        center = min(range(len(docs)), key=lambda idx: abs(docs[idx].get("line_start", 1) - line_start))
    selected = docs[max(0, center - window): min(len(docs), center + window + 1)]
    return [_tool_result(doc, "read_neighbors", 0.004 + abs(doc.get("line_start", 1) - line_start) / 1_000_000) for doc in selected]


def search_attribute_reads(attr_name: str, limit: int = MAX_TOOL_RESULTS_PER_ACTION) -> List[Dict[str, Any]]:
    """Find chunks reading object attributes such as route.tags or self.summary."""
    attr_name = attr_name.strip().lstrip(".")
    if len(attr_name) < 3 or attr_name.lower() in COMMON_ATTRIBUTE_NOISE:
        return []
    pattern = rf"\.[ \t]*{re.escape(attr_name)}\b"
    hits = []
    for doc in _load_index_docs():
        file_path = doc.get("file", "")
        if not _is_probably_source_file(file_path):
            continue
        content = doc.get("content", "")
        if re.search(pattern, content):
            score = 0.006 if _is_docs_noise(file_path) else 0.0025
            hits.append(_tool_result(doc, "search_attribute_reads", score))
    hits.sort(key=lambda item: (item["score"], item["metadata"]["file"], item["metadata"]["chunk_index"]))
    return hits[:limit]


def search_constructor_calls(class_name: str, limit: int = MAX_TOOL_RESULTS_PER_ACTION) -> List[Dict[str, Any]]:
    """Find chunks constructing a class by exact constructor call, excluding the class definition."""
    class_name = class_name.strip()
    if len(class_name) < 3 or not class_name[0].isupper():
        return []
    call_pattern = rf"\b{re.escape(class_name)}\s*\("
    definition_pattern = rf"\bclass\s+{re.escape(class_name)}\b"
    hits = []
    for doc in _load_index_docs():
        file_path = doc.get("file", "")
        content = doc.get("content", "")
        if not _is_probably_source_file(file_path):
            continue
        if re.search(call_pattern, content) and not re.search(definition_pattern, content):
            score = 0.006 if _is_docs_noise(file_path) else 0.002
            hits.append(_tool_result(doc, "search_constructor_calls", score))
    hits.sort(key=lambda item: (item["score"], item["metadata"]["file"], item["metadata"]["chunk_index"]))
    return hits[:limit]


def _symbol_terms_from_text(text: str) -> List[str]:
    terms = []
    terms.extend(re.findall(r"`([^`]{3,80})`", text))
    terms.extend(re.findall(r"\b[A-Za-z_][A-Za-z0-9_]*\.[A-Za-z_][A-Za-z0-9_\.]*\b", text))
    terms.extend(re.findall(r"\b[A-Z][A-Za-z0-9_]{2,}\b", text))
    terms.extend(re.findall(r"\b[a-z_][A-Za-z0-9_]{3,}\(\)", text))
    return _dedupe(terms, 12)


def _attribute_terms_from_text(text: str) -> List[str]:
    attrs = []
    attrs.extend(re.findall(r"\.\s*([A-Za-z_][A-Za-z0-9_]{2,})\b", text))
    attrs.extend(re.findall(r"`([a-z_][A-Za-z0-9_]{2,})`", text))
    # In questions, comma-separated implementation nouns often name attributes
    # being traced, e.g. "tags, summary, and description". Keep only a few.
    stop_tokens = _normalize_tokens(text)
    for token in re.findall(r"[A-Za-z_][A-Za-z0-9_]{2,}", re.sub(r"([a-z])([A-Z])", r"\1 \2", text)):
        token = token.lower()
        if token not in stop_tokens:
            continue
        if 3 <= len(token) <= 32 and re.match(r"^[a-z_][a-z0-9_]*$", token):
            attrs.append(token)
    noise = COMMON_ATTRIBUTE_NOISE | {
        "api", "open", "openapi", "fastapi", "lambda", "route", "routes", "request", "response", "metadata",
        "documentation", "decorator", "decorators", "implementation", "question",
        "function", "class", "method", "methods", "file", "files", "code",
        "trace", "flow", "compare", "comparison", "extract", "extraction",
    }
    return [item for item in _dedupe(attrs, 8) if item.lower() not in noise]


def extract_structured_candidates(file_path: str, chunk_text: str, fact_texts: List[str] | None = None) -> Dict[str, List[str]]:
    fact_texts = fact_texts or []
    symbols: List[str] = []
    files: List[str] = [file_path]
    paths: List[str] = []
    patterns = {
        "imports": r"^\s*(?:from|import)\s+([A-Za-z_][A-Za-z0-9_\.]*)",
        "defs": r"\b(?:async\s+def|def)\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(",
        "classes": r"\bclass\s+([A-Za-z_][A-Za-z0-9_]*)\b",
        "method_calls": r"\.([A-Za-z_][A-Za-z0-9_]*)\s*\(",
        "decorators": r"^\s*@([A-Za-z_][A-Za-z0-9_\.]*)",
        "dotted": r"\b[A-Za-z_][A-Za-z0-9_]*\.[A-Za-z_][A-Za-z0-9_\.]*\b",
        "file_paths": r"\b[A-Za-z0-9_./-]+\.(?:py|md|yml|yaml|json|js|ts|go|java|cs)\b",
        "config_keys": r"^\s*([A-Za-z_][A-Za-z0-9_-]{2,})\s*:",
    }
    for key, pattern in patterns.items():
        matches = re.findall(pattern, chunk_text, flags=re.MULTILINE)
        if key == "file_paths":
            files.extend(matches)
        elif key == "imports":
            paths.extend(matches)
            symbols.extend([m.rsplit(".", 1)[-1] for m in matches])
        else:
            symbols.extend(matches)

    # Generic call-site leads: if a chunk calls foo(...) or self.foo(...),
    # the next useful read is often the definition/reference of foo. Keep this
    # lexical and bounded; do not seed benchmark-specific chain symbols.
    generic_calls = re.findall(r"\b(?:self\.)?([A-Za-z_][A-Za-z0-9_]*)\s*\(", chunk_text)
    call_noise = {
        "if", "for", "while", "return", "yield", "with", "assert", "raise",
        "len", "str", "int", "float", "bool", "list", "dict", "set", "tuple",
        "print", "range", "enumerate", "isinstance", "super",
    }
    symbols.extend([name for name in generic_calls if name not in call_noise and len(name) >= 3])

    for fact in fact_texts:
        symbols.extend(_symbol_terms_from_text(fact))
        files.extend(re.findall(patterns["file_paths"], fact))
    return {
        "candidate_symbols": _dedupe(symbols, MAX_CANDIDATES),
        "candidate_files": _dedupe(files, MAX_CANDIDATES),
        "candidate_paths": _dedupe(paths, MAX_CANDIDATES),
    }


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


def _merge_search_agenda(state: Dict[str, Any], new_items: List[Dict[str, str]]) -> None:
    by_key = {item.get("missing_relation", "").lower(): item for item in state.get("search_agenda", [])}
    for item in new_items:
        relation = item.get("missing_relation", "").strip()
        if not relation:
            continue
        key = relation.lower()
        priority = item.get("priority", "medium")
        old = by_key.get(key)
        if old is None or PRIORITY_RANK.get(priority, 2) < PRIORITY_RANK.get(old.get("priority", "low"), 2):
            by_key[key] = {"missing_relation": relation, "priority": priority}
    state["search_agenda"] = sorted(by_key.values(), key=lambda x: PRIORITY_RANK.get(x.get("priority", "low"), 2))[:MAX_AGENDA_ITEMS]


def _fact_key(fact: Dict[str, Any]) -> tuple:
    return (
        str(fact.get("fact", "")).lower(),
        str(fact.get("supports_question_part", "")).lower(),
        str(fact.get("support_type", "context")).lower(),
    )


def _fact_resolves_relation(fact: Dict[str, Any], relation: str) -> bool:
    if fact.get("support_type") == "context":
        return False
    relation_tokens = _normalize_tokens(relation)
    if not relation_tokens:
        return False
    fact_blob = f"{fact.get('fact', '')} {fact.get('supports_question_part', '')}"
    fact_tokens = _normalize_tokens(fact_blob)
    if not fact_tokens:
        return False
    overlap = len(relation_tokens & fact_tokens)
    return overlap >= 3 and overlap / max(1, min(len(relation_tokens), len(fact_tokens))) >= 0.35


def _prune_resolved_agenda(state: Dict[str, Any], new_facts: List[Dict[str, Any]]) -> None:
    if not new_facts or not state.get("search_agenda"):
        return
    remaining = []
    for item in state["search_agenda"]:
        relation = item.get("missing_relation", "")
        if any(_fact_resolves_relation(fact, relation) for fact in new_facts):
            continue
        remaining.append(item)
    state["search_agenda"] = remaining[:MAX_AGENDA_ITEMS]


def _fact_matches_facet(
    fact: Dict[str, Any],
    facet: Dict[str, Any],
    evidence_by_id: Dict[str, Dict[str, Any]] | None = None,
    file_path: str | None = None,
) -> bool:
    facet_id = facet.get("id", "")
    blob = f"{fact.get('fact', '')} {fact.get('supports_question_part', '')}"
    blob_tokens = _normalize_tokens(blob)
    evidence_files = []
    if file_path:
        evidence_files.append(file_path)
    if evidence_by_id:
        for evidence_id in fact.get("evidence_ids", []):
            item = evidence_by_id.get(evidence_id)
            if item:
                evidence_files.append(item.get("file", ""))

    if facet_id == "side_fastapi":
        return "fastapi" in blob.lower() or any(_dataset_of_path(path) == "fastapi" for path in evidence_files)
    if facet_id == "side_lambda":
        return any(term in blob.lower() for term in ["aws", "lambda", "sam"]) or any(
            _dataset_of_path(path) == "lambda" for path in evidence_files
        )
    if facet_id == "comparison_relation":
        return _has_comparison_relation_fact([fact])
    if facet_id == "main_answer":
        return fact.get("support_type") in {"answer", "bridge"}
    if facet_id == "ordered_implementation_flow":
        return fact.get("support_type") in {"answer", "bridge"}

    keywords = set(facet.get("keywords", []) or [])
    if not keywords:
        return False
    overlap = len(keywords & blob_tokens)
    return overlap >= max(1, min(2, len(keywords)))


def _covered_facet_ids(
    fact: Dict[str, Any],
    facets: List[Dict[str, Any]],
    evidence_by_id: Dict[str, Dict[str, Any]] | None = None,
    file_path: str | None = None,
) -> List[str]:
    return [
        facet["id"]
        for facet in facets
        if _fact_matches_facet(fact, facet, evidence_by_id=evidence_by_id, file_path=file_path)
    ]


def _trim_verified_facts(state: Dict[str, Any]) -> None:
    """Keep decisive facts first instead of blindly keeping the most recent facts."""
    facts = state.get("verified_facts", [])
    if len(facts) <= MAX_VERIFIED_FACTS:
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
            -idx,  # tie-break toward recent facts without discarding early answer facts first
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
        if len(selected_indexes) >= MAX_VERIFIED_FACTS:
            break
        selected_indexes.add(item[4])

    kept = [facts[idx] for idx in sorted(selected_indexes)]
    state["verified_facts"] = kept[:MAX_VERIFIED_FACTS]


def build_answer_plan(
    state: Dict[str, Any],
    evidence_by_id: Dict[str, Dict[str, Any]],
    max_facts_per_facet: int = 4,
) -> Dict[str, Any]:
    facets = state.get("required_facets", []) or [{"id": "main_answer", "label": "main requested answer", "keywords": []}]
    decisive = [
        fact for fact in state.get("verified_facts", [])
        if fact.get("support_type", "context") in {"answer", "bridge"}
    ]
    plan_facets = []
    missing = []
    for facet in facets:
        matches = [
            fact for fact in decisive
            if facet.get("id") in (fact.get("covered_facets") or [])
            or _fact_matches_facet(fact, facet, evidence_by_id=evidence_by_id)
        ]
        if not matches:
            missing.append(facet.get("label", facet.get("id", "unknown facet")))
        plan_facets.append({
            "id": facet.get("id"),
            "label": facet.get("label"),
            "facts": matches[:max_facts_per_facet],
        })
    return {"facets": plan_facets, "missing_facets": missing}


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


def _evidence_files_for_facts(state: Dict[str, Any], evidence_by_id: Dict[str, Dict[str, Any]]) -> List[str]:
    files = []
    for fact in state.get("verified_facts", []):
        for evidence_id in fact.get("evidence_ids", []):
            item = evidence_by_id.get(evidence_id)
            if item:
                files.append(item["file"])
    return _dedupe(files)


def _facts_by_support_type(state: Dict[str, Any], *support_types: str) -> List[Dict[str, Any]]:
    wanted = set(support_types)
    return [fact for fact in state.get("verified_facts", []) if fact.get("support_type", "context") in wanted]


def _fact_datasets(facts: List[Dict[str, Any]], evidence_by_id: Dict[str, Dict[str, Any]]) -> set:
    datasets = set()
    for fact in facts:
        for evidence_id in fact.get("evidence_ids", []):
            item = evidence_by_id.get(evidence_id)
            if item:
                datasets.add(_dataset_of_path(item["file"]))
    return datasets


def _has_comparison_relation_fact(facts: List[Dict[str, Any]]) -> bool:
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
    answer_facts = _facts_by_support_type(state, "answer")
    bridge_facts = _facts_by_support_type(state, "bridge")
    decisive_facts = answer_facts + bridge_facts
    if not decisive_facts:
        return False
    mode = state.get("question_type", "extract")
    answer_plan = build_answer_plan(state, evidence_by_id)
    if answer_plan["missing_facets"]:
        return False
    evidence_files = _evidence_files_for_facts({"verified_facts": decisive_facts}, evidence_by_id)
    if mode == "compare":
        return (
            len(_fact_datasets(decisive_facts, evidence_by_id)) >= 2
            and len(answer_facts) >= 2
            and _has_comparison_relation_fact(answer_facts)
        )
    if mode == "trace":
        return len(decisive_facts) >= 2 and (len(evidence_files) >= 2 or len(bridge_facts) >= 2)
    required_facets = state.get("required_facets", [])
    if len(required_facets) <= 1:
        return len(answer_facts) >= 1
    return all(
        any(
            facet.get("id") in (fact.get("covered_facets") or [])
            or _fact_matches_facet(fact, facet, evidence_by_id=evidence_by_id)
            for fact in answer_facts
        )
        for facet in required_facets
    )


def should_stop(state: Dict[str, Any], evidence_by_id: Dict[str, Dict[str, Any]],
                no_progress_rounds: int, retrieval_empty: bool = False) -> bool:
    coverage_ready = covers_required_subparts(state, evidence_by_id)
    unresolved_high = any(item.get("priority") == "high" for item in state.get("search_agenda", []))
    # answer_ready is now a secondary gate: it can confirm stopping only after
    # typed evidence coverage exists. It can never stop the loop by itself.
    if coverage_ready and (state.get("answer_ready") or not unresolved_high):
        return True
    if no_progress_rounds >= 2:
        return True
    if retrieval_empty and state.get("verified_facts"):
        return True
    return False


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
