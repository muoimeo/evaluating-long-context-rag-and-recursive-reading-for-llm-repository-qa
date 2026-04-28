from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, List, Set

from config import INDEX_PATH
from reranker import is_docs_noise

# These defaults mirror the bounded iterative-reader settings in
# method_c_iterative.py. They live here so the deterministic tool layer can be
# reused without importing the full reader loop.
DEFAULT_MAX_CANDIDATES = 16
DEFAULT_MAX_TOOL_RESULTS = 3
DEFAULT_NEIGHBOR_WINDOW = 1

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

_INDEX_DOCS_CACHE: List[Dict[str, Any]] | None = None


def dedupe_text(items: List[Any], limit: int | None = None) -> List[str]:
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


def normalize_tokens(text: str) -> Set[str]:
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


def load_index_docs() -> List[Dict[str, Any]]:
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


def get_index_docs_cache() -> List[Dict[str, Any]] | None:
    return _INDEX_DOCS_CACHE


def set_index_docs_cache(docs: List[Dict[str, Any]] | None) -> None:
    global _INDEX_DOCS_CACHE
    _INDEX_DOCS_CACHE = docs


def doc_to_result(doc: Dict[str, Any], score: float = 0.01) -> Dict[str, Any]:
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


def is_specific_symbol(symbol: str) -> bool:
    clean = symbol.rstrip("()").strip()
    if len(clean) < 3 or clean.lower() in COMMON_SYMBOL_NOISE:
        return False
    if "." in clean:
        return True
    if clean[0].isupper():
        return True
    return "_" in clean or clean.endswith(("er", "or", "ion", "ity"))


def _reference_usage_score(content: str, symbol: str) -> float | None:
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


def definition_score_for_doc(
    doc: Dict[str, Any],
    symbol_terms: List[str],
    max_candidates: int = DEFAULT_MAX_CANDIDATES,
) -> float | None:
    content = doc.get("content", "")
    best: float | None = None
    for symbol in symbol_terms[:max_candidates]:
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


def matched_definition_symbols(
    doc: Dict[str, Any],
    symbol_terms: List[str],
    max_candidates: int = DEFAULT_MAX_CANDIDATES,
) -> List[str]:
    content = doc.get("content", "")
    matched = []
    for symbol in symbol_terms[:max_candidates]:
        symbol = symbol.rstrip("()")
        if not symbol:
            continue
        if re.search(rf"\b(?:async\s+def|def)\s+{re.escape(symbol)}\s*\(", content):
            matched.append(symbol)
        elif re.search(rf"\bclass\s+{re.escape(symbol)}\b", content):
            matched.append(symbol)
    return dedupe_text(matched)


def _tool_result(doc: Dict[str, Any], tool_name: str, score: float) -> Dict[str, Any]:
    result = doc_to_result(doc, score=score)
    result.setdefault("metadata", {})["tool"] = tool_name
    return result


def find_definition(symbol: str, limit: int = DEFAULT_MAX_TOOL_RESULTS) -> List[Dict[str, Any]]:
    symbol = symbol.rstrip("()").strip()
    if len(symbol) < 3:
        return []
    hits = []
    for doc in load_index_docs():
        content = doc.get("content", "")
        if re.search(rf"\b(?:async\s+def|def)\s+{re.escape(symbol)}\s*\(", content):
            hits.append(_tool_result(doc, "find_definition", 0.001))
        elif re.search(rf"\bclass\s+{re.escape(symbol)}\b", content):
            hits.append(_tool_result(doc, "find_definition", 0.001))
    hits.sort(key=lambda item: (item["score"], item["metadata"]["file"], item["metadata"]["chunk_index"]))
    return hits[:limit]


def find_references(symbol: str, limit: int = DEFAULT_MAX_TOOL_RESULTS) -> List[Dict[str, Any]]:
    symbol = symbol.rstrip("()").strip()
    if not is_specific_symbol(symbol):
        return []
    definition_pattern = rf"\b(?:async\s+def|def|class)\s+{re.escape(symbol)}\b"
    hits = []
    for doc in load_index_docs():
        file_path = doc.get("file", "")
        content = doc.get("content", "")
        if not _is_probably_source_file(file_path):
            continue
        usage_score = _reference_usage_score(content, symbol)
        if usage_score is not None and not re.search(definition_pattern, content):
            score = usage_score + (0.004 if is_docs_noise(file_path) else 0.0)
            hits.append(_tool_result(doc, "find_references", score))
    hits.sort(key=lambda item: (item["score"], item["metadata"]["file"], item["metadata"]["chunk_index"]))
    return hits[:limit]


def read_neighbors(
    file_path: str,
    line_start: int,
    window: int = DEFAULT_NEIGHBOR_WINDOW,
) -> List[Dict[str, Any]]:
    docs = [doc for doc in load_index_docs() if doc.get("file") == file_path]
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
    return [
        _tool_result(doc, "read_neighbors", 0.004 + abs(doc.get("line_start", 1) - line_start) / 1_000_000)
        for doc in selected
    ]


def search_attribute_reads(attr_name: str, limit: int = DEFAULT_MAX_TOOL_RESULTS) -> List[Dict[str, Any]]:
    attr_name = attr_name.strip().lstrip(".")
    if len(attr_name) < 3 or attr_name.lower() in COMMON_ATTRIBUTE_NOISE:
        return []
    pattern = rf"\.[ \t]*{re.escape(attr_name)}\b"
    hits = []
    for doc in load_index_docs():
        file_path = doc.get("file", "")
        if not _is_probably_source_file(file_path):
            continue
        content = doc.get("content", "")
        if re.search(pattern, content):
            score = 0.006 if is_docs_noise(file_path) else 0.0025
            hits.append(_tool_result(doc, "search_attribute_reads", score))
    hits.sort(key=lambda item: (item["score"], item["metadata"]["file"], item["metadata"]["chunk_index"]))
    return hits[:limit]


def search_constructor_calls(class_name: str, limit: int = DEFAULT_MAX_TOOL_RESULTS) -> List[Dict[str, Any]]:
    class_name = class_name.strip()
    if len(class_name) < 3 or not class_name[0].isupper():
        return []
    call_pattern = rf"\b{re.escape(class_name)}\s*\("
    definition_pattern = rf"\bclass\s+{re.escape(class_name)}\b"
    hits = []
    for doc in load_index_docs():
        file_path = doc.get("file", "")
        content = doc.get("content", "")
        if not _is_probably_source_file(file_path):
            continue
        if re.search(call_pattern, content) and not re.search(definition_pattern, content):
            score = 0.006 if is_docs_noise(file_path) else 0.002
            hits.append(_tool_result(doc, "search_constructor_calls", score))
    hits.sort(key=lambda item: (item["score"], item["metadata"]["file"], item["metadata"]["chunk_index"]))
    return hits[:limit]


def symbol_terms_from_text(text: str, limit: int = 12) -> List[str]:
    terms = []
    terms.extend(re.findall(r"`([^`]{3,80})`", text))
    terms.extend(re.findall(r"\b[A-Za-z_][A-Za-z0-9_]*\.[A-Za-z_][A-Za-z0-9_\.]*\b", text))
    terms.extend(re.findall(r"\b[A-Z][A-Za-z0-9_]{2,}\b", text))
    terms.extend(re.findall(r"\b[a-z_][A-Za-z0-9_]{3,}\(\)", text))
    return dedupe_text(terms, limit)


def attribute_terms_from_text(text: str, limit: int = 8) -> List[str]:
    attrs = []
    attrs.extend(re.findall(r"\.\s*([A-Za-z_][A-Za-z0-9_]{2,})\b", text))
    attrs.extend(re.findall(r"`([a-z_][A-Za-z0-9_]{2,})`", text))
    stop_tokens = normalize_tokens(text)
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
    return [item for item in dedupe_text(attrs, limit) if item.lower() not in noise]


def extract_structured_candidates(
    file_path: str,
    chunk_text: str,
    fact_texts: List[str] | None = None,
    max_candidates: int = DEFAULT_MAX_CANDIDATES,
) -> Dict[str, List[str]]:
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

    generic_calls = re.findall(r"\b(?:self\.)?([A-Za-z_][A-Za-z0-9_]*)\s*\(", chunk_text)
    call_noise = {
        "if", "for", "while", "return", "yield", "with", "assert", "raise",
        "len", "str", "int", "float", "bool", "list", "dict", "set", "tuple",
        "print", "range", "enumerate", "isinstance", "super",
    }
    symbols.extend([name for name in generic_calls if name not in call_noise and len(name) >= 3])

    for fact in fact_texts:
        symbols.extend(symbol_terms_from_text(fact))
        files.extend(re.findall(patterns["file_paths"], fact))

    return {
        "candidate_symbols": dedupe_text(symbols, max_candidates),
        "candidate_files": dedupe_text(files, max_candidates),
        "candidate_paths": dedupe_text(paths, max_candidates),
    }
