"""
Controlled context-rot evaluation for repository QA.

This module isolates context-rot effects by constructing the same bounded context
pool for every method at each question/level. Method A receives that pool as a
direct prompt context. Method B and Method C are run with their retrieval layer
temporarily restricted to the same pool, so increasing context size/noise is the
controlled variable.

Important fairness note:
- L0 is built from gold-overlapping chunks when ground-truth evidence exists.
  Retrieval is only used to fill the clean core if fewer than RELEVANT_K chunks
  overlap gold evidence. This isolates context-rot from first-pass retrieval
  failure.
- Method B and Method C share the same source/docs filtering rule through
  reranker.py. Use --disable-c-source-filter only as an ablation for Method C.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import random
import re
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Sequence

sys.path.insert(0, os.path.dirname(__file__))

from config import INDEX_PATH, TOP_K_RETRIEVAL
from context_rot_report import (
    aggregate_context_rot_scores,
    print_context_rot_table,
    write_context_rot_report,
)
import method_c_tooling
from pipeline_schema import normalize_reasoning_type

generate_answer = None
RealVectorStore = None
method_b_rag = None
method_c_iterative = None


def _ensure_runtime_imports() -> None:
    """Load model/retrieval modules only when inference is actually needed.

    Report generation should work from scored JSONL files without requiring the
    local model stack or a specific pydantic version.
    """
    global generate_answer, RealVectorStore, method_b_rag, method_c_iterative
    if generate_answer is not None:
        return
    from llm_client import generate_answer as _generate_answer
    from vector_store import VectorStore as _RealVectorStore
    import method_b_rag as _method_b_rag
    import method_c_iterative as _method_c_iterative

    generate_answer = _generate_answer
    RealVectorStore = _RealVectorStore
    method_b_rag = _method_b_rag
    method_c_iterative = _method_c_iterative


RELEVANT_K = 5
CONTEXT_LEVELS = {
    "L0": 0,
    "L1": 10,
    "L2": 20,
    "L3": 40,
}
SEMANTIC_NOISE_POOL_K = 80
RANDOM_NOISE_POOL_K = 200
VECTOR_FILTER_K = 200
NOISE_MODES = ("mixed_noise", "random_noise", "semantic_noise", "same_file_noise", "inverted_gold_noise")
C_ABLATION_MODES = ("full", "no_tools", "no_followup")


def _stable_seed(seed: int, *parts: Any) -> int:
    payload = "::".join(str(part) for part in (seed, *parts))
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return int(digest[:16], 16)


def _chunk_id_from_doc(doc: Dict[str, Any]) -> str:
    return f"{doc['file']}_{doc.get('chunk_index', 0)}"


def _chunk_id_from_result(result: Dict[str, Any]) -> str:
    return result.get("id") or f"{result['metadata']['file']}_{result['metadata'].get('chunk_index', 0)}"


def _strip_file_header(content: str) -> str:
    lines = content.splitlines()
    if lines and lines[0].startswith("FILE: "):
        return "\n".join(lines[1:])
    return content


def _index_doc_to_result(doc: Dict[str, Any], score: float = 1.0) -> Dict[str, Any]:
    return {
        "id": _chunk_id_from_doc(doc),
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


def _result_to_index_doc(result: Dict[str, Any]) -> Dict[str, Any]:
    meta = result["metadata"]
    return {
        "file": meta["file"],
        "language": meta.get("language", ""),
        "total_tokens": max(1, len(result.get("document", "")) // 4),
        "content": result.get("document", ""),
        "chunk_index": meta.get("chunk_index", 0),
        "total_chunks": meta.get("total_chunks", 1),
        "line_start": meta.get("line_start", 1),
        "line_end": meta.get("line_end", 1),
    }


def _ranges_overlap(a_start: int, a_end: int, b_start: int, b_end: int) -> bool:
    return max(a_start, b_start) <= min(a_end, b_end)


def _overlaps_gold_evidence(result: Dict[str, Any], gold_evidence: Sequence[Dict[str, Any]]) -> bool:
    meta = result["metadata"]
    for ev in gold_evidence or []:
        if meta["file"] != ev.get("file"):
            continue
        if _ranges_overlap(
            int(meta.get("line_start", 1)),
            int(meta.get("line_end", 1)),
            int(ev.get("line_start", 1)),
            int(ev.get("line_end", 1)),
        ):
            return True
    return False


def _token_set(text: str) -> set[str]:
    return {tok for tok in text.lower().replace("_", " ").replace("-", " ").split() if len(tok) > 2}


def _lexical_distance(query: str, result: Dict[str, Any]) -> float:
    q_tokens = _token_set(query)
    if not q_tokens:
        return 1.0
    blob = f"{result['metadata']['file']} {result.get('document', '')}"
    overlap = len(q_tokens & _token_set(blob))
    return 1.0 / (1.0 + overlap)


def load_index_docs(index_path: str = INDEX_PATH) -> List[Dict[str, Any]]:
    with open(index_path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, list):
        raise ValueError(f"{index_path} must contain a list of indexed chunks")
    return payload


def _dedupe_results(results: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out, seen = [], set()
    for result in results:
        rid = _chunk_id_from_result(result)
        if rid in seen:
            continue
        result = dict(result)
        result["id"] = rid
        seen.add(rid)
        out.append(result)
    return out


def _gold_overlapping_results(
    gold_evidence: Sequence[Dict[str, Any]],
    docs: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Return indexed chunks that overlap gold evidence spans."""
    if not gold_evidence:
        return []
    results = []
    for doc in docs:
        result = _index_doc_to_result(doc, score=0.0)
        if _overlaps_gold_evidence(result, gold_evidence):
            results.append(result)
    results.sort(key=lambda result: (
        result["metadata"]["file"],
        int(result["metadata"].get("line_start", 1)),
        int(result["metadata"].get("chunk_index", 0)),
    ))
    return _dedupe_results(results)


def _invert_gold_like_text(text: str, variant: int = 0) -> str:
    """Create a deterministic contradiction-style distractor from gold-like text.

    This synthetic mode tests whether a method over-trusts near-gold chunks that
    share file/symbol vocabulary but state reversed behavior.
    """
    replacements = [
        (r"\bAllow\b", "Deny"),
        (r"\bDeny\b", "Allow"),
        (r"\btrue\b", "false"),
        (r"\bfalse\b", "true"),
        (r"\breturns?\b", "does not return"),
        (r"\breads?\b", "does not read"),
        (r"\bwrites?\b", "does not write"),
        (r"\bvalidates?\b", "skips validation of"),
        (r"\braises?\b", "suppresses"),
        (r"\badds?\b", "removes"),
        (r"\bremoves?\b", "adds"),
        (r"\bparses?\b", "does not parse"),
        (r"\bserializes?\b", "does not serialize"),
        (r"\bextracts?\b", "does not extract"),
        (r"\bgrants?\b", "does not grant"),
        (r"\ballows?\b", "does not allow"),
    ]
    inverted = text
    for pattern, repl in replacements:
        inverted = re.sub(pattern, repl, inverted, flags=re.I)
    if variant % 3 == 1:
        inverted = re.sub(r"\bmust\b", "must not", inverted, flags=re.I)
        inverted = re.sub(r"\brequired\b", "not required", inverted, flags=re.I)
        inverted = re.sub(r"\benabled\b", "disabled", inverted, flags=re.I)
    elif variant % 3 == 2:
        inverted = re.sub(r"\buses?\b", "does not use", inverted, flags=re.I)
        inverted = re.sub(r"\bcalls?\b", "does not call", inverted, flags=re.I)
        inverted = re.sub(r"\bincludes?\b", "excludes", inverted, flags=re.I)
    return (
        "FILE: synthetic/inverted_gold_distractor.txt\n"
        "WARNING: SYNTHETIC CONTRADICTORY DISTRACTOR FOR CONTEXT-ROT TESTING.\n"
        "This chunk is intentionally generated by reversing logic from a gold-like evidence chunk.\n\n"
        + inverted
    )


def _inverted_gold_noise(gold_relevant: List[Dict[str, Any]], rng: random.Random) -> List[Dict[str, Any]]:
    out = []
    target_count = max(CONTEXT_LEVELS.values())
    if not gold_relevant:
        return out
    for synthetic_idx in range(target_count):
        result = gold_relevant[synthetic_idx % len(gold_relevant)]
        meta = result["metadata"]
        variant = synthetic_idx // len(gold_relevant)
        fake_file = f"synthetic/inverted_gold/{meta['file'].replace('/', '__')}_{synthetic_idx}.txt"
        document = _invert_gold_like_text(result.get("document", ""), variant=variant)
        out.append({
            "id": f"inverted_gold_{_chunk_id_from_result(result)}_{synthetic_idx}",
            "score": result.get("score", 0.2) + 0.05 + (variant * 0.001),
            "document": document,
            "metadata": {
                "file": fake_file,
                "chunk_index": synthetic_idx,
                "total_chunks": target_count,
                "language": "text",
                "line_start": 1,
                "line_end": max(1, len(document.splitlines())),
            },
        })
    rng.shuffle(out)
    return out


def _synthetic_noise_result(
    noise_mode: str,
    idx: int,
    question: str,
    gold_relevant: List[Dict[str, Any]],
    rng: random.Random,
) -> Dict[str, Any]:
    base = gold_relevant[idx % len(gold_relevant)] if gold_relevant else None
    base_meta = base["metadata"] if base else {"file": "unknown.txt", "line_start": 1, "line_end": 1}
    base_text = base.get("document", "") if base else question
    fake_file = f"synthetic/{noise_mode}/{str(base_meta.get('file', 'unknown')).replace('/', '__')}_{idx}.txt"

    if noise_mode == "same_file_noise":
        fake_file = str(base_meta.get("file", "synthetic/same_file_noise.txt"))
        document = (
            "SYNTHETIC SAME-FILE DISTRACTOR FOR CONTEXT-ROT TESTING.\n"
            "This chunk uses the same file neighborhood vocabulary but is not gold evidence.\n\n"
            + "\n".join(reversed(base_text.splitlines()[:80]))
        )
    elif noise_mode == "semantic_noise":
        q_terms = " ".join(sorted(_token_set(question))[:24])
        document = (
            "SYNTHETIC SEMANTIC DISTRACTOR FOR CONTEXT-ROT TESTING.\n"
            "This chunk repeats question vocabulary but does not contain the ground-truth implementation.\n\n"
            f"Question terms: {q_terms}\n"
            f"Related-looking path: {base_meta.get('file', 'unknown')}\n"
            "This paragraph discusses similar concepts at a high level without the decisive code path, exact policy, or implementation flow.\n"
        )
    elif noise_mode == "random_noise":
        alphabet = list("abcdefghijklmnopqrstuvwxyz")
        rng.shuffle(alphabet)
        document = (
            "SYNTHETIC RANDOM DISTRACTOR FOR CONTEXT-ROT TESTING.\n"
            + " ".join("".join(rng.choice(alphabet) for _ in range(8)) for _ in range(120))
        )
    else:
        document = _invert_gold_like_text(base_text, variant=idx)

    return {
        "id": f"synthetic_{noise_mode}_{idx}",
        "score": 3.0 + idx * 0.001,
        "document": document,
        "metadata": {
            "file": fake_file,
            "chunk_index": idx,
            "total_chunks": max(CONTEXT_LEVELS.values()),
            "language": "text",
            "line_start": 1,
            "line_end": max(1, len(document.splitlines())),
        },
    }


def _ensure_noise_count(
    noise: List[Dict[str, Any]],
    noise_mode: str,
    question: str,
    gold_relevant: List[Dict[str, Any]] | None,
    rng: random.Random,
) -> List[Dict[str, Any]]:
    """Pad pure noise modes with synthetic same-type distractors if needed."""
    target_count = max(CONTEXT_LEVELS.values())
    out = _dedupe_results(noise)
    seen = {_chunk_id_from_result(item) for item in out}
    idx = 0
    while len(out) < target_count:
        candidate = _synthetic_noise_result(noise_mode, idx, question, gold_relevant or [], rng)
        cid = _chunk_id_from_result(candidate)
        if cid not in seen:
            out.append(candidate)
            seen.add(cid)
        idx += 1
    return out


def _build_noise_candidates(
    question: str,
    relevant_ids: set[str],
    gold_evidence: Sequence[Dict[str, Any]],
    docs: List[Dict[str, Any]],
    vector_store: RealVectorStore,
    rng: random.Random,
    noise_mode: str = "mixed_noise",
    gold_relevant: List[Dict[str, Any]] | None = None,
) -> List[Dict[str, Any]]:
    if noise_mode not in NOISE_MODES:
        raise ValueError(f"noise_mode must be one of {NOISE_MODES}, got {noise_mode}")

    semantic_raw = vector_store.retrieve(question, top_k=SEMANTIC_NOISE_POOL_K + RELEVANT_K)
    semantic_noise = [
        result for result in semantic_raw
        if _chunk_id_from_result(result) not in relevant_ids
        and not _overlaps_gold_evidence(result, gold_evidence)
    ]

    relevant_files = {ev.get("file") for ev in gold_evidence or [] if ev.get("file")}
    same_file_noise = []
    if relevant_files:
        for doc in docs:
            if doc.get("file") not in relevant_files:
                continue
            result = _index_doc_to_result(doc, score=1.5)
            if _chunk_id_from_result(result) in relevant_ids:
                continue
            if _overlaps_gold_evidence(result, gold_evidence):
                continue
            same_file_noise.append(result)

    random_docs = docs[:]
    rng.shuffle(random_docs)
    random_noise = []
    for doc in random_docs:
        result = _index_doc_to_result(doc, score=2.0)
        if _chunk_id_from_result(result) in relevant_ids:
            continue
        if _overlaps_gold_evidence(result, gold_evidence):
            continue
        random_noise.append(result)
        if len(random_noise) >= RANDOM_NOISE_POOL_K:
            break

    rng.shuffle(semantic_noise)
    rng.shuffle(random_noise)
    rng.shuffle(same_file_noise)
    if noise_mode == "inverted_gold_noise":
        return _ensure_noise_count(_inverted_gold_noise(gold_relevant or [], rng), noise_mode, question, gold_relevant, rng)
    if noise_mode == "semantic_noise":
        return _ensure_noise_count(semantic_noise, noise_mode, question, gold_relevant, rng)
    if noise_mode == "random_noise":
        return _ensure_noise_count(random_noise, noise_mode, question, gold_relevant, rng)
    if noise_mode == "same_file_noise":
        return _ensure_noise_count(same_file_noise, noise_mode, question, gold_relevant, rng)

    mixed = []
    for i in range(max(len(semantic_noise), len(random_noise))):
        if i < len(semantic_noise):
            mixed.append(semantic_noise[i])
        if i < len(random_noise):
            mixed.append(random_noise[i])
    return _ensure_noise_count(mixed, noise_mode, question, gold_relevant, rng)


def _interleave_relevant_and_noise(
    relevant: List[Dict[str, Any]],
    noise: List[Dict[str, Any]],
    rng: random.Random,
) -> List[Dict[str, Any]]:
    relevant_order = relevant[:]
    noise_order = noise[:]
    rng.shuffle(relevant_order)
    rng.shuffle(noise_order)
    out = []
    for i in range(max(len(relevant_order), len(noise_order))):
        if i < len(relevant_order):
            out.append(relevant_order[i])
        if i < len(noise_order):
            out.append(noise_order[i])
    return out


def build_context_levels(
    question: str,
    gt_item: Dict[str, Any] | None = None,
    vector_store: RealVectorStore | None = None,
    docs: List[Dict[str, Any]] | None = None,
    seed: int = 42,
    relevant_k: int = RELEVANT_K,
    noise_mode: str = "mixed_noise",
    levels: Sequence[str] | None = None,
) -> List[Dict[str, Any]]:
    """Build L0-L3 context pools for one question.

    Relevant chunks use gold-overlapping evidence as the clean core when gold
    spans exist. Retrieval fills only if the gold core has fewer than
    relevant_k chunks. Noise mode controls which distractor family is added.
    """
    if vector_store is None:
        _ensure_runtime_imports()
        vector_store = RealVectorStore()
    docs = docs if docs is not None else load_index_docs()
    gt_item = gt_item or {}
    gold_evidence = gt_item.get("evidence", [])
    qid = gt_item.get("id", hashlib.sha1(question.encode("utf-8")).hexdigest()[:10])
    rng = random.Random(_stable_seed(seed, qid, "context-levels"))
    selected_levels = list(levels) if levels is not None else list(CONTEXT_LEVELS.keys())
    invalid_levels = [level for level in selected_levels if level not in CONTEXT_LEVELS]
    if invalid_levels:
        raise ValueError(f"Unknown context level(s): {invalid_levels}. Valid levels: {list(CONTEXT_LEVELS)}")

    gold_relevant = _gold_overlapping_results(gold_evidence, docs)
    relevant = list(gold_relevant)
    if len(relevant) < relevant_k:
        retrieved_fillers = vector_store.retrieve(question, top_k=max(relevant_k * 4, relevant_k))
        existing_ids = {_chunk_id_from_result(result) for result in relevant}
        for result in _dedupe_results(retrieved_fillers):
            rid = _chunk_id_from_result(result)
            if rid in existing_ids:
                continue
            relevant.append(result)
            existing_ids.add(rid)
            if len(relevant) >= relevant_k:
                break
    if not relevant:
        relevant = _dedupe_results(vector_store.retrieve(question, top_k=relevant_k))[:relevant_k]
    relevant_ids = {_chunk_id_from_result(result) for result in relevant}
    noise_candidates = _build_noise_candidates(
        question,
        relevant_ids,
        gold_evidence,
        docs,
        vector_store,
        rng,
        noise_mode=noise_mode,
        gold_relevant=gold_relevant,
    )

    levels = []
    for level_name in selected_levels:
        noise_count = CONTEXT_LEVELS[level_name]
        level_rng = random.Random(_stable_seed(seed, qid, level_name))
        level_noise = noise_candidates[:noise_count]
        chunks = _interleave_relevant_and_noise(relevant, level_noise, level_rng)
        levels.append({
            "context_level": level_name,
            "noise_count": noise_count,
            "noise_mode": noise_mode,
            "chunks": chunks,
            "relevant_ids": sorted(relevant_ids),
            "gold_relevant_ids": [_chunk_id_from_result(result) for result in gold_relevant],
            "noise_ids": [_chunk_id_from_result(result) for result in level_noise],
            "context_size": len(chunks),
            "context_tokens_est": sum(max(1, len(result.get("document", "")) // 4) for result in chunks),
        })
    return levels


def _format_context_for_method_a(chunks: List[Dict[str, Any]]) -> str:
    blocks = []
    for result in chunks:
        meta = result["metadata"]
        text = _strip_file_header(result.get("document", ""))
        blocks.append("===== CHUNK START =====")
        blocks.append(f"PATH: {meta['file']}")
        blocks.append(f"LINES: {meta.get('line_start', 1)}-{meta.get('line_end', 1)}")
        blocks.append("---------------------")
        for offset, line in enumerate(text.splitlines(), start=int(meta.get("line_start", 1))):
            blocks.append(f"{offset}: {line}")
        blocks.append("===== CHUNK END =====")
        blocks.append("")
    return "\n".join(blocks)


class PoolVectorStore:
    """VectorStore-compatible adapter that restricts retrieval to a fixed pool."""

    pool_results: List[Dict[str, Any]] = []
    pool_by_id: Dict[str, Dict[str, Any]] = {}
    base_store: RealVectorStore | None = None

    @classmethod
    def configure(cls, pool_results: List[Dict[str, Any]], base_store: RealVectorStore | None = None) -> None:
        cls.pool_results = _dedupe_results(pool_results)
        cls.pool_by_id = {_chunk_id_from_result(result): result for result in cls.pool_results}
        cls.base_store = base_store

    def validate_manifest(self, *args: Any, **kwargs: Any) -> bool:
        return True

    def retrieve(self, query: str, top_k: int = TOP_K_RETRIEVAL) -> List[Dict[str, Any]]:
        if not self.pool_results:
            return []
        base_store = self.base_store or RealVectorStore()
        found = []
        seen = set()
        try:
            candidates = base_store.retrieve(query, top_k=max(VECTOR_FILTER_K, top_k * 20))
            for result in candidates:
                rid = _chunk_id_from_result(result)
                if rid not in self.pool_by_id or rid in seen:
                    continue
                pooled = dict(self.pool_by_id[rid])
                pooled["score"] = result.get("score", pooled.get("score", 1.0))
                found.append(pooled)
                seen.add(rid)
                if len(found) >= top_k:
                    return found
        except Exception:
            pass

        remaining = [
            result for result in self.pool_results
            if _chunk_id_from_result(result) not in seen
        ]
        remaining.sort(key=lambda result: (_lexical_distance(query, result), result["metadata"]["file"]))
        return (found + remaining)[:top_k]


@contextlib.contextmanager
def restricted_retrieval_pool(
    pool_chunks: List[Dict[str, Any]],
    base_store: RealVectorStore | None = None,
    disable_c_source_filter: bool = False,
    c_ablation: str = "full",
) -> Iterator[None]:
    """Temporarily restrict Method B/C retrieval and Method C tool reads to one pool."""
    if c_ablation not in C_ABLATION_MODES:
        raise ValueError(f"Unknown C ablation: {c_ablation}. Valid modes: {C_ABLATION_MODES}")
    PoolVectorStore.configure(pool_chunks, base_store=base_store)

    old_b_vector_store = method_b_rag.VectorStore
    old_c_vector_store = method_c_iterative.VectorStore
    old_c_index_cache = method_c_tooling.get_index_docs_cache()
    old_c_is_docs_noise = method_c_iterative._is_docs_noise
    old_c_run_followup_tools = method_c_iterative.run_followup_tools
    old_c_expand_retrieval = method_c_iterative.expand_retrieval

    method_b_rag.VectorStore = PoolVectorStore
    method_c_iterative.VectorStore = PoolVectorStore
    method_c_tooling.set_index_docs_cache([_result_to_index_doc(result) for result in pool_chunks])
    if disable_c_source_filter:
        method_c_iterative._is_docs_noise = lambda _file_path: False
    if c_ablation == "no_tools":
        # Keep semantic follow-up queries, but remove deterministic code-navigation
        # tools. This tests whether tools, not just extra retrieval calls, create
        # Method C's robustness under noise.
        method_c_iterative.run_followup_tools = lambda _question, _state, _visited: []
    elif c_ablation == "no_followup":
        # Initial read + final synthesis only. This tests whether iterative
        # evidence expansion is the mechanism behind C's advantage.
        method_c_iterative.expand_retrieval = lambda _vs, _question, _state, _visited: []

    try:
        yield
    finally:
        method_b_rag.VectorStore = old_b_vector_store
        method_c_iterative.VectorStore = old_c_vector_store
        method_c_tooling.set_index_docs_cache(old_c_index_cache)
        method_c_iterative._is_docs_noise = old_c_is_docs_noise
        method_c_iterative.run_followup_tools = old_c_run_followup_tools
        method_c_iterative.expand_retrieval = old_c_expand_retrieval


def _normalize_method_result(result: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "predicted_answer": result.get("answer", ""),
        "predicted_evidence": result.get("evidence", []),
        "latency_sec": result.get("latency", 0.0),
        "input_tokens": result.get("input_tokens", 0),
        "output_tokens": result.get("output_tokens", 0),
        "model_calls": result.get("model_calls", 1),
        "success": bool(result.get("success", True)),
        "raw_error": result.get("error"),
        "raw_warning": result.get("warning"),
    }


def run_method_on_context(
    method: str,
    question: str,
    chunks: List[Dict[str, Any]],
    base_store: RealVectorStore,
    disable_c_source_filter: bool = False,
    c_ablation: str = "full",
) -> Dict[str, Any]:
    _ensure_runtime_imports()
    method = method.upper()
    if method == "A":
        context = _format_context_for_method_a(chunks)
        return generate_answer(prompt_context=context, question=question)

    with restricted_retrieval_pool(
        chunks,
        base_store=base_store,
        disable_c_source_filter=disable_c_source_filter,
        c_ablation=c_ablation if method == "C" else "full",
    ):
        if method == "B":
            return method_b_rag.run_method_b(question, top_k=RELEVANT_K)
        if method == "C":
            return method_c_iterative.run_method_c(question, top_k=RELEVANT_K)
    raise ValueError(f"Unknown method: {method}")


def run_context_rot_experiment(
    dataset: List[Dict[str, Any]],
    methods: Sequence[str] = ("A", "B", "C"),
    output_path: str | None = None,
    seed: int = 42,
    samples: int | None = None,
    disable_c_source_filter: bool = False,
    c_ablation: str = "full",
    noise_mode: str = "mixed_noise",
    levels: Sequence[str] | None = None,
) -> List[Dict[str, Any]]:
    """Run context-rot inference and optionally save raw JSONL predictions."""
    _ensure_runtime_imports()
    if c_ablation not in C_ABLATION_MODES:
        raise ValueError(f"Unknown C ablation: {c_ablation}. Valid modes: {C_ABLATION_MODES}")
    if samples is not None:
        dataset = dataset[:samples]
    methods = [method.upper() for method in methods]
    docs = load_index_docs()
    base_store = RealVectorStore()
    rows: List[Dict[str, Any]] = []

    selected_level_names = list(levels) if levels is not None else None

    for item in dataset:
        context_levels = build_context_levels(
            item["question"],
            gt_item=item,
            vector_store=base_store,
            docs=docs,
            seed=seed,
            noise_mode=noise_mode,
            levels=selected_level_names,
        )
        for level in context_levels:
            for method in methods:
                start = time.time()
                try:
                    result = run_method_on_context(
                        method,
                        item["question"],
                        level["chunks"],
                        base_store,
                        disable_c_source_filter=disable_c_source_filter,
                        c_ablation=c_ablation,
                    )
                    normalized = _normalize_method_result(result)
                except Exception as exc:
                    normalized = {
                        "predicted_answer": "",
                        "predicted_evidence": [],
                        "latency_sec": time.time() - start,
                        "input_tokens": 0,
                        "output_tokens": 0,
                        "model_calls": 0,
                        "success": False,
                        "raw_error": str(exc),
                        "raw_warning": None,
                    }
                row = {
                    "question_id": item["id"],
                    "id": item["id"],
                    "method": method,
                    "context_level": level["context_level"],
                    "context_size": level["context_size"],
                    "context_tokens_est": level["context_tokens_est"],
                    "noise_count": level["noise_count"],
                    "noise_mode": level["noise_mode"],
                    "dataset": item.get("dataset", "unknown"),
                    "difficulty": item.get("difficulty", "unknown"),
                    "reasoning_type": normalize_reasoning_type(item),
                    "question": item["question"],
                    "answer": normalized["predicted_answer"],
                    "evidence": normalized["predicted_evidence"],
                    **normalized,
                    "relevant_ids": level["relevant_ids"],
                    "gold_relevant_ids": level["gold_relevant_ids"],
                    "noise_ids": level["noise_ids"],
                    "context_chunk_ids": [_chunk_id_from_result(result) for result in level["chunks"]],
                    "context_rot_seed": seed,
                    "c_source_filter_disabled": disable_c_source_filter,
                    "c_ablation": c_ablation if method == "C" else "not_applicable",
                }
                rows.append(row)
                if output_path:
                    save_results([row], output_path, append=True)
    return rows


def save_results(rows: List[Dict[str, Any]], output_path: str, append: bool = False) -> None:
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if append else "w"
    with open(output_path, mode, encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def score_context_rot_predictions(
    predictions_file: str,
    qa_file: str,
    output_file: str,
    run_judge: bool = True,
    with_citation_judge: bool = False,
    with_bertscore: bool = True,
    bertscore_model: str | None = None,
    bertscore_device: str | None = None,
    bertscore_rescale: bool = False,
) -> str:
    """Score context-rot predictions while preserving context_level metadata."""
    import score as score_module

    score_module.load_judge_cache()
    gt_map = score_module.load_ground_truth(qa_file)
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    with open(predictions_file, "r", encoding="utf-8") as fin, open(output_file, "w", encoding="utf-8") as fout:
        for line in fin:
            if not line.strip():
                continue
            record = json.loads(line)
            gt_item = gt_map[record["id"]]
            scored = score_module.score_record(
                record,
                gt_item,
                run_judge=run_judge,
                with_citation_judge=with_citation_judge,
                with_bertscore=with_bertscore,
                bertscore_model=bertscore_model or score_module.BERTSCORE_DEFAULT_MODEL,
                bertscore_device=bertscore_device,
                bertscore_rescale=bertscore_rescale,
            )
            for field in [
                "question_id",
                "context_level",
                "context_size",
                "context_tokens_est",
                "noise_count",
                "noise_mode",
                "relevant_ids",
                "gold_relevant_ids",
                "noise_ids",
                "context_chunk_ids",
                "context_rot_seed",
                "c_source_filter_disabled",
                "c_ablation",
            ]:
                if field in record:
                    scored[field] = record[field]
            fout.write(json.dumps(scored, ensure_ascii=False) + "\n")
    return output_file


def main() -> None:
    parser = argparse.ArgumentParser(description="Run controlled context-rot evaluation.")
    parser.add_argument("--qa-file", default="qa_dataset/seed_v3_test.json")
    parser.add_argument("--methods", nargs="+", default=["A", "B", "C"], choices=["A", "B", "C"])
    parser.add_argument("--output-dir", default="results/context_rot")
    parser.add_argument("--samples", type=int, default=None)
    parser.add_argument("--subset", default="", help="Optional file with question IDs, one per line.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--noise-mode", default="mixed_noise", choices=list(NOISE_MODES),
                        help="Distractor construction mode for L1-L3.")
    parser.add_argument("--levels", nargs="+", default=None, choices=list(CONTEXT_LEVELS.keys()),
                        help="Context levels to run, e.g. --levels L0 L2 L3.")
    parser.add_argument("--disable-c-source-filter", action="store_true",
                        help="Ablation: disable Method C's hard docs/source filtering during context-rot runs.")
    parser.add_argument("--c-ablation", default="full", choices=list(C_ABLATION_MODES),
                        help=(
                            "Method C ablation. full=normal C; no_tools disables deterministic "
                            "code-navigation tools but keeps semantic follow-up retrieval; "
                            "no_followup disables iterative follow-up expansion."
                        ))
    parser.add_argument("--score", action="store_true", help="Score predictions after inference.")
    parser.add_argument("--no-judge", action="store_true", help="When scoring, skip LLM judge and compute citation metrics only.")
    parser.add_argument("--with-citation-judge", action="store_true")
    parser.add_argument("--with-bertscore", action="store_true",
                        help="Deprecated: BERTScore is computed by default unless --no-bertscore is set.")
    parser.add_argument("--no-bertscore", action="store_true",
                        help="Disable automatic BERTScore computation when scoring.")
    parser.add_argument("--bertscore-model", default=None,
                        help="Optional BERTScore model override, e.g. roberta-large or distilbert-base-uncased.")
    parser.add_argument("--bertscore-device", default=None,
                        help="Optional BERTScore device, e.g. cuda:0 or cpu.")
    parser.add_argument("--bertscore-rescale", action="store_true",
                        help="Use BERTScore baseline rescaling when available.")
    parser.add_argument("--report-scored", default="",
                        help="Create a detailed context-rot report from an existing scored JSONL and exit.")
    args = parser.parse_args()

    if args.report_scored:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_prefix = os.path.join(args.output_dir, "analysis", f"context_rot_report_{timestamp}")
        paths = write_context_rot_report(args.report_scored, report_prefix)
        print(f"Context-rot report JSON saved to {paths['json']}")
        print(f"Context-rot report TXT saved to {paths['txt']}")
        return

    with open(args.qa_file, "r", encoding="utf-8") as f:
        dataset = json.load(f)
    if args.subset:
        available_ids = {item["id"] for item in dataset}
        with open(args.subset, "r", encoding="utf-8") as f:
            keep = {line.strip() for line in f if line.strip()}
        if not keep:
            raise ValueError(f"Subset file is empty: {args.subset}")
        dataset = [item for item in dataset if item["id"] in keep]
        if not dataset:
            preview = ", ".join(sorted(list(keep))[:3])
            raise ValueError(
                f"Subset matched 0 questions. --subset expects a plain text file with one question ID per line, "
                f"not a dataset JSON. Example IDs in subset input: {preview}. "
                f"Available ID examples: {', '.join(sorted(list(available_ids))[:5])}"
            )
    if not dataset:
        raise ValueError("No questions selected for context-rot evaluation.")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    raw_path = os.path.join(args.output_dir, "raw", f"context_rot_predictions_{timestamp}.jsonl")
    run_context_rot_experiment(
        dataset,
        methods=args.methods,
        output_path=raw_path,
        seed=args.seed,
        samples=args.samples,
        disable_c_source_filter=args.disable_c_source_filter,
        c_ablation=args.c_ablation,
        noise_mode=args.noise_mode,
        levels=args.levels,
    )
    print(f"Raw context-rot predictions saved to {raw_path}")

    if args.score:
        scored_path = os.path.join(args.output_dir, "scored", f"context_rot_scored_{timestamp}.jsonl")
        score_context_rot_predictions(
            raw_path,
            args.qa_file,
            scored_path,
            run_judge=not args.no_judge,
            with_citation_judge=args.with_citation_judge,
            with_bertscore=not args.no_bertscore,
            bertscore_model=args.bertscore_model,
            bertscore_device=args.bertscore_device,
            bertscore_rescale=args.bertscore_rescale,
        )
        summary_path = os.path.join(args.output_dir, "analysis", f"context_rot_summary_{timestamp}.json")
        table = aggregate_context_rot_scores(scored_path, output_json=summary_path)
        print_context_rot_table(table)
        report_prefix = os.path.join(args.output_dir, "analysis", f"context_rot_report_{timestamp}")
        paths = write_context_rot_report(scored_path, report_prefix)
        print(f"Scored context-rot results saved to {scored_path}")
        print(f"Context-rot summary saved to {summary_path}")
        print(f"Context-rot report JSON saved to {paths['json']}")
        print(f"Context-rot report TXT saved to {paths['txt']}")


if __name__ == "__main__":
    main()
