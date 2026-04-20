import argparse
import json
import os
import time
from datetime import datetime
from typing import Dict, List, Tuple

import sys
sys.path.insert(0, os.path.dirname(__file__))

from config import INDEX_PATH, TOP_K_RETRIEVAL
from method_b_rag import synthesize_answer
from method_c_recursive import (
    MAX_EVIDENCE_SNIPPETS,
    dedupe_citations,
    final_answer as method_c_final_answer,
    make_initial_state,
    read_and_reason,
    select_final_evidence,
    update_state_from_step,
)
from reranker import rerank_and_select
from utils.retrieval_eval import files_match
from vector_store import VectorStore


def _candidate_disk_paths(rel_path: str, repo_root: str = "dataset") -> List[str]:
    """
    Resolve seed_v1 evidence paths to actual local files.

    FastAPI paths in the dataset are stored like `fastapi/routing.py`, but on disk the
    repo lives under `dataset/repos/fastapi/fastapi/...`, so we must prepend the repo root.
    """
    fastapi_root_files = {"pyproject.toml", "README.md", "CONTRIBUTING.md"}
    if rel_path.startswith("fastapi/") or rel_path.startswith("docs_src/") or rel_path.startswith("docs/") or rel_path in fastapi_root_files:
        return [
            os.path.join(repo_root, "repos", "fastapi", rel_path),
            os.path.join(repo_root, "repos", rel_path),
        ]
    return [
        os.path.join(repo_root, "docs", "aws-lambda-developer-guide", rel_path),
    ]


def _resolve_disk_path(rel_path: str, repo_root: str = "dataset") -> str:
    for candidate in _candidate_disk_paths(rel_path, repo_root=repo_root):
        if os.path.exists(candidate):
            return candidate
    return ""


def load_oracle_chunks(evidence: List[Dict], repo_root: str = "dataset") -> Tuple[List[Dict], List[Dict]]:
    chunks = []
    missing = []
    for idx, ev in enumerate(evidence):
        file_path = ev.get("file", "")
        line_start = int(ev.get("line_start", 1))
        line_end = int(ev.get("line_end", 1))
        disk = _resolve_disk_path(file_path, repo_root=repo_root)
        if not disk:
            missing.append({
                "file": file_path,
                "line_start": line_start,
                "line_end": line_end,
                "attempted_paths": _candidate_disk_paths(file_path, repo_root=repo_root),
            })
            continue
        with open(disk, "r", encoding="utf-8") as f:
            lines = f.readlines()
        snippet = "".join(lines[max(0, line_start - 1): line_end])
        chunks.append({
            "id": f"oracle_{idx}_{file_path}_{line_start}_{line_end}",
            "score": 0.0,
            "document": snippet,
            "metadata": {
                "file": file_path,
                "chunk_index": idx,
                "total_chunks": len(evidence),
                "line_start": line_start,
                "line_end": line_end,
            }
        })
    return chunks, missing


def add_noise_chunks(question: str, oracle_chunks: List[Dict], noise_k: int) -> List[Dict]:
    if noise_k <= 0:
        return oracle_chunks

    vs = VectorStore()
    vs.validate_manifest(INDEX_PATH, raise_on_mismatch=True)
    raw = vs.retrieve(question, top_k=max(noise_k * 4, TOP_K_RETRIEVAL * 2))
    reranked = rerank_and_select(raw, top_k=max(noise_k * 2, noise_k), query=question)

    gold_files = [chunk["metadata"]["file"] for chunk in oracle_chunks]
    selected_noise = []
    seen = {chunk["id"] for chunk in oracle_chunks}
    for item in reranked:
        if item["id"] in seen:
            continue
        if any(files_match(item["metadata"]["file"], gold_file) for gold_file in gold_files):
            continue
        selected_noise.append(item)
        seen.add(item["id"])
        if len(selected_noise) >= noise_k:
            break
    return oracle_chunks + selected_noise


def oracle_method_b(question: str, evidence_chunks: List[Dict]) -> Dict:
    return synthesize_answer(question=question, evidence_chunks=evidence_chunks[:MAX_EVIDENCE_SNIPPETS])


def oracle_method_c(question: str, evidence_chunks: List[Dict]) -> Dict:
    start_time = time.time()
    total_tokens_in = 0
    total_tokens_out = 0
    model_calls = 0
    state = make_initial_state(question)
    grounded_citations = []
    gathered_evidence = []

    for i, chunk in enumerate(evidence_chunks, start=1):
        meta = chunk["metadata"]
        file_path = meta["file"]
        line_start = meta.get("line_start", 1)
        line_end = meta.get("line_end", 1)
        chunk_text = chunk["document"]
        step_result = read_and_reason(
            question=question,
            state=state,
            file_path=file_path,
            chunk_text=chunk_text,
            line_start=line_start,
            line_end=line_end,
            chunk_num=i,
            total_chunks=len(evidence_chunks),
        )
        model_calls += 1
        total_tokens_in += step_result.get("tokens_in", 0)
        total_tokens_out += step_result.get("tokens_out", 0)
        update_state_from_step(state, step_result, file_path, chunk_text, chunk["id"])

        if step_result.get("evidence_role") in {"direct", "bridge"} or step_result.get("verified_facts"):
            grounded_citations.append({
                "file": file_path,
                "line_start": line_start,
                "line_end": line_end,
            })
            gathered_evidence.append({
                "id": chunk["id"],
                "file": file_path,
                "line_start": line_start,
                "line_end": line_end,
                "text": chunk_text,
            })

    grounded_citations = dedupe_citations(grounded_citations)
    if not gathered_evidence:
        gathered_evidence = [
            {
                "id": chunk["id"],
                "file": chunk["metadata"]["file"],
                "line_start": chunk["metadata"].get("line_start", 1),
                "line_end": chunk["metadata"].get("line_end", 1),
                "text": chunk["document"],
            }
            for chunk in evidence_chunks
        ]
        grounded_citations = [
            {
                "file": chunk["metadata"]["file"],
                "line_start": chunk["metadata"].get("line_start", 1),
                "line_end": chunk["metadata"].get("line_end", 1),
            }
            for chunk in evidence_chunks
        ]

    selected = select_final_evidence(gathered_evidence, state, question)
    answer_result = method_c_final_answer(
        question=question,
        state=state,
        citations=grounded_citations,
        selected_raw_evidence=selected,
    )
    model_calls += 1
    total_tokens_in += answer_result.get("input_tokens", 0)
    total_tokens_out += answer_result.get("output_tokens", 0)

    return {
        "success": True,
        "answer": answer_result.get("answer", ""),
        "evidence": grounded_citations,
        "latency": time.time() - start_time,
        "input_tokens": total_tokens_in,
        "output_tokens": total_tokens_out,
        "model_calls": model_calls,
    }


def run_oracle_eval(qa_file: str, method: str, output_dir: str, limit: int = None, add_noise: int = 0) -> str:
    with open(qa_file, "r", encoding="utf-8") as f:
        qa_items = json.load(f)
    if limit:
        qa_items = qa_items[:limit]

    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = f"_noise{add_noise}" if add_noise > 0 else ""
    out_path = os.path.join(output_dir, f"predictions_oracle_{method.lower()}{suffix}_{timestamp}.jsonl")

    total_missing = 0
    samples_with_missing = 0
    with open(out_path, "w", encoding="utf-8") as out:
        for item in qa_items:
            oracle_chunks, missing_evidence = load_oracle_chunks(item.get("evidence", []))
            loaded_gold_count = len(oracle_chunks)
            if missing_evidence:
                samples_with_missing += 1
                total_missing += len(missing_evidence)
                print(
                    f"[Oracle warning] {item['id']}: missing {len(missing_evidence)} "
                    f"ground-truth evidence file(s) on disk."
                )
            oracle_chunks = add_noise_chunks(item["question"], oracle_chunks, add_noise)

            if method == "B":
                result = oracle_method_b(item["question"], oracle_chunks)
            elif method == "C":
                result = oracle_method_c(item["question"], oracle_chunks)
            else:
                raise ValueError(f"Unsupported method: {method}")

            record = {
                "id": item["id"],
                "method": f"oracle_{method}",
                "dataset": item.get("dataset", "unknown"),
                "difficulty": item.get("difficulty", "unknown"),
                "reasoning_type": item.get("reasoning_type", "unknown"),
                "question": item["question"],
                "predicted_answer": result.get("answer", ""),
                "predicted_evidence": result.get("evidence", []),
                "success": result.get("success", True),
                "error_type": None,
                "raw_error": result.get("error"),
                "raw_warning": (
                    f"oracle_missing_evidence:{len(missing_evidence)}"
                    if missing_evidence else None
                ),
                "latency_sec": result.get("latency", 0.0),
                "input_tokens": result.get("input_tokens", 0),
                "output_tokens": result.get("output_tokens", 0),
                "model_calls": result.get("model_calls", 0),
                "oracle_mode": "oracle+noise" if add_noise > 0 else "oracle",
                "noise_chunks_added": add_noise,
                "oracle_expected_evidence_count": len(item.get("evidence", [])),
                "oracle_loaded_evidence_count": loaded_gold_count,
                "oracle_missing_evidence": missing_evidence,
            }
            out.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Saved oracle predictions to {out_path}")
    if total_missing:
        print(
            f"[Oracle summary] {samples_with_missing} sample(s) had missing ground-truth "
            f"evidence on disk ({total_missing} total missing spans)."
        )
    return out_path


def main():
    parser = argparse.ArgumentParser(description="Run oracle-retrieval evaluation to separate retrieval from reasoning.")
    parser.add_argument("--qa-file", type=str, default="qa_dataset/seed_v1.json")
    parser.add_argument("--method", type=str, required=True, choices=["B", "C"])
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--add-noise", type=int, default=0, help="Append this many non-gold retrieved chunks to the oracle evidence.")
    parser.add_argument("--output-dir", type=str, default="results/oracle_eval")
    args = parser.parse_args()

    if os.path.basename(os.getcwd()) == "src":
        os.chdir("..")

    run_oracle_eval(
        qa_file=args.qa_file,
        method=args.method,
        output_dir=args.output_dir,
        limit=args.limit,
        add_noise=args.add_noise,
    )


if __name__ == "__main__":
    main()
