import argparse
import json
import os
from datetime import datetime
from statistics import mean
from typing import Dict, List

import sys
sys.path.insert(0, os.path.dirname(__file__))

from config import INDEX_PATH
from method_c_iterative import _is_docs_noise
from reranker import rerank_and_select
from utils.retrieval_eval import (
    compute_basic_retrieval_metrics,
    summarize_candidates,
    summarize_spans,
)
from vector_store import VectorStore


def _selected_for_method(method: str, reranked: List[Dict]) -> List[Dict]:
    if method == "C":
        source_only = [r for r in reranked if not _is_docs_noise(r["metadata"]["file"])]
        return source_only if source_only else reranked
    return reranked


def _pool_size(method: str, top_k: int) -> int:
    if method == "B":
        return top_k * 4
    if method == "C":
        return top_k * 3
    raise ValueError(f"Unsupported method: {method}")


def _span_view(results: List[Dict]) -> List[Dict]:
    return [
        {
            "file": r["metadata"]["file"],
            "line_start": r["metadata"].get("line_start", 1),
            "line_end": r["metadata"].get("line_end", 1),
        }
        for r in results
    ]


def run_audit(qa_file: str, method: str, top_k: int, output_dir: str, limit: int = None) -> str:
    with open(qa_file, "r", encoding="utf-8") as f:
        qa_items = json.load(f)
    if limit:
        qa_items = qa_items[:limit]

    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_jsonl = os.path.join(output_dir, f"retrieval_audit_{method.lower()}_{timestamp}.jsonl")
    out_summary = os.path.join(output_dir, f"retrieval_audit_summary_{method.lower()}_{timestamp}.json")

    vs = VectorStore()
    vs.validate_manifest(INDEX_PATH, raise_on_mismatch=True)

    records = []
    for item in qa_items:
        question = item["question"]
        gold = item.get("evidence", [])
        raw = vs.retrieve(question, top_k=_pool_size(method, top_k))
        reranked = rerank_and_select(raw, top_k=top_k, query=question)
        selected = _selected_for_method(method, reranked)

        record = {
            "id": item["id"],
            "method": method,
            "question": question,
            "reasoning_type": item.get("reasoning_type", "unknown"),
            "gold_evidence": summarize_spans(gold),
            "raw_top_pool": summarize_candidates(raw),
            "reranked_top_k": summarize_candidates(reranked),
            "selected_for_read": summarize_candidates(selected),
            "raw_metrics": compute_basic_retrieval_metrics(_span_view(raw), gold, k=min(top_k, len(raw)) or top_k),
            "reranked_metrics": compute_basic_retrieval_metrics(_span_view(reranked), gold, k=top_k),
            "selected_metrics": compute_basic_retrieval_metrics(_span_view(selected), gold, k=min(top_k, len(selected)) or top_k),
        }
        records.append(record)

    with open(out_jsonl, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    summary = {
        "qa_file": qa_file,
        "method": method,
        "top_k": top_k,
        "num_questions": len(records),
        "raw_metrics_mean": {},
        "reranked_metrics_mean": {},
        "selected_metrics_mean": {},
    }
    for stage in ("raw_metrics", "reranked_metrics", "selected_metrics"):
        metric_names = set()
        for rec in records:
            metric_names.update(rec[stage].keys())
        target = summary[f"{stage}_mean"]
        for metric_name in sorted(metric_names):
            values = [rec[stage][metric_name] for rec in records if rec[stage].get(metric_name) is not None]
            target[metric_name] = mean(values) if values else None

    with open(out_summary, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"Saved audit records to {out_jsonl}")
    print(f"Saved summary to {out_summary}")
    return out_jsonl


def main():
    parser = argparse.ArgumentParser(description="Audit retrieval quality before answer generation.")
    parser.add_argument("--qa-file", type=str, default="qa_dataset/seed_v1.json")
    parser.add_argument("--method", type=str, required=True, choices=["B", "C"])
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--limit", type=int, default=None, help="Optional number of questions to audit")
    parser.add_argument("--output-dir", type=str, default="results/retrieval_audit")
    args = parser.parse_args()

    if os.path.basename(os.getcwd()) == "src":
        os.chdir("..")

    run_audit(
        qa_file=args.qa_file,
        method=args.method,
        top_k=args.top_k,
        output_dir=args.output_dir,
        limit=args.limit,
    )


if __name__ == "__main__":
    main()
