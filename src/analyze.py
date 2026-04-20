"""
analyze.py - Error analysis + statistical tests over scored JSONL outputs.
"""
import os
import json
import glob
import random
import argparse
import statistics
from typing import List, Dict
from datetime import datetime

import sys
sys.path.insert(0, os.path.dirname(__file__))

from pipeline_schema import ensure_required_fields


REQUIRED_SCORED_FIELDS = [
    "id",
    "method",
    "dataset",
    "difficulty",
    "reasoning_type",
    "latency_sec",
    "input_tokens",
    "output_tokens",
    "model_calls",
    "judge_status",
    "answer_score",
    "citation_file_precision",
    "citation_file_recall",
    "citation_file_f1",
    "evidence_line_iou",
    "citation_support_score",
]


def categorize_error(record: Dict, tolerance_accuracy: float = 0.5) -> str:
    acc = record.get("answer_score", 0.0) or 0.0
    cit_recall = record.get("citation_file_recall", 0.0) or 0.0
    cit_f1 = record.get("citation_file_f1", 0.0) or 0.0
    support = record.get("citation_support_score", 0.0) or 0.0
    span_score = record.get("citation_weighted_span_score", support) or 0.0
    error_type = record.get("error_type")
    method = record.get("method", "")
    model_calls = record.get("model_calls", 1) or 1
    pred_answer = record.get("predicted_answer", "") or ""

    answer_ok = acc >= tolerance_accuracy

    if cit_recall == 0.0 and not answer_ok and error_type in ("no_citation", None, "empty_answer"):
        return "retrieval_miss"
    if 0.0 < span_score < 0.3 and not answer_ok:
        return "insufficient_evidence"
    if cit_f1 >= 0.4 and not answer_ok:
        return "reasoning_failure"
    if answer_ok and cit_f1 < 0.3:
        return "citation_failure"
    if len(pred_answer.split()) > 150 and span_score < 0.2 and not answer_ok:
        return "over_generation"
    if not answer_ok:
        return "reasoning_failure"
    # This is not a retrieval failure. It means Method C reached an acceptable
    # answer but used a costly trace/read loop, so keep it after correctness and
    # citation failures instead of letting it mask real errors.
    if method == "C" and model_calls > 8:
        return "trace_inefficiency"
    return None


def latency_stats(values: List[float]) -> Dict:
    if not values:
        return {}
    sv = sorted(values)
    n = len(sv)
    return {
        "mean": round(sum(sv) / n, 3),
        "std": round(statistics.stdev(sv) if n > 1 else 0.0, 3),
        "median": round(statistics.median(sv), 3),
        "p75": round(sv[int(n * 0.75)], 3),
        "p95": round(sv[min(int(n * 0.95), n - 1)], 3),
        "max": round(sv[-1], 3),
    }


def aggregate_metrics(records: List[Dict]) -> Dict:
    def mean_of(field):
        vals = [r[field] for r in records if r.get(field) is not None]
        return round(sum(vals) / len(vals), 4) if vals else None

    agg = {field: mean_of(field) for field in [
        "answer_score",
        "citation_file_precision",
        "citation_file_recall",
        "citation_file_f1",
        "citation_containment_recall",
        "citation_span_precision",
        "citation_span_f2",
        "citation_weighted_span_score",
        "evidence_line_iou",
        "citation_support_score",
        "citation_support_llm",
        "bertscore_precision",
        "bertscore_recall",
        "bertscore_f1",
        "latency_sec",
        "input_tokens",
        "output_tokens",
        "model_calls",
    ]}
    agg["latency_stats"] = latency_stats([r["latency_sec"] for r in records if r.get("latency_sec") is not None])
    agg["n"] = len(records)
    return agg


def bootstrap_test(scores_a: List[float], scores_b: List[float], n_iter: int = 10000, seed: int = 42) -> Dict:
    random.seed(seed)
    n = min(len(scores_a), len(scores_b))
    pairs = list(zip(scores_a[:n], scores_b[:n]))
    observed_diff = sum(b - a for a, b in pairs) / n
    boot_diffs = []
    for _ in range(n_iter):
        sample = random.choices(pairs, k=n)
        boot_diffs.append(sum(b - a for a, b in sample) / n)

    if observed_diff > 0:
        p_value = 2 * sum(1 for d in boot_diffs if d <= 0) / n_iter
    else:
        p_value = 2 * sum(1 for d in boot_diffs if d >= 0) / n_iter
    p_value = min(1.0, p_value)

    ci_lo = sorted(boot_diffs)[int(n_iter * 0.025)]
    ci_hi = sorted(boot_diffs)[int(n_iter * 0.975)]
    return {
        "observed_diff": round(observed_diff, 4),
        "p_value": round(p_value, 4),
        "ci_95_lo": round(ci_lo, 4),
        "ci_95_hi": round(ci_hi, 4),
        "significant_p05": p_value < 0.05,
        "n_pairs": n
    }


def load_scored(scored_dir: str) -> Dict[str, List[Dict]]:
    by_method: Dict[str, List[Dict]] = {}
    files = glob.glob(os.path.join(scored_dir, "scored_*.jsonl"))
    if not files:
        raise FileNotFoundError(f"No scored_*.jsonl found in {scored_dir}")
    for path in files:
        with open(path, "r", encoding="utf-8") as f:
            records = [json.loads(line) for line in f if line.strip()]
        if not records:
            continue
        ensure_required_fields(records[0], REQUIRED_SCORED_FIELDS, f"scored record in {path}")
        method = records[0]["method"]
        by_method.setdefault(method, []).extend(records)
    return by_method


def run_analysis(scored_dir: str, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    by_method = load_scored(scored_dir)

    analysis = {}
    def fmt(value):
        return "N/A" if value is None else f"{value:.3f}"

    for method, records in sorted(by_method.items()):
        agg = aggregate_metrics(records)
        error_counts = {
            "retrieval_miss": 0,
            "insufficient_evidence": 0,
            "reasoning_failure": 0,
            "citation_failure": 0,
            "over_generation": 0,
            "trace_inefficiency": 0,
            "success": 0,
        }
        for record in records:
            cat = categorize_error(record)
            if cat is None:
                error_counts["success"] += 1
            else:
                error_counts[cat] += 1

        print(f"\n--- Method {method} (N={len(records)}) ---")
        print(f"  answer_score      : {fmt(agg.get('answer_score'))}")
        print(f"  citation_file_f1  : {fmt(agg.get('citation_file_f1'))}")
        if agg.get("citation_weighted_span_score") is not None:
            print(f"  containment_recall: {fmt(agg.get('citation_containment_recall'))}")
            print(f"  span_precision    : {fmt(agg.get('citation_span_precision'))}")
            print(f"  citation_span_f2  : {fmt(agg.get('citation_span_f2'))}")
            print(f"  weighted_span     : {fmt(agg.get('citation_weighted_span_score'))}")
        if agg.get("bertscore_f1") is not None:
            print(f"  bertscore_f1      : {fmt(agg.get('bertscore_f1'))}")
        print(f"  citation_support  : {fmt(agg.get('citation_support_score'))} (legacy)")
        print(f"  Latency           : {agg['latency_stats']}")
        print("  Error breakdown   :")
        for cat, cnt in error_counts.items():
            pct = cnt / len(records) * 100
            print(f"    {cat:20s} {cnt:3d} ({pct:.1f}%)")

        analysis[method] = {"aggregate": agg, "error_breakdown": error_counts}

    bootstrap_results = {}
    methods = sorted(by_method.keys())
    pairs_to_test = [(a, b) for i, a in enumerate(methods) for b in methods[i + 1:]]
    for metric in ["answer_score", "citation_file_f1", "citation_span_f2"]:
        bootstrap_results[metric] = {}
        for m1, m2 in pairs_to_test:
            dict1 = {r["id"]: (r.get(metric, 0.0) or 0.0) for r in by_method[m1]}
            dict2 = {r["id"]: (r.get(metric, 0.0) or 0.0) for r in by_method[m2]}
            common_ids = sorted(set(dict1).intersection(dict2))
            if not common_ids:
                continue
            result = bootstrap_test([dict1[qid] for qid in common_ids], [dict2[qid] for qid in common_ids])
            bootstrap_results[metric][f"{m1}_vs_{m2}"] = result
    analysis["bootstrap_tests"] = bootstrap_results

    out_path = os.path.join(output_dir, f"analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(analysis, f, indent=2)
    print(f"\nFull analysis saved to {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Error analysis + statistical tests")
    parser.add_argument("--scored", type=str, default="results/scored")
    parser.add_argument("--output", type=str, default="results/analysis")
    args = parser.parse_args()

    if os.path.basename(os.getcwd()) == "src":
        os.chdir("..")

    run_analysis(args.scored, args.output)


if __name__ == "__main__":
    main()
