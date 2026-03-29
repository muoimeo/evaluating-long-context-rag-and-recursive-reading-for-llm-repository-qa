"""
analyze.py — Error Analysis + Statistical Tests

Reads scored results from score.py and produces:
  1. Error categorization table (6 error types) per method
  2. Full latency statistics (mean, std, median, p75, p95, max)
  3. Pairwise bootstrap significance tests (A vs B, B vs C, A vs C)
     for accuracy_primary and citation_file_f1

Usage:
  python src/analyze.py --scored results/scored/ --output results/analysis/
"""
import os
import json
import glob
import random
import argparse
import statistics
from typing import List, Dict, Any, Tuple
from datetime import datetime


# ---------------------------------------------------------------------------
# Error Categorization
# ---------------------------------------------------------------------------

def categorize_error(record: Dict, tolerance_accuracy: float = 0.4) -> str:
    """
    Assign one of 6 error categories based on record fields.

    Priority order:
      1. retrieval_miss     — no citation at all, answer wrong/empty
      2. insufficient_evidence — some citation but citation_support_score < 0.3
      3. citation_failure   — answer ok but citation missing/wrong
      4. reasoning_failure  — citations retrieved but answer wrong
      5. over_generation    — answer exceeds evidence (heuristic: long answer, low citation)
      6. trace_inefficiency — (Method C) too many model calls
    """
    acc = record.get("accuracy_primary", 0.0) or 0.0
    cit_recall = record.get("citation_file_recall", 0.0) or 0.0
    cit_f1 = record.get("citation_file_f1", 0.0) or 0.0
    support = record.get("citation_support_score", 0.0) or 0.0
    error_type = record.get("error_type")
    method = record.get("method", "")
    model_calls = record.get("model_calls", 1) or 1
    pred_answer = record.get("predicted_answer", "") or ""

    answer_ok = acc >= tolerance_accuracy

    # Tool/trace inefficiency (Method C with excessive calls)
    if method == "C" and model_calls > 8:
        return "trace_inefficiency"

    # No citation at all + answer wrong → retrieval miss
    if cit_recall == 0.0 and not answer_ok and error_type in ("no_citation", None):
        return "retrieval_miss"

    # Partial retrieval — some evidence but not enough for multi-hop
    if 0.0 < support < 0.3 and not answer_ok:
        return "insufficient_evidence"

    # Good citation but wrong answer
    if cit_f1 >= 0.4 and not answer_ok:
        return "reasoning_failure"

    # Good answer but missing citations
    if answer_ok and cit_f1 < 0.3:
        return "citation_failure"

    # Answer long but citation score low (over-generation heuristic)
    word_count = len(pred_answer.split())
    if word_count > 150 and support < 0.2 and not answer_ok:
        return "over_generation"

    # Fallback: if poor on both → reasoning failure
    if not answer_ok:
        return "reasoning_failure"

    return None  # No error — success


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def latency_stats(values: List[float]) -> Dict:
    if not values:
        return {}
    sv = sorted(values)
    n = len(sv)
    return {
        "mean":   round(sum(sv) / n, 3),
        "std":    round(statistics.stdev(sv) if n > 1 else 0.0, 3),
        "median": round(statistics.median(sv), 3),
        "p75":    round(sv[int(n * 0.75)], 3),
        "p95":    round(sv[min(int(n * 0.95), n - 1)], 3),
        "max":    round(sv[-1], 3),
    }


def aggregate_metrics(records: List[Dict]) -> Dict:
    """Compute mean of all numeric metric fields."""
    n = len(records)
    if n == 0:
        return {}

    def mean_of(field):
        vals = [r[field] for r in records if r.get(field) is not None]
        return round(sum(vals) / len(vals), 4) if vals else None

    fields = ["accuracy_llmjudge", "accuracy_em", "accuracy_primary",
              "citation_file_precision", "citation_file_recall", "citation_file_f1",
              "line_iou", "citation_support_score",
              "latency_sec", "input_tokens", "output_tokens", "model_calls"]
    agg = {f: mean_of(f) for f in fields}
    agg["latency_stats"] = latency_stats([r["latency_sec"] for r in records if r.get("latency_sec") is not None])
    agg["n"] = n
    return agg


# ---------------------------------------------------------------------------
# Paired Bootstrap Significance Test
# ---------------------------------------------------------------------------

def bootstrap_test(scores_a: List[float], scores_b: List[float],
                   n_iter: int = 10000, seed: int = 42) -> Dict:
    """
    Paired bootstrap test: H0 = mean(B) - mean(A) = 0.
    Returns p-value (one-sided: B > A) and 95% CI for the difference.
    """
    random.seed(seed)
    n = min(len(scores_a), len(scores_b))
    pairs = list(zip(scores_a[:n], scores_b[:n]))
    observed_diff = sum(b - a for a, b in pairs) / n
    boot_diffs = []
    for _ in range(n_iter):
        sample = random.choices(pairs, k=n)
        d = sum(b - a for a, b in sample) / n
        boot_diffs.append(d)
        
    # Calculate two-sided empirical p-value testing H0: mean(B - A) = 0
    if observed_diff > 0:
        p_value = 2 * sum(1 for d in boot_diffs if d <= 0) / n_iter
    else:
        p_value = 2 * sum(1 for d in boot_diffs if d >= 0) / n_iter
    p_value = min(1.0, p_value)  # Cap at 1.0

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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def load_scored(scored_dir: str) -> Dict[str, List[Dict]]:
    """Load all scored JSON files, grouped by method."""
    by_method = {}
    for path in glob.glob(os.path.join(scored_dir, "scored_*.json")):
        with open(path, "r", encoding="utf-8") as f:
            records = json.load(f)
        if records:
            method = records[0].get("method", "?")
            by_method.setdefault(method, []).extend(records)
    return by_method


def run_analysis(scored_dir: str, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    by_method = load_scored(scored_dir)

    if not by_method:
        print(f"No scored_*.json found in {scored_dir}")
        return

    analysis = {}

    # Per-method: aggregate metrics + error breakdown
    for method, records in sorted(by_method.items()):
        print(f"\n--- Method {method} (N={len(records)}) ---")
        agg = aggregate_metrics(records)

        # Error categorization
        error_counts = {
            "retrieval_miss": 0,
            "insufficient_evidence": 0,
            "reasoning_failure": 0,
            "citation_failure": 0,
            "over_generation": 0,
            "trace_inefficiency": 0,
            "success": 0,
        }
        for r in records:
            cat = categorize_error(r)
            if cat is None:
                error_counts["success"] += 1
            else:
                error_counts[cat] = error_counts.get(cat, 0) + 1

        n = len(records)
        print(f"  accuracy_primary  : {agg.get('accuracy_primary'):.3f}")
        print(f"  citation_file_f1  : {agg.get('citation_file_f1'):.3f}")
        print(f"  citation_support  : {agg.get('citation_support_score'):.3f}")
        print(f"  Latency          : {agg['latency_stats']}")
        print(f"  Error breakdown  :")
        for cat, cnt in error_counts.items():
            pct = cnt / n * 100
            print(f"    {cat:30s} {cnt:3d} ({pct:.1f}%)")

        analysis[method] = {"aggregate": agg, "error_breakdown": error_counts}

    # Pairwise bootstrap tests
    print("\n--- Pairwise Bootstrap Tests (10,000 iterations) ---")
    bootstrap_results = {}
    methods = sorted(by_method.keys())
    pairs_to_test = [(a, b) for i, a in enumerate(methods) for b in methods[i+1:]]

    for metric in ["accuracy_primary", "citation_file_f1"]:
        bootstrap_results[metric] = {}
        for m1, m2 in pairs_to_test:
            # Pair scores precisely by question 'id'
            dict1 = {r["id"]: (r.get(metric, 0.0) or 0.0) for r in by_method[m1] if "id" in r}
            dict2 = {r["id"]: (r.get(metric, 0.0) or 0.0) for r in by_method[m2] if "id" in r}
            
            common_ids = sorted(list(set(dict1.keys()).intersection(set(dict2.keys()))))
            if not common_ids:
                print(f"  Warning: No common IDs found for {m1} vs {m2} on {metric}.")
                continue
                
            s1 = [dict1[qid] for qid in common_ids]
            s2 = [dict2[qid] for qid in common_ids]
            
            result = bootstrap_test(s1, s2)
            key = f"{m1}_vs_{m2}"
            bootstrap_results[metric][key] = result
            sig = "✓ p<0.05" if result["significant_p05"] else "✗ not significant"
            print(f"  {metric} | {m1} vs {m2}: diff={result['observed_diff']:+.4f}  "
                  f"p={result['p_value']:.4f}  95%CI=[{result['ci_95_lo']:.4f},{result['ci_95_hi']:.4f}]  {sig}")

    analysis["bootstrap_tests"] = bootstrap_results

    # Save full analysis
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
