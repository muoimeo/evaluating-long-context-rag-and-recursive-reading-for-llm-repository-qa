"""
context_rot_deep_analysis.py - deeper post-hoc analysis for context-rot runs.

This script reads scored context-rot JSONL files and produces discussion-ready
tables. It is intentionally separate from context_rot_eval.py: experiment
execution should stay controlled, while this file interprets already-scored
outputs.

Usage:
  python src/context_rot_deep_analysis.py ^
    --scored results_context_rot/phase1_semantic_v2/scored/context_rot_scored_*.jsonl ^
             results_context_rot/phase1_same_file_v2/scored/context_rot_scored_*.jsonl ^
             results_context_rot/phase1_inverted_gold_v2/scored/context_rot_scored_*.jsonl ^
             results_context_rot/phase2_semantic_60/scored/context_rot_scored_*.jsonl ^
    --output-dir results_context_rot/deep_analysis
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
from collections import Counter, defaultdict
from datetime import datetime
from typing import Any, Dict, Iterable, List, Tuple


METRICS = [
    "answer_score",
    "citation_file_precision",
    "citation_file_recall",
    "citation_file_f1",
    "citation_containment_recall",
    "citation_span_precision",
    "citation_span_f2",
    "citation_weighted_span_score",
    "bertscore_f1",
    "latency_sec",
    "model_calls",
]

LEVEL_ORDER = {"L0": 0, "L1": 1, "L2": 2, "L3": 3}


def expand_paths(patterns: Iterable[str]) -> List[str]:
    paths: List[str] = []
    for pattern in patterns:
        matches = glob.glob(pattern)
        paths.extend(matches if matches else [pattern])
    deduped = []
    seen = set()
    for path in paths:
        norm = os.path.normpath(path)
        if norm not in seen:
            deduped.append(norm)
            seen.add(norm)
    return deduped


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                row = json.loads(line)
                row["_source_file"] = path
                row["_run_name"] = infer_run_name(path)
                rows.append(row)
    return rows


def infer_run_name(path: str) -> str:
    parts = os.path.normpath(path).split(os.sep)
    if "results_context_rot" in parts:
        idx = parts.index("results_context_rot")
        if idx + 1 < len(parts):
            return parts[idx + 1]
    return os.path.basename(os.path.dirname(os.path.dirname(path))) or "run"


def mean(rows: List[Dict[str, Any]], field: str) -> float | None:
    vals = [r.get(field) for r in rows if isinstance(r.get(field), (int, float))]
    return sum(vals) / len(vals) if vals else None


def fmt(value: float | None) -> str:
    return "NA" if value is None else f"{value:.3f}"


def aggregate(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    out: Dict[str, Any] = {"n": len(rows)}
    ids = {r.get("id") for r in rows}
    out["n_questions"] = len(ids)
    for metric in METRICS:
        out[metric] = mean(rows, metric)
    return out


def group_summary(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[str, str, str, str], List[Dict[str, Any]]] = defaultdict(list)
    for r in rows:
        key = (
            r.get("_run_name", "run"),
            r.get("noise_mode", "unknown"),
            r.get("method", "unknown"),
            r.get("context_level", "unknown"),
        )
        grouped[key].append(r)

    summaries: List[Dict[str, Any]] = []
    baseline: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
    for (run, noise, method, level), items in sorted(grouped.items(), key=sort_group_key):
        row = {
            "run": run,
            "noise_mode": noise,
            "method": method,
            "context_level": level,
            "context_size": mean(items, "context_size"),
            "noise_count": mean(items, "noise_count"),
            **aggregate(items),
        }
        summaries.append(row)
        if level == "L0":
            baseline[(run, noise, method)] = row

    for row in summaries:
        base = baseline.get((row["run"], row["noise_mode"], row["method"]))
        for metric in ["answer_score", "citation_file_f1", "citation_weighted_span_score"]:
            key = f"delta_{metric}_vs_L0"
            row[key] = None if not base or row.get(metric) is None or base.get(metric) is None else row[metric] - base[metric]
    return summaries


def sort_group_key(item: Tuple[Tuple[str, str, str, str], Any]) -> Tuple[Any, ...]:
    run, noise, method, level = item[0]
    return (run, noise, method, LEVEL_ORDER.get(level, 99), level)


def classify_failure(record: Dict[str, Any]) -> str:
    """Assign one primary failure mode for context-rot discussion.

    This is heuristic, not a replacement for manual error analysis. It separates
    likely retrieval/selection problems from line localization and synthesis
    failures using the already-computed score fields.
    """
    answer = record.get("answer_score") or 0.0
    file_recall = record.get("citation_file_recall") or 0.0
    file_precision = record.get("citation_file_precision") or 0.0
    file_f1 = record.get("citation_file_f1") or 0.0
    span = record.get("citation_weighted_span_score") or 0.0
    span_precision = record.get("citation_span_precision") or 0.0
    noise_count = record.get("noise_count") or 0

    answer_ok = answer >= 0.5
    if answer_ok and file_recall >= 0.5 and span >= 0.3:
        return "success_or_acceptable"
    if noise_count > 0 and file_recall < 0.3:
        return "distractor_selection_failure"
    if file_recall < 0.3:
        return "retrieval_or_selection_failure"
    if file_recall >= 0.5 and span < 0.3:
        return "evidence_localization_failure"
    if not answer_ok and (file_f1 >= 0.4 or span >= 0.3):
        return "synthesis_reasoning_failure"
    if answer_ok and (file_precision < 0.5 or span_precision < 0.2):
        return "citation_overprediction_or_wrong_span"
    if not answer_ok:
        return "answer_failure_unclear"
    return "citation_grounding_failure"


def failure_taxonomy(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[str, str, str, str], Counter] = defaultdict(Counter)
    totals: Counter = Counter()
    for r in rows:
        key = (
            r.get("_run_name", "run"),
            r.get("noise_mode", "unknown"),
            r.get("method", "unknown"),
            r.get("context_level", "unknown"),
        )
        cat = classify_failure(r)
        grouped[key][cat] += 1
        totals[key] += 1

    out: List[Dict[str, Any]] = []
    for key, counts in sorted(grouped.items(), key=lambda kv: sort_group_key((kv[0], None))):
        run, noise, method, level = key
        n = totals[key]
        row = {
            "run": run,
            "noise_mode": noise,
            "method": method,
            "context_level": level,
            "n": n,
        }
        for cat, count in sorted(counts.items()):
            row[cat] = count
            row[f"{cat}_pct"] = count / n if n else 0.0
        out.append(row)
    return out


def b_vs_c(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    by_key: Dict[Tuple[str, str, str, str], Dict[str, Any]] = {}
    for r in rows:
        if r.get("method") in ("B", "C"):
            key = (r.get("_run_name", "run"), r.get("noise_mode", "unknown"), r.get("context_level", "unknown"), r.get("id"))
            by_key[(key[0], key[1], key[2], key[3], r["method"])] = r

    grouped: Dict[Tuple[str, str, str], List[Tuple[Dict[str, Any], Dict[str, Any]]]] = defaultdict(list)
    base_keys = {(k[0], k[1], k[2], k[3]) for k in by_key}
    for run, noise, level, qid in base_keys:
        b = by_key.get((run, noise, level, qid, "B"))
        c = by_key.get((run, noise, level, qid, "C"))
        if b and c:
            grouped[(run, noise, level)].append((b, c))

    out: List[Dict[str, Any]] = []
    for (run, noise, level), pairs in sorted(grouped.items(), key=lambda kv: (kv[0][0], kv[0][1], LEVEL_ORDER.get(kv[0][2], 99))):
        answer_gaps = [(c.get("answer_score") or 0.0) - (b.get("answer_score") or 0.0) for b, c in pairs]
        span_gaps = [(c.get("citation_weighted_span_score") or 0.0) - (b.get("citation_weighted_span_score") or 0.0) for b, c in pairs]
        file_gaps = [(c.get("citation_file_f1") or 0.0) - (b.get("citation_file_f1") or 0.0) for b, c in pairs]
        row = {
            "run": run,
            "noise_mode": noise,
            "context_level": level,
            "n_pairs": len(pairs),
            "c_answer_wins": sum(1 for x in answer_gaps if x > 0),
            "b_answer_wins": sum(1 for x in answer_gaps if x < 0),
            "answer_ties": sum(1 for x in answer_gaps if x == 0),
            "mean_answer_gap_c_minus_b": sum(answer_gaps) / len(answer_gaps),
            "mean_weighted_span_gap_c_minus_b": sum(span_gaps) / len(span_gaps),
            "mean_file_f1_gap_c_minus_b": sum(file_gaps) / len(file_gaps),
        }
        out.append(row)
    return out


def same_file_utility(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    same = [r for r in rows if r.get("noise_mode") == "same_file_noise"]
    by_key: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
    for r in same:
        by_key[(r.get("_run_name", "run"), r.get("method", "unknown"), r.get("id", "")) + (r.get("context_level", "unknown"),)] = r

    grouped: Dict[Tuple[str, str, str], List[Tuple[Dict[str, Any], Dict[str, Any]]]] = defaultdict(list)
    base_items = [r for r in same if r.get("context_level") == "L0"]
    for base in base_items:
        for level in ("L1", "L2", "L3"):
            cur = by_key.get((base.get("_run_name", "run"), base.get("method", "unknown"), base.get("id", ""), level))
            if cur:
                grouped[(base.get("_run_name", "run"), base.get("method", "unknown"), level)].append((base, cur))

    out: List[Dict[str, Any]] = []
    for (run, method, level), pairs in sorted(grouped.items(), key=lambda kv: (kv[0][0], kv[0][1], LEVEL_ORDER.get(kv[0][2], 99))):
        answer_deltas = [(cur.get("answer_score") or 0.0) - (base.get("answer_score") or 0.0) for base, cur in pairs]
        span_deltas = [(cur.get("citation_weighted_span_score") or 0.0) - (base.get("citation_weighted_span_score") or 0.0) for base, cur in pairs]
        file_deltas = [(cur.get("citation_file_f1") or 0.0) - (base.get("citation_file_f1") or 0.0) for base, cur in pairs]
        out.append({
            "run": run,
            "method": method,
            "context_level": level,
            "n_pairs": len(pairs),
            "answer_improved": sum(1 for x in answer_deltas if x > 0),
            "answer_degraded": sum(1 for x in answer_deltas if x < 0),
            "answer_unchanged": sum(1 for x in answer_deltas if x == 0),
            "span_improved": sum(1 for x in span_deltas if x > 0),
            "span_degraded": sum(1 for x in span_deltas if x < 0),
            "span_unchanged": sum(1 for x in span_deltas if x == 0),
            "mean_answer_delta": sum(answer_deltas) / len(answer_deltas),
            "mean_weighted_span_delta": sum(span_deltas) / len(span_deltas),
            "mean_file_f1_delta": sum(file_deltas) / len(file_deltas),
        })
    return out


def validity_flags(rows: List[Dict[str, Any]]) -> List[str]:
    flags: List[str] = []
    by_run: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in rows:
        by_run[r.get("_run_name", "run")].append(r)
    for run, items in sorted(by_run.items()):
        ids = {r.get("id") for r in items}
        noise_modes = {r.get("noise_mode") for r in items}
        levels = {r.get("context_level") for r in items}
        methods = {r.get("method") for r in items}
        if len(ids) < 30:
            flags.append(f"{run}: small pilot subset ({len(ids)} questions); do not generalize strongly.")
        if "semantic" in run and noise_modes and noise_modes != {"semantic_noise"}:
            flags.append(f"{run}: folder name suggests semantic noise but actual noise modes are {sorted(noise_modes)}.")
        if "same_file_noise" in noise_modes:
            flags.append(f"{run}: same-file noise may be local context expansion, not pure distractor noise.")
        if "L1" not in levels:
            flags.append(f"{run}: missing L1; degradation curve is not fully observed.")
        if methods == {"B", "C"}:
            flags.append(f"{run}: excludes Method A; compare only B vs C for this phase.")
    return flags


def write_csv(path: str, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    keys: List[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def write_txt(path: str, summary_rows: List[Dict[str, Any]], failures: List[Dict[str, Any]], gaps: List[Dict[str, Any]], same_file: List[Dict[str, Any]], flags: List[str]) -> None:
    lines: List[str] = []
    lines.append("Context-Rot Deep Analysis")
    lines.append("=" * 80)
    lines.append("")
    lines.append("Group summary")
    lines.append("run                          noise              method level n   answer file_f1 span   d_ans  d_file d_span  calls latency")
    for r in summary_rows:
        lines.append(
            f"{r['run'][:28]:28s} {r['noise_mode'][:18]:18s} {r['method']:6s} {r['context_level']:5s} "
            f"{int(r['n']):3d} {fmt(r.get('answer_score')):>6s} {fmt(r.get('citation_file_f1')):>7s} "
            f"{fmt(r.get('citation_weighted_span_score')):>6s} {fmt(r.get('delta_answer_score_vs_L0')):>6s} "
            f"{fmt(r.get('delta_citation_file_f1_vs_L0')):>6s} {fmt(r.get('delta_citation_weighted_span_score_vs_L0')):>6s} "
            f"{fmt(r.get('model_calls')):>6s} {fmt(r.get('latency_sec')):>7s}"
        )
    lines.append("")
    lines.append("B vs C paired gaps")
    lines.append("run                          noise              level n  C_ans_win B_ans_win tie  gap_ans gap_file gap_span")
    for r in gaps:
        lines.append(
            f"{r['run'][:28]:28s} {r['noise_mode'][:18]:18s} {r['context_level']:5s} "
            f"{int(r['n_pairs']):2d} {int(r['c_answer_wins']):10d} {int(r['b_answer_wins']):9d} {int(r['answer_ties']):3d} "
            f"{fmt(r.get('mean_answer_gap_c_minus_b')):>7s} {fmt(r.get('mean_file_f1_gap_c_minus_b')):>8s} "
            f"{fmt(r.get('mean_weighted_span_gap_c_minus_b')):>8s}"
        )
    if same_file:
        lines.append("")
        lines.append("Same-file utility check")
        lines.append("method level n  ans_up ans_down ans_same span_up span_down span_same mean_d_ans mean_d_span")
        for r in same_file:
            lines.append(
                f"{r['method']:6s} {r['context_level']:5s} {int(r['n_pairs']):2d} "
                f"{int(r['answer_improved']):6d} {int(r['answer_degraded']):8d} {int(r['answer_unchanged']):8d} "
                f"{int(r['span_improved']):7d} {int(r['span_degraded']):9d} {int(r['span_unchanged']):9d} "
                f"{fmt(r.get('mean_answer_delta')):>10s} {fmt(r.get('mean_weighted_span_delta')):>11s}"
            )
    lines.append("")
    lines.append("Failure taxonomy")
    for r in failures:
        cats = {k: v for k, v in r.items() if k not in {"run", "noise_mode", "method", "context_level", "n"} and not k.endswith("_pct")}
        top = sorted(cats.items(), key=lambda kv: kv[1], reverse=True)[:3]
        top_s = ", ".join(f"{name}={count}" for name, count in top)
        lines.append(f"{r['run']} | {r['noise_mode']} | {r['method']} | {r['context_level']} | n={r['n']} | {top_s}")
    lines.append("")
    lines.append("Threats to validity / caveats")
    for flag in flags:
        lines.append(f"- {flag}")
    lines.append("")
    lines.append("Interpretation guidance")
    lines.append("- Same-file noise should be described as same-file context expansion unless manual inspection proves the added chunks are pure distractors.")
    lines.append("- Inverted-gold failures mostly test adversarial evidence selection, not ordinary retrieval difficulty.")
    lines.append("- BERTScore should not be used as the main correctness signal; in these runs it is weakly sensitive to inverted logic.")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description="Deep analysis for scored context-rot outputs")
    parser.add_argument("--scored", nargs="+", required=True, help="Scored JSONL files or glob patterns")
    parser.add_argument("--output-dir", default="results_context_rot/deep_analysis")
    args = parser.parse_args()

    paths = expand_paths(args.scored)
    if not paths:
        raise FileNotFoundError("No scored files matched.")
    rows: List[Dict[str, Any]] = []
    for path in paths:
        rows.extend(load_jsonl(path))
    if not rows:
        raise ValueError("No records loaded from scored files.")

    os.makedirs(args.output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    summaries = group_summary(rows)
    failures = failure_taxonomy(rows)
    gaps = b_vs_c(rows)
    same = same_file_utility(rows)
    flags = validity_flags(rows)

    prefix = os.path.join(args.output_dir, f"context_rot_deep_analysis_{timestamp}")
    write_csv(prefix + "_group_summary.csv", summaries)
    write_csv(prefix + "_failure_taxonomy.csv", failures)
    write_csv(prefix + "_b_vs_c.csv", gaps)
    write_csv(prefix + "_same_file_utility.csv", same)
    write_txt(prefix + ".txt", summaries, failures, gaps, same, flags)
    with open(prefix + ".json", "w", encoding="utf-8") as f:
        json.dump({
            "source_files": paths,
            "n_records": len(rows),
            "group_summary": summaries,
            "failure_taxonomy": failures,
            "b_vs_c": gaps,
            "same_file_utility": same,
            "validity_flags": flags,
        }, f, indent=2)

    print(f"Loaded {len(rows)} records from {len(paths)} files.")
    print(f"Wrote {prefix}.txt")
    print(f"Wrote CSV/JSON tables with prefix {prefix}")


if __name__ == "__main__":
    main()
