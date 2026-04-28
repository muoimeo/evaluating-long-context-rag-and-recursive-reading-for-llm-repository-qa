from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List


def aggregate_context_rot_scores(scored_file: str, output_json: str | None = None) -> List[Dict[str, Any]]:
    groups: Dict[tuple[str, str, str], List[Dict[str, Any]]] = defaultdict(list)
    with open(scored_file, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)
            groups[(record.get("noise_mode", "mixed_noise"), record["method"], record["context_level"])].append(record)

    def mean(records: List[Dict[str, Any]], field: str) -> float | None:
        values = [record.get(field) for record in records if record.get(field) is not None]
        return round(sum(values) / len(values), 4) if values else None

    table = []
    for (noise_mode, method, level), records in sorted(groups.items()):
        table.append({
            "noise_mode": noise_mode,
            "method": method,
            "context_level": level,
            "context_size": mean(records, "context_size"),
            "context_tokens_est": mean(records, "context_tokens_est"),
            "n": len(records),
            "answer_score": mean(records, "answer_score"),
            "bertscore_f1": mean(records, "bertscore_f1"),
            "citation_weighted_span_score": mean(records, "citation_weighted_span_score"),
            "citation_span_f2": mean(records, "citation_span_f2"),
            "citation_file_f1": mean(records, "citation_file_f1"),
            "latency_sec": mean(records, "latency_sec"),
            "model_calls": mean(records, "model_calls"),
        })

    if output_json:
        Path(output_json).parent.mkdir(parents=True, exist_ok=True)
        with open(output_json, "w", encoding="utf-8") as f:
            json.dump(table, f, indent=2)
    return table


def _safe_delta(value: float | None, baseline: float | None) -> float | None:
    if value is None or baseline is None:
        return None
    return round(value - baseline, 4)


def build_context_rot_report(scored_file: str) -> Dict[str, Any]:
    with open(scored_file, "r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]

    metrics = [
        "answer_score",
        "bertscore_f1",
        "citation_weighted_span_score",
        "citation_span_f2",
        "citation_file_f1",
        "citation_file_precision",
        "citation_file_recall",
        "citation_containment_recall",
        "citation_span_precision",
        "evidence_line_iou",
        "citation_support_score",
        "latency_sec",
        "model_calls",
        "context_size",
        "context_tokens_est",
    ]

    def mean(items: List[Dict[str, Any]], field: str) -> float | None:
        values = [item.get(field) for item in items if isinstance(item.get(field), (int, float))]
        return round(sum(values) / len(values), 4) if values else None

    grouped: Dict[tuple[str, str, str], List[Dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[(
            record.get("noise_mode", "unknown"),
            record.get("method", "unknown"),
            record.get("context_level", "unknown"),
        )].append(record)

    by_group = []
    baseline_by_noise_method: Dict[tuple[str, str], Dict[str, Any]] = {}
    for (noise_mode, method, level), items in sorted(grouped.items()):
        row = {
            "noise_mode": noise_mode,
            "method": method,
            "context_level": level,
            "n": len(items),
            **{metric: mean(items, metric) for metric in metrics},
        }
        by_group.append(row)
        if level == "L0":
            baseline_by_noise_method[(noise_mode, method)] = row

    for row in by_group:
        baseline = baseline_by_noise_method.get((row["noise_mode"], row["method"]), {})
        row["delta_from_L0"] = {
            metric: _safe_delta(row.get(metric), baseline.get(metric))
            for metric in [
                "answer_score",
                "bertscore_f1",
                "citation_weighted_span_score",
                "citation_span_f2",
                "citation_file_f1",
                "latency_sec",
                "model_calls",
            ]
        }

    methods_by_noise_level: Dict[tuple[str, str], Dict[str, Dict[str, Any]]] = defaultdict(dict)
    for row in by_group:
        methods_by_noise_level[(row["noise_mode"], row["context_level"])][row["method"]] = row

    method_gaps = []
    for (noise_mode, level), methods in sorted(methods_by_noise_level.items()):
        if "B" in methods and "C" in methods:
            b, c = methods["B"], methods["C"]
            method_gaps.append({
                "noise_mode": noise_mode,
                "context_level": level,
                "comparison": "C_minus_B",
                "answer_score": _safe_delta(c.get("answer_score"), b.get("answer_score")),
                "bertscore_f1": _safe_delta(c.get("bertscore_f1"), b.get("bertscore_f1")),
                "citation_weighted_span_score": _safe_delta(c.get("citation_weighted_span_score"), b.get("citation_weighted_span_score")),
                "citation_file_f1": _safe_delta(c.get("citation_file_f1"), b.get("citation_file_f1")),
                "latency_sec": _safe_delta(c.get("latency_sec"), b.get("latency_sec")),
                "model_calls": _safe_delta(c.get("model_calls"), b.get("model_calls")),
            })

    per_question = []
    by_question_level: Dict[tuple[str, str], Dict[str, Dict[str, Any]]] = defaultdict(dict)
    for record in records:
        by_question_level[(record["id"], record.get("context_level", "unknown"))][record.get("method", "unknown")] = record
    for (qid, level), methods in sorted(by_question_level.items()):
        if "B" not in methods or "C" not in methods:
            continue
        b, c = methods["B"], methods["C"]
        per_question.append({
            "id": qid,
            "context_level": level,
            "noise_mode": b.get("noise_mode", c.get("noise_mode", "unknown")),
            "difficulty": b.get("difficulty"),
            "reasoning_type": b.get("reasoning_type"),
            "answer_score_B": b.get("answer_score"),
            "answer_score_C": c.get("answer_score"),
            "answer_delta_C_minus_B": _safe_delta(c.get("answer_score"), b.get("answer_score")),
            "bertscore_f1_B": b.get("bertscore_f1"),
            "bertscore_f1_C": c.get("bertscore_f1"),
            "bertscore_delta_C_minus_B": _safe_delta(c.get("bertscore_f1"), b.get("bertscore_f1")),
            "span_score_B": b.get("citation_weighted_span_score"),
            "span_score_C": c.get("citation_weighted_span_score"),
            "span_delta_C_minus_B": _safe_delta(c.get("citation_weighted_span_score"), b.get("citation_weighted_span_score")),
            "file_f1_B": b.get("citation_file_f1"),
            "file_f1_C": c.get("citation_file_f1"),
            "file_f1_delta_C_minus_B": _safe_delta(c.get("citation_file_f1"), b.get("citation_file_f1")),
        })

    return {
        "scored_file": scored_file,
        "n_records": len(records),
        "by_group": by_group,
        "method_gaps": method_gaps,
        "per_question_B_vs_C": per_question,
    }


def format_context_rot_report(report: Dict[str, Any]) -> str:
    lines = []
    lines.append("Context-Rot Analysis Report")
    lines.append("=" * 80)
    lines.append(f"Scored file: {report['scored_file']}")
    lines.append(f"Records: {report['n_records']}")
    lines.append("")

    lines.append("Averages by Noise / Method / Level")
    lines.append("-" * 80)
    header = (
        f"{'noise':15s} {'method':6s} {'level':5s} {'n':>3s} {'ctx':>6s} "
        f"{'answer':>8s} {'ans_d':>8s} {'bert':>8s} {'bert_d':>8s} {'span':>8s} {'span_d':>8s} "
        f"{'file_f1':>8s} {'file_d':>8s} {'lat':>8s} {'calls':>7s}"
    )
    lines.append(header)
    for row in report["by_group"]:
        delta = row.get("delta_from_L0", {})
        lines.append(
            f"{row['noise_mode'][:15]:15s} {row['method']:6s} {row['context_level']:5s} "
            f"{row['n']:3d} {row.get('context_size')!s:>6s} "
            f"{row.get('answer_score')!s:>8s} {delta.get('answer_score')!s:>8s} "
            f"{row.get('bertscore_f1')!s:>8s} {delta.get('bertscore_f1')!s:>8s} "
            f"{row.get('citation_weighted_span_score')!s:>8s} {delta.get('citation_weighted_span_score')!s:>8s} "
            f"{row.get('citation_file_f1')!s:>8s} {delta.get('citation_file_f1')!s:>8s} "
            f"{row.get('latency_sec')!s:>8s} {row.get('model_calls')!s:>7s}"
        )

    if report["method_gaps"]:
        lines.append("")
        lines.append("Method Gap: C minus B")
        lines.append("-" * 80)
        lines.append(
            f"{'noise':15s} {'level':5s} {'answer':>8s} {'bert':>8s} {'span':>8s} "
            f"{'file_f1':>8s} {'latency':>8s} {'calls':>8s}"
        )
        for row in report["method_gaps"]:
            lines.append(
                f"{row['noise_mode'][:15]:15s} {row['context_level']:5s} "
                f"{row.get('answer_score')!s:>8s} {row.get('bertscore_f1')!s:>8s} "
                f"{row.get('citation_weighted_span_score')!s:>8s} "
                f"{row.get('citation_file_f1')!s:>8s} {row.get('latency_sec')!s:>8s} "
                f"{row.get('model_calls')!s:>8s}"
            )

    interesting = [
        row for row in report["per_question_B_vs_C"]
        if row.get("answer_delta_C_minus_B") not in (None, 0)
        or abs(row.get("span_delta_C_minus_B") or 0) >= 0.15
        or abs(row.get("file_f1_delta_C_minus_B") or 0) >= 0.2
    ]
    if interesting:
        lines.append("")
        lines.append("Notable Per-Question B vs C Differences")
        lines.append("-" * 80)
        lines.append(
            f"{'id':18s} {'level':5s} {'type':24s} {'ansB':>5s} {'ansC':>5s} "
            f"{'dAns':>6s} {'dSpan':>7s} {'dFile':>7s}"
        )
        for row in interesting[:80]:
            lines.append(
                f"{row['id'][:18]:18s} {row['context_level']:5s} "
                f"{str(row.get('reasoning_type'))[:24]:24s} "
                f"{row.get('answer_score_B')!s:>5s} {row.get('answer_score_C')!s:>5s} "
                f"{row.get('answer_delta_C_minus_B')!s:>6s} "
                f"{row.get('span_delta_C_minus_B')!s:>7s} "
                f"{row.get('file_f1_delta_C_minus_B')!s:>7s}"
            )

    lines.append("")
    lines.append("Reading Guide")
    lines.append("-" * 80)
    lines.append("ans_d/span_d/file_d are degradation from L0 for the same method.")
    lines.append("C_minus_B > 0 means Method C is better on that metric; latency/calls > 0 means C is more expensive.")
    lines.append("BERTScore is semantic similarity to the reference answer, not correctness; disagreement with answer_score needs manual review.")
    lines.append("If citation stays high but answer drops, the failure is synthesis/reasoning, not retrieval.")
    return "\n".join(lines)


def write_context_rot_report(scored_file: str, output_prefix: str) -> Dict[str, str]:
    report = build_context_rot_report(scored_file)
    output_json = output_prefix + ".json"
    output_txt = output_prefix + ".txt"
    Path(output_json).parent.mkdir(parents=True, exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    rendered = format_context_rot_report(report)
    with open(output_txt, "w", encoding="utf-8") as f:
        f.write(rendered + "\n")
    print(rendered)
    return {"json": output_json, "txt": output_txt}


def print_context_rot_table(table: List[Dict[str, Any]]) -> None:
    header = ["noise", "method", "level", "ctx", "answer", "bert", "span", "file_f1", "lat", "calls", "n"]
    print("\t".join(header))
    for row in table:
        print("\t".join([
            str(row["noise_mode"]),
            str(row["method"]),
            str(row["context_level"]),
            str(row["context_size"]),
            str(row["answer_score"]),
            str(row.get("bertscore_f1")),
            str(row["citation_weighted_span_score"]),
            str(row["citation_file_f1"]),
            str(row["latency_sec"]),
            str(row["model_calls"]),
            str(row["n"]),
        ]))
