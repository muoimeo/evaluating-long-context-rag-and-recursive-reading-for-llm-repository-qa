"""
verify_evidence.py — Citation Scoring Functions

Implements a two-layer citation evaluation system:

Layer 1 (File-level):  citation_file_precision, citation_file_recall, citation_file_f1
Layer 2 (Line-level):  line_iou_per_file, citation_support_score

All functions are independent so score.py can save each field separately.
"""
import os
from typing import List, Dict, Tuple


# ---------------------------------------------------------------------------
# Path normalization
# ---------------------------------------------------------------------------

def normalize_path(path: str) -> str:
    """Normalize a file path for consistent comparison."""
    if not path:
        return ""
    p = path.replace("\\", "/").strip("/")
    # Strip known repo prefixes
    if "repos/fastapi/" in p:
        return p[p.find("repos/fastapi/") + len("repos/fastapi/"):]
    if "fastapi/" in p:
        return p[p.find("fastapi/"):]
    if "aws-lambda-developer-guide/" in p:
        return p[p.find("aws-lambda-developer-guide/") + len("aws-lambda-developer-guide/"):]
    return p


def _files_match(pred_path: str, gt_path: str) -> bool:
    """Check if two file paths refer to the same file."""
    p = normalize_path(pred_path)
    g = normalize_path(gt_path)
    if p == g:
        return True
    # Basename match: only if one is suffix of the other (prevents routing.py == security/oauth2.py)
    p_base = os.path.basename(p)
    g_base = os.path.basename(g)
    if p_base == g_base and (p.endswith(g) or g.endswith(p)):
        return True
    return False


def _matched_pairs(predicted: List[Dict], ground_truth: List[Dict]) -> List[Tuple[Dict, Dict]]:
    """Return (pred, gt) pairs where files match. Each GT matched at most once (best pred)."""
    pairs = []
    used_pred = set()
    for gt in ground_truth:
        best = None
        for i, pred in enumerate(predicted):
            if i in used_pred:
                continue
            if _files_match(pred.get("file", ""), gt.get("file", "")):
                best = (i, pred)
                break  # first match wins; could be improved with best-IoU selection
        if best is not None:
            used_pred.add(best[0])
            pairs.append((best[1], gt))
    return pairs


# ---------------------------------------------------------------------------
# Layer 1 — File-level precision / recall / F1
# ---------------------------------------------------------------------------

def citation_file_precision(predicted: List[Dict], ground_truth: List[Dict]) -> float:
    """What fraction of predicted files hit a GT file?"""
    if not predicted:
        return 1.0 if not ground_truth else 0.0
    hits = sum(
        1 for pred in predicted
        if any(_files_match(pred.get("file", ""), gt.get("file", "")) for gt in ground_truth)
    )
    return hits / len(predicted)


def citation_file_recall(predicted: List[Dict], ground_truth: List[Dict]) -> float:
    """What fraction of GT files were found in predictions?"""
    if not ground_truth:
        return 1.0 if not predicted else 0.0
    hits = sum(
        1 for gt in ground_truth
        if any(_files_match(pred.get("file", ""), gt.get("file", "")) for pred in predicted)
    )
    return hits / len(ground_truth)


def citation_file_f1(predicted: List[Dict], ground_truth: List[Dict]) -> float:
    """Harmonic mean of file-level precision and recall."""
    p = citation_file_precision(predicted, ground_truth)
    r = citation_file_recall(predicted, ground_truth)
    if p + r == 0:
        return 0.0
    return 2 * p * r / (p + r)


# ---------------------------------------------------------------------------
# Layer 2 — Evidence localization
# ---------------------------------------------------------------------------

def _line_iou(pred: Dict, gt: Dict) -> float:
    """IoU of line ranges for a matched file pair."""
    try:
        p_start = int(pred.get("line_start", 0))
        p_end = int(pred.get("line_end", 0))
        g_start = int(gt.get("line_start", 0))
        g_end = int(gt.get("line_end", 0))
    except (ValueError, TypeError):
        return 0.0
    intersection = max(0, min(p_end, g_end) - max(p_start, g_start) + 1)
    union = max(p_end, g_end) - min(p_start, g_start) + 1
    if union <= 0:
        return 0.0
    return intersection / union


def _line_bounds(span: Dict) -> Tuple[int, int] | None:
    try:
        start = int(span.get("line_start", 0))
        end = int(span.get("line_end", 0))
    except (ValueError, TypeError):
        return None
    if start <= 0 or end < start:
        return None
    return start, end


def _overlap_len(a: Dict, b: Dict) -> int:
    a_bounds = _line_bounds(a)
    b_bounds = _line_bounds(b)
    if not a_bounds or not b_bounds:
        return 0
    a_start, a_end = a_bounds
    b_start, b_end = b_bounds
    return max(0, min(a_end, b_end) - max(a_start, b_start) + 1)


def _span_len(span: Dict) -> int:
    bounds = _line_bounds(span)
    if not bounds:
        return 0
    start, end = bounds
    return end - start + 1


def _merged_interval_len(intervals: List[Tuple[int, int]]) -> int:
    if not intervals:
        return 0
    intervals = sorted(intervals)
    merged = [intervals[0]]
    for start, end in intervals[1:]:
        last_start, last_end = merged[-1]
        if start <= last_end + 1:
            merged[-1] = (last_start, max(last_end, end))
        else:
            merged.append((start, end))
    return sum(end - start + 1 for start, end in merged)


def citation_containment_recall(predicted: List[Dict], ground_truth: List[Dict]) -> float:
    """
    How much of each gold span is contained inside predicted spans on the same file.
    This rewards chunk-level retrieval that fully contains the gold evidence.
    """
    if not ground_truth:
        return 1.0 if not predicted else 0.0
    total = 0.0
    for gt in ground_truth:
        gt_bounds = _line_bounds(gt)
        if not gt_bounds:
            continue
        g_start, g_end = gt_bounds
        overlaps = []
        for pred in predicted:
            if not _files_match(pred.get("file", ""), gt.get("file", "")):
                continue
            p_bounds = _line_bounds(pred)
            if not p_bounds:
                continue
            p_start, p_end = p_bounds
            start = max(g_start, p_start)
            end = min(g_end, p_end)
            if start <= end:
                overlaps.append((start, end))
        total += _merged_interval_len(overlaps) / max(1, g_end - g_start + 1)
    return total / len(ground_truth)


def citation_span_precision(predicted: List[Dict], ground_truth: List[Dict]) -> float:
    """
    Fraction of predicted citation lines that overlap a gold span on the same file.
    This penalizes overly broad chunks without treating containment as failure.
    """
    if not predicted:
        return 1.0 if not ground_truth else 0.0
    total_pred_lines = 0
    total_overlap = 0
    for pred in predicted:
        pred_bounds = _line_bounds(pred)
        if not pred_bounds:
            continue
        p_start, p_end = pred_bounds
        total_pred_lines += p_end - p_start + 1
        overlaps = []
        for gt in ground_truth:
            if not _files_match(pred.get("file", ""), gt.get("file", "")):
                continue
            gt_bounds = _line_bounds(gt)
            if not gt_bounds:
                continue
            g_start, g_end = gt_bounds
            start = max(p_start, g_start)
            end = min(p_end, g_end)
            if start <= end:
                overlaps.append((start, end))
        total_overlap += _merged_interval_len(overlaps)
    if total_pred_lines <= 0:
        return 0.0
    return total_overlap / total_pred_lines


def citation_span_fbeta(predicted: List[Dict], ground_truth: List[Dict], beta: float = 2.0) -> float:
    recall = citation_containment_recall(predicted, ground_truth)
    precision = citation_span_precision(predicted, ground_truth)
    if recall + precision == 0:
        return 0.0
    beta_sq = beta * beta
    return (1 + beta_sq) * precision * recall / ((beta_sq * precision) + recall)


def citation_weighted_span_score(predicted: List[Dict], ground_truth: List[Dict],
                                 recall_weight: float = 0.7) -> float:
    recall = citation_containment_recall(predicted, ground_truth)
    precision = citation_span_precision(predicted, ground_truth)
    return recall_weight * recall + (1.0 - recall_weight) * precision


def line_iou_per_file(predicted: List[Dict], ground_truth: List[Dict]) -> float:
    """Average line IoU across all matched (pred, GT) file pairs."""
    pairs = _matched_pairs(predicted, ground_truth)
    if not pairs:
        return 0.0
    return sum(_line_iou(p, g) for p, g in pairs) / len(ground_truth)


def citation_support_score(predicted: List[Dict], ground_truth: List[Dict]) -> float:
    """
    Combined quality: file_match × line_iou, averaged over GT citations.
    0 if file not matched; partial credit for partial line overlap.
    """
    if not ground_truth:
        return 1.0 if not predicted else 0.0
    total = 0.0
    for gt in ground_truth:
        best_iou = 0.0
        for pred in predicted:
            if _files_match(pred.get("file", ""), gt.get("file", "")):
                iou = _line_iou(pred, gt)
                best_iou = max(best_iou, iou)
        total += best_iou  # 0 if no file match, partial if matched but line range differs
    return total / len(ground_truth)


# ---------------------------------------------------------------------------
# Legacy compat (used in rescore.py)
# ---------------------------------------------------------------------------

def calculate_citation_overlap(predicted: List[Dict], ground_truth: List[Dict]) -> float:
    """Legacy: returns citation_support_score (file_match × line_iou)."""
    return citation_support_score(predicted, ground_truth)


# ---------------------------------------------------------------------------
# CLI: verify ground truth evidence against disk
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import json
    import argparse

    parser = argparse.ArgumentParser(description="Verify GT evidence against local files.")
    parser.add_argument("--qa-file", type=str, default="qa_dataset/seed_v0.json")
    parser.add_argument("--repo-root", type=str, default="dataset")
    args = parser.parse_args()

    with open(args.qa_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    for item in data[:5]:
        print(f"\n[{item['id']}] {item['question'][:80]}...")
        for ev in item["evidence"]:
            rel = ev["file"]
            fastapi_root_files = {"pyproject.toml", "README.md", "CONTRIBUTING.md"}
            is_fastapi = (
                rel.startswith("fastapi/")
                or rel.startswith("docs_src/")
                or rel.startswith("docs/")
                or rel in fastapi_root_files
            )
            disk = (
                os.path.join(args.repo_root, "repos", "fastapi", rel)
                if is_fastapi
                else os.path.join(args.repo_root, "docs", "aws-lambda-developer-guide", rel)
            )
            if os.path.exists(disk):
                with open(disk, "r", encoding="utf-8") as fc:
                    lines = fc.readlines()
                snippet = lines[ev.get("line_start", 1) - 1: ev.get("line_end", 1)]
                print(f"  ✓ {rel} (L{ev['line_start']}-{ev['line_end']}): {snippet[0].strip()[:60]}")
            else:
                print(f"  ✗ NOT FOUND: {disk}")
