import os
from typing import Dict, List, Optional

from utils.verify_evidence import normalize_path


def dataset_of_path(file_path: str) -> str:
    path = normalize_path(file_path)
    fastapi_root_files = {"pyproject.toml", "README.md", "CONTRIBUTING.md"}
    if (
        path.startswith("fastapi/")
        or path.startswith("docs/")
        or path.startswith("docs_src/")
        or path in fastapi_root_files
    ):
        return "fastapi"
    return "lambda"


def files_match(lhs: str, rhs: str) -> bool:
    left = normalize_path(lhs)
    right = normalize_path(rhs)
    if left == right:
        return True
    left_base = os.path.basename(left)
    right_base = os.path.basename(right)
    return left_base == right_base and (left.endswith(right) or right.endswith(left))


def line_iou(pred: Dict, gt: Dict) -> float:
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


def _required_datasets(ground_truth: List[Dict]) -> List[str]:
    required = []
    for ev in ground_truth:
        ds = dataset_of_path(ev.get("file", ""))
        if ds not in required:
            required.append(ds)
    return required


def file_hit_at_k(candidates: List[Dict], ground_truth: List[Dict], k: int = 5) -> float:
    top = candidates[:k]
    return 1.0 if any(
        files_match(c["file"], gt.get("file", ""))
        for c in top
        for gt in ground_truth
    ) else 0.0


def all_gold_files_hit_at_k(candidates: List[Dict], ground_truth: List[Dict], k: int = 5) -> float:
    top = candidates[:k]
    gold_files = []
    for gt in ground_truth:
        gt_file = gt.get("file", "")
        if gt_file and not any(files_match(gt_file, existing) for existing in gold_files):
            gold_files.append(gt_file)
    if not gold_files:
        return 1.0
    covered = 0
    for gt_file in gold_files:
        if any(files_match(c["file"], gt_file) for c in top):
            covered += 1
    return 1.0 if covered == len(gold_files) else 0.0


def max_line_iou_at_k(candidates: List[Dict], ground_truth: List[Dict], k: int = 5) -> float:
    top = candidates[:k]
    best = 0.0
    for gt in ground_truth:
        for cand in top:
            if files_match(cand["file"], gt.get("file", "")):
                best = max(best, line_iou(cand, gt))
    return best


def dataset_coverage_at_k(candidates: List[Dict], ground_truth: List[Dict], k: int = 5) -> Optional[float]:
    required = _required_datasets(ground_truth)
    if len(required) <= 1:
        return None
    observed = {dataset_of_path(c["file"]) for c in candidates[:k]}
    return 1.0 if all(ds in observed for ds in required) else 0.0


def summarize_candidates(results: List[Dict]) -> List[Dict]:
    summary = []
    for rank, item in enumerate(results, start=1):
        meta = item.get("metadata", {})
        summary.append({
            "rank": rank,
            "id": item.get("id"),
            "score": item.get("score"),
            "file": meta.get("file"),
            "line_start": meta.get("line_start"),
            "line_end": meta.get("line_end"),
            "dataset": dataset_of_path(meta.get("file", "")),
        })
    return summary


def summarize_spans(spans: List[Dict]) -> List[Dict]:
    return [
        {
            "rank": i + 1,
            "file": item.get("file"),
            "line_start": item.get("line_start"),
            "line_end": item.get("line_end"),
            "dataset": dataset_of_path(item.get("file", "")),
        }
        for i, item in enumerate(spans)
    ]


def compute_basic_retrieval_metrics(candidates: List[Dict], ground_truth: List[Dict], k: int = 5) -> Dict[str, Optional[float]]:
    return {
        f"file_hit@{k}": file_hit_at_k(candidates, ground_truth, k),
        f"all_gold_files_hit@{k}": all_gold_files_hit_at_k(candidates, ground_truth, k),
        f"max_line_iou@{k}": max_line_iou_at_k(candidates, ground_truth, k),
        f"dataset_coverage@{k}": dataset_coverage_at_k(candidates, ground_truth, k),
    }
