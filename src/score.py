"""
score.py — Standalone Scoring Engine

Reads raw predictions (JSONL from evaluate.py) + ground truth (seed_v1.json),
computes ALL metrics using a dual LLM-judge approach + geometric citation metrics,
and saves flattened scored results to results/scored/.

Usage:
  python src/score.py --predictions results/raw/ --ground-truth qa_dataset/seed_v1.json
"""
import os
import json
import glob
import argparse
import time
import hashlib
from typing import Dict, Any, List, Optional

from openai import OpenAI
from pydantic import BaseModel, Field

import sys
sys.path.insert(0, os.path.dirname(__file__))

from config import JUDGE_BASE_URL, JUDGE_API_KEY, JUDGE_MODEL_NAME
from utils.verify_evidence import (
    citation_file_precision,
    citation_file_recall,
    citation_file_f1,
    line_iou_per_file as evidence_line_iou,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
JUDGE_PROMPT_VERSION = "v1.1"
SCORING_VERSION = "v1.0"

# ---------------------------------------------------------------------------
# LLM Judge Setup
# ---------------------------------------------------------------------------

# Combined judge output — both metrics in a single API call
class JudgeOutput(BaseModel):
    answer_score: float = Field(
        description="Float 0.0-1.0. Correctness of the predicted answer vs ground truth. "
                    "0.0=completely wrong, 0.5=partially correct, 1.0=fully correct."
    )
    answer_reason: str = Field(description="One-sentence reason for the answer score.")
    citation_support: float = Field(
        description="Float 0.0-1.0. How well the cited evidence supports the predicted answer. "
                    "0.0=no support, 0.5=partial support, 1.0=fully supports answer."
    )
    citation_reason: str = Field(description="One-sentence reason for the citation support score.")

# Lightweight output — only answer_score (use when citation judge is disabled)
class AnswerOnlyOutput(BaseModel):
    answer_score: float = Field(
        description="Float 0.0-1.0. Correctness of the predicted answer vs ground truth."
    )
    answer_reason: str = Field(description="One-sentence reason for the answer score.")

_judge_client = OpenAI(base_url=JUDGE_BASE_URL, api_key=JUDGE_API_KEY)

# ---------------------------------------------------------------------------
# Caching & Pacing
# ---------------------------------------------------------------------------

JUDGE_CACHE_FILE = "results/judge_cache.jsonl"
_judge_cache: Dict[str, Dict] = {}

def load_judge_cache():
    global _judge_cache
    if os.path.exists(JUDGE_CACHE_FILE):
        with open(JUDGE_CACHE_FILE, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line: continue
                try:
                    data = json.loads(line)
                    _judge_cache[data["cache_key"]] = data["result"]
                except json.JSONDecodeError:
                    pass

def append_to_cache(cache_key: str, result: Dict):
    global _judge_cache
    _judge_cache[cache_key] = result
    with open(JUDGE_CACHE_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps({"cache_key": cache_key, "result": result}) + "\n")

class RateLimiter:
    def __init__(self, rpm: int = 4):
        self.min_interval = 60.0 / rpm if rpm > 0 else 0.0
        self.last_call = 0.0

    def wait(self):
        if self.min_interval <= 0: return
        now = time.time()
        elapsed = now - self.last_call
        if elapsed < self.min_interval:
            sleep_time = self.min_interval - elapsed
            print(f"  [Pacing] Sleeping {sleep_time:.1f}s to respect RPM limit...")
            time.sleep(sleep_time)
        self.last_call = time.time()

_pacer = RateLimiter(rpm=4) # Will be configurable via args

def _call_llm_with_retry(messages: List[Dict], response_format, max_retries: int = 3):
    """Call OpenAI SDK with exponential backoff for 429 Rate Limits."""
    base_wait = 15  # seconds
    for attempt in range(max_retries):
        try:
            return _judge_client.beta.chat.completions.parse(
                model=JUDGE_MODEL_NAME,
                messages=messages,
                response_format=response_format,
                temperature=0.0
            )
        except Exception as e:
            err_str = str(e).lower()
            is_rate_limit = any(term in err_str for term in ["429", "quota", "rate limit", "resource_exhausted", "too many requests"])
            if is_rate_limit:
                if attempt < max_retries - 1:
                    wait_time = base_wait * (2 ** attempt)
                    print(f"  [Rate Limit 429] Waiting {wait_time}s before retry ({attempt+1}/{max_retries})...")
                    time.sleep(wait_time)
                else:
                    print("  [Rate Limit] Max retries reached.")
                    raise e
            else:
                raise e


def _get_evidence_text(evidence_list: List[Dict], repo_root: str = "dataset") -> str:
    """Read the actual text snippets for the predicted citations from disk."""
    snippets = []
    for ev in evidence_list:
        rel = ev.get("file", "")
        if not rel:
            continue
        try:
            line_start = int(ev.get("line_start", 1))
            line_end   = int(ev.get("line_end",   1))
        except (ValueError, TypeError):
            continue
        disk = (
            os.path.join(repo_root, "repos", rel)
            if rel.startswith("fastapi/")
            else os.path.join(repo_root, "docs", "aws-lambda-developer-guide", rel)
        )
        if os.path.exists(disk):
            try:
                with open(disk, "r", encoding="utf-8") as fc:
                    lines = fc.readlines()
                snippet_lines = lines[max(0, line_start - 1): line_end]
                snippets.append(f"--- {rel} (L{line_start}-{line_end}) ---\n" + "".join(snippet_lines))
            except Exception:
                pass
    return "\n".join(snippets)


def llm_judge(question: str,
              predicted: str,
              gt_answer: str,
              reasoning_type: str,
              grading_notes: str,
              pred_evidence: List[Dict],
              with_citation: bool = False) -> tuple[Dict[str, Any], str]:
    """
    Single LLM call. Returns (output_dict, status_string).
    """
    if not predicted or not predicted.strip():
        return {"answer_score": 0.0, "answer_reason": "Empty prediction.",
                "citation_support": 0.0 if with_citation else None, "citation_reason": "No answer."}, "ok"

    if with_citation:
        evidence_text = _get_evidence_text(pred_evidence)
        citation_block = f"""

Cited Evidence (text from predicted files):
{evidence_text if evidence_text.strip() else '(No readable evidence found on disk)'}

For citation_support: score 0.0 if no evidence matches, 0.5 if partially relevant, 1.0 if evidence directly proves the answer."""
        response_format = JudgeOutput
    else:
        evidence_block = ""
        citation_block = ""
        response_format = AnswerOnlyOutput

    prompt = f"""You are an expert grading judge evaluating a RAG system's output.

Question: {question}
Grounding Task: {reasoning_type}
Grading Notes: {grading_notes}

Ground Truth Answer:
{gt_answer}

Predicted Answer:
{predicted}{citation_block}

Score answer_score: 0.0=completely wrong, 0.5=partially correct, 1.0=fully correct.
Focus on factual correctness. Ignore verbosity/formatting differences."""

    try:
        resp = _call_llm_with_retry(
            messages=[
                {"role": "system", "content": "You are a grading judge. Output JSON only."},
                {"role": "user", "content": prompt}
            ],
            response_format=response_format
        )
        parsed = resp.choices[0].message.parsed
        ans = max(0.0, min(1.0, float(parsed.answer_score)))
        if with_citation:
            cit = max(0.0, min(1.0, float(parsed.citation_support)))
            return {
                "answer_score": ans,
                "answer_reason": parsed.answer_reason,
                "citation_support": cit,
                "citation_reason": parsed.citation_reason,
            }, "ok"
        else:
            return {
                "answer_score": ans,
                "answer_reason": getattr(parsed, "answer_reason", ""),
                "citation_support": None,
                "citation_reason": "",
            }, "ok"
    except Exception as e:
        err_str = str(e).lower()
        if any(term in err_str for term in ["429", "quota", "rate limit", "resource_exhausted"]):
            status = "rate_limit_error"
        else:
            status = "api_error"
        print(f"  [Judge error] {status}: {e}")
        return {"answer_score": None, "answer_reason": str(e),
                "citation_support": None, "citation_reason": ""}, status


# ---------------------------------------------------------------------------
# Failure-aware scoring
# ---------------------------------------------------------------------------

def score_record(record: Dict, gt_item: Dict,
                 run_judge: bool = True, with_citation_judge: bool = False) -> Dict:
    """
    Score a single prediction record against its ground truth.
    - run_judge=False : skip all LLM calls (fast/offline mode)
    - with_citation_judge=True : include citation_support in the single LLM call (doubles cost)
    Returns a flattened dictionary.
    """
    error_type = record.get("error_type")

    pred_answer = record.get("predicted_answer") or record.get("answer", "")
    pred_evidence = record.get("predicted_evidence") or record.get("evidence", [])

    gt_answer = gt_item["answer"]
    gt_evidence = gt_item.get("evidence", [])
    reasoning_type = gt_item.get("reasoning_type", "unknown")
    grading_notes = gt_item.get("grading_notes", "")

    # Hard-fail states — skip LLM entirely
    if error_type in ("timeout", "json_parse_error", "tool_error_fatal"):
        judge_out = {"answer_score": 0.0, "answer_reason": f"hard-fail:{error_type}",
                     "citation_support": 0.0 if with_citation_judge else None, "citation_reason": ""}
        judge_status = "skipped_hard_fail"
        cit_fp = cit_fr = cit_ff1 = liou = 0.0
    else:
        # Deterministic geometric metrics (always computed for valid parses)
        cit_fp  = citation_file_precision(pred_evidence, gt_evidence)
        cit_fr  = citation_file_recall(pred_evidence, gt_evidence)
        cit_ff1 = citation_file_f1(pred_evidence, gt_evidence)
        liou    = evidence_line_iou(pred_evidence, gt_evidence)

        if not run_judge:
            judge_out = {"answer_score": None, "answer_reason": "judge_skipped",
                         "citation_support": None, "citation_reason": ""}
            judge_status = "skipped"
        else:
            cache_keys = {
                "question": record["question"],
                "gt_answer": gt_answer,
                "predicted_answer": pred_answer,
                "judge_model": JUDGE_MODEL_NAME,
                "judge_mode": "answer+citation" if with_citation_judge else "answer_only",
                "judge_prompt_version": JUDGE_PROMPT_VERSION,
                "scoring_version": SCORING_VERSION,
                "reasoning_type": reasoning_type,
                "grading_notes": grading_notes
            }
            if with_citation_judge:
                cache_keys["predicted_evidence"] = pred_evidence

            cache_hash = hashlib.sha256(json.dumps(cache_keys, sort_keys=True).encode("utf-8")).hexdigest()

            if cache_hash in _judge_cache:
                judge_out = _judge_cache[cache_hash]
                judge_status = "cache_hit"
            else:
                _pacer.wait()
                judge_out, judge_status = llm_judge(
                    record["question"], pred_answer, gt_answer,
                    reasoning_type, grading_notes, pred_evidence,
                    with_citation=with_citation_judge
                )
                if judge_status == "ok":
                    append_to_cache(cache_hash, judge_out)

    return {
        "id": record["id"],
        "method": record.get("method", "unknown"),
        "question": record["question"],
        "predicted_answer": pred_answer,
        "predicted_evidence": pred_evidence,
        "latency_sec": record.get("latency_sec", 0.0),
        "error_type": error_type,
        "efficiency_flag": record.get("efficiency_flag"),
        
        "judge_status": judge_status,
        "answer_score": judge_out["answer_score"],
        "answer_reason": judge_out.get("answer_reason", ""),
        "citation_support_llm": judge_out["citation_support"],
        "citation_reason": judge_out.get("citation_reason", ""),
        "citation_file_precision": cit_fp,
        "citation_file_recall": cit_fr,
        "citation_file_f1": cit_ff1,
        "evidence_line_iou": liou
    }


# ---------------------------------------------------------------------------
# Main scoring pipeline
# ---------------------------------------------------------------------------

def load_ground_truth(qa_file: str) -> Dict[str, Dict]:
    with open(qa_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    return {item["id"]: item for item in data}


def score_predictions_file(predictions_file: str, gt_map: Dict[str, Dict],
                            output_dir: str, run_judge: bool = True,
                            with_citation_judge: bool = False,
                            subset_ids: Optional[set] = None) -> str:
    """Score one predictions JSONL file, save scored JSONL."""
    basename = os.path.basename(predictions_file).replace("predictions_", "scored_")
    out_path = os.path.join(output_dir, basename)

    scored_records = []
    with open(predictions_file, "r", encoding="utf-8") as f:
        lines = [json.loads(l) for l in f if l.strip()]

    target_records = [rec for rec in lines if not subset_ids or rec["id"] in subset_ids]
    if not target_records:
        print(f"No records matching subset filter in {predictions_file}")
        return ""

    mode = "answer+citation" if with_citation_judge else "answer-only"
    print(f"\nScoring {len(target_records)} predictions from {os.path.basename(predictions_file)} [{mode}]...")

    for i, record in enumerate(target_records):
        qid = record["id"]
        if qid not in gt_map:
            print(f"  Warning: ID {qid} not in ground truth, skipping.")
            continue
        gt_item = gt_map[qid]
        scored = score_record(record, gt_item, run_judge=run_judge,
                              with_citation_judge=with_citation_judge)
        scored_records.append(scored)

        if (i + 1) % 5 == 0:
            print(f"  {i+1}/{len(target_records)} scored...")

    # Write as JSONL
    with open(out_path, "w", encoding="utf-8") as f:
        for rec in scored_records:
            f.write(json.dumps(rec) + "\n")

    print(f"Scored results saved to {out_path}")
    _print_summary(scored_records)
    return out_path


def _print_summary(records: List[Dict]):
    """Print aggregate stats to console."""
    import statistics
    n = len(records)
    if n == 0:
        return

    def mean(vals): return sum(vals) / len(vals) if vals else 0.0
    def safe_median(vals): return statistics.median(vals) if vals else 0.0
    def pct(vals, p): return sorted(vals)[int(len(vals) * p / 100)] if vals else 0.0

    method = records[0].get("method", "?")
    
    # Filter out None scores (from API errors or skips)
    ans_scores = [r["answer_score"] for r in records if r.get("answer_score") is not None]
    cit_supp_scores = [r["citation_support_llm"] for r in records if r.get("citation_support_llm") is not None]
    cit_f1 = [r["citation_file_f1"] for r in records]
    liou = [r["evidence_line_iou"] for r in records]
    
    # Analyze status
    statuses = [r["judge_status"] for r in records]
    n_ok = statuses.count("ok")
    n_hit = statuses.count("cache_hit")
    n_err = statuses.count("api_error") + statuses.count("rate_limit_error")

    print(f"\n{'='*55}")
    print(f"  Method {method} — {n} samples scored")
    print(f"  Judge Status API: {n_ok} ok, Cache: {n_hit} hits, Errors: {n_err}")
    print(f"{'='*55}")
    if ans_scores:
        print(f"  answer_score            : {mean(ans_scores):.3f}  (n={len(ans_scores)})")
    else:
        print(f"  answer_score            : N/A")
    if cit_supp_scores:
        print(f"  citation_support_llm    : {mean(cit_supp_scores):.3f}  (n={len(cit_supp_scores)})")
    else:
        print(f"  citation_support_llm    : skipped (use --with-citation-judge)")
    print(f"  citation_file_f1        : {mean(cit_f1):.3f}")
    print(f"  evidence_line_iou       : {mean(liou):.3f}")
    print(f"{'='*55}")


def main():
    parser = argparse.ArgumentParser(description="Score raw predictions from evaluate.py")
    parser.add_argument("--predictions", type=str, default="results/raw",
                        help="Path to JSONL predictions file or directory of JSONL files")
    parser.add_argument("--ground-truth", type=str, default="qa_dataset/seed_v1.json")
    parser.add_argument("--output", type=str, default="results/scored")
    parser.add_argument("--no-judge", action="store_true",
                        help="Skip LLM-as-judge entirely (fast/offline mode, all LLM scores = 0.0)")
    parser.add_argument("--with-citation-judge", action="store_true",
                        help="Include citation_support in the single LLM call (costs 0 extra API calls, "
                             "adds citation_support_llm field to output). Off by default to save quota.")
    parser.add_argument("--rpm-limit", type=int, default=4,
                        help="Rate limit for the judge API in requests per minute. Implements proactive pacing.")
    parser.add_argument("--subset", type=str, default=None,
                        help="Path to a text file containing issue IDs to evaluate (e.g., qa_dataset/dev_ids.txt)")
    args = parser.parse_args()

    if os.path.basename(os.getcwd()) == "src":
        os.chdir("..")

    os.makedirs(args.output, exist_ok=True)
    gt_map = load_ground_truth(args.ground_truth)
    load_judge_cache()
    
    global _pacer
    _pacer = RateLimiter(rpm=args.rpm_limit)

    subset_ids = None
    if args.subset:
        with open(args.subset, "r") as f:
            subset_ids = {line.strip() for line in f if line.strip()}
        print(f"Loaded {len(subset_ids)} target IDs from subset filter.")

    if os.path.isdir(args.predictions):
        files = glob.glob(os.path.join(args.predictions, "predictions_*.jsonl"))
        if not files:
            print(f"No predictions_*.jsonl found in {args.predictions}")
            return
    else:
        files = [args.predictions]

    for f in sorted(files):
        score_predictions_file(f, gt_map, args.output,
                               run_judge=not args.no_judge,
                               with_citation_judge=args.with_citation_judge,
                               subset_ids=subset_ids)

    print("\nAll scoring complete.")


if __name__ == "__main__":
    main()
