"""
score.py - Standalone Scoring Engine

Reads raw predictions (JSONL from evaluate.py) + ground truth (seed_v0.json by default),
computes answer and citation metrics, and saves scored JSONL to results/scored/.
"""
import os
import json
import glob
import argparse
import time
import hashlib
import re
from typing import Dict, Any, List, Optional

from openai import OpenAI
from pydantic import BaseModel, Field

import sys
sys.path.insert(0, os.path.dirname(__file__))

from config import (
    API_KEY,
    JUDGE_BASE_URL,
    JUDGE_API_KEY,
    JUDGE_MODEL_NAME,
    MODEL_NAME,
    OLLAMA_BASE_URL,
)
from pipeline_schema import (
    ERROR_MALFORMED_JSON,
    ERROR_TIMEOUT,
    ERROR_TOOL,
    ensure_required_fields,
    normalize_error_type,
    normalize_reasoning_type,
    quantize_judge_score,
)
from utils.verify_evidence import (
    citation_file_precision,
    citation_file_recall,
    citation_file_f1,
    line_iou_per_file as evidence_line_iou,
    citation_support_score,
    citation_containment_recall,
    citation_span_precision,
    citation_span_fbeta,
    citation_weighted_span_score,
)

JUDGE_PROMPT_VERSION = "v1.2"
SCORING_VERSION = "v2.2"
NARROWING_WINDOW = 2
BERTSCORE_DEFAULT_MODEL = os.getenv("BERTSCORE_MODEL", "roberta-large")
BERTSCORE_DEFAULT_LANG = os.getenv("BERTSCORE_LANG", "en")
BERTSCORE_DEFAULT_DEVICE = os.getenv("BERTSCORE_DEVICE", None)
_bertscore_scorer = None
_bertscore_scorer_key = None


class JudgeOutput(BaseModel):
    answer_score: float = Field(description="0.0, 0.5, or 1.0 answer correctness score.")
    answer_reason: str = Field(description="One-sentence reason for the answer score.")
    citation_support: float = Field(description="0.0, 0.5, or 1.0 evidence support score.")
    citation_reason: str = Field(description="One-sentence reason for the citation support score.")


class AnswerOnlyOutput(BaseModel):
    answer_score: float = Field(description="0.0, 0.5, or 1.0 answer correctness score.")
    answer_reason: str = Field(description="One-sentence reason for the answer score.")


_primary_judge_client = OpenAI(base_url=JUDGE_BASE_URL, api_key=JUDGE_API_KEY)
_fallback_judge_client = OpenAI(base_url=OLLAMA_BASE_URL, api_key=API_KEY)
FALLBACK_JUDGE_MODEL_NAME = MODEL_NAME

JUDGE_CACHE_FILE = os.getenv("JUDGE_CACHE_FILE", "runs/judge_cache.jsonl")
_judge_cache: Dict[str, Dict] = {}


def load_judge_cache():
    global _judge_cache
    if os.path.exists(JUDGE_CACHE_FILE):
        with open(JUDGE_CACHE_FILE, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    _judge_cache[data["cache_key"]] = data["result"]
                except json.JSONDecodeError:
                    pass


def append_to_cache(cache_key: str, result: Dict):
    global _judge_cache
    _judge_cache[cache_key] = result
    cache_dir = os.path.dirname(JUDGE_CACHE_FILE)
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
    with open(JUDGE_CACHE_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps({"cache_key": cache_key, "result": result}) + "\n")


class RateLimiter:
    def __init__(self, rpm: int = 4):
        self.min_interval = 60.0 / rpm if rpm > 0 else 0.0
        self.last_call = 0.0

    def wait(self):
        if self.min_interval <= 0:
            return
        now = time.time()
        elapsed = now - self.last_call
        if elapsed < self.min_interval:
            sleep_time = self.min_interval - elapsed
            print(f"  [Pacing] Sleeping {sleep_time:.1f}s to respect RPM limit...")
            time.sleep(sleep_time)
        self.last_call = time.time()


_pacer = RateLimiter(rpm=4)
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _is_transient_judge_error(exc: Exception) -> bool:
    err_str = str(exc).lower()
    transient_terms = [
        "429",
        "quota",
        "rate limit",
        "resource_exhausted",
        "too many requests",
        "503",
        "502",
        "500",
        "temporarily unavailable",
        "service unavailable",
        "gateway",
        "timed out",
        "timeout",
        "connection reset",
        "connection error",
        "connection aborted",
    ]
    non_transient_terms = [
        "validation error",
        "json",
        "schema",
        "parse",
        "malformed",
        "missing required",
    ]
    return any(term in err_str for term in transient_terms) and not any(
        term in err_str for term in non_transient_terms
    )


def _call_llm_with_retry(client: OpenAI, model_name: str, messages: List[Dict], response_format, max_retries: int = 3):
    base_wait = 15
    for attempt in range(max_retries):
        try:
            return client.beta.chat.completions.parse(
                model=model_name,
                messages=messages,
                response_format=response_format,
                temperature=0.0
            )
        except Exception as e:
            is_rate_limit = _is_transient_judge_error(e)
            if is_rate_limit and attempt < max_retries - 1:
                wait_time = base_wait * (2 ** attempt)
                print(f"  [Rate Limit 429] Waiting {wait_time}s before retry ({attempt + 1}/{max_retries})...")
                time.sleep(wait_time)
                continue
            raise e


def compute_bertscore(
    predicted: str,
    reference: str,
    model_type: str = BERTSCORE_DEFAULT_MODEL,
    lang: str = BERTSCORE_DEFAULT_LANG,
    device: Optional[str] = BERTSCORE_DEFAULT_DEVICE,
    rescale_with_baseline: bool = False,
) -> Dict[str, Any]:
    """Compute BERTScore as a non-LLM semantic similarity check.

    This is intentionally separate from answer_score. BERTScore can reward
    lexical/semantic similarity while missing implementation-level correctness,
    so disagreement with the LLM judge is a signal for manual review.
    """
    if not predicted or not predicted.strip():
        return {
            "bertscore_precision": 0.0,
            "bertscore_recall": 0.0,
            "bertscore_f1": 0.0,
            "bertscore_model": model_type,
            "bertscore_status": "empty_prediction",
            "bertscore_error": "",
        }
    try:
        global _bertscore_scorer, _bertscore_scorer_key
        from bert_score import BERTScorer

        key = (model_type, lang, device, bool(rescale_with_baseline))
        if _bertscore_scorer is None or _bertscore_scorer_key != key:
            kwargs = {
                "model_type": model_type,
                "lang": lang,
                "rescale_with_baseline": rescale_with_baseline,
            }
            if device:
                kwargs["device"] = device
            _bertscore_scorer = BERTScorer(**kwargs)
            _bertscore_scorer_key = key

        precision, recall, f1 = _bertscore_scorer.score([predicted], [reference])
        return {
            "bertscore_precision": round(float(precision[0]), 6),
            "bertscore_recall": round(float(recall[0]), 6),
            "bertscore_f1": round(float(f1[0]), 6),
            "bertscore_model": model_type,
            "bertscore_status": "ok",
            "bertscore_error": "",
        }
    except Exception as exc:
        return {
            "bertscore_precision": None,
            "bertscore_recall": None,
            "bertscore_f1": None,
            "bertscore_model": model_type,
            "bertscore_status": "error",
            "bertscore_error": str(exc),
        }


def _read_evidence_from_disk(rel: str, line_start: int, line_end: int, repo_root: str) -> str:
    dataset_root = repo_root if os.path.isabs(repo_root) else os.path.join(_REPO_ROOT, repo_root)
    candidates = []
    if rel.startswith("fastapi/") or rel.startswith("docs_src/") or rel.startswith("docs/"):
        candidates.extend([
            os.path.join(dataset_root, "repos", "fastapi", rel),
            os.path.join(dataset_root, "repos", rel),
        ])
    candidates.extend([
        os.path.join(dataset_root, "docs", "aws-lambda-developer-guide", rel),
        os.path.join(_REPO_ROOT, rel),
    ])
    for disk in candidates:
        if not os.path.exists(disk):
            continue
        try:
            with open(disk, "r", encoding="utf-8") as fc:
                lines = fc.readlines()
            return "".join(lines[max(0, line_start - 1): line_end])
        except Exception:
            continue
    return ""


def _read_evidence_from_index(rel: str, line_start: int, line_end: int) -> str:
    index_path = os.path.join(_REPO_ROOT, "docs_index.json")
    if not os.path.exists(index_path):
        return ""
    try:
        with open(index_path, "r", encoding="utf-8") as f:
            docs = json.load(f)
    except Exception:
        return ""
    snippets = []
    for doc in docs:
        if doc.get("file") != rel:
            continue
        d_start = int(doc.get("line_start", 1))
        d_end = int(doc.get("line_end", d_start))
        if line_end < d_start or line_start > d_end:
            continue
        content_lines = doc.get("content", "").splitlines(True)
        start_offset = max(0, line_start - d_start)
        end_offset = min(len(content_lines), line_end - d_start + 1)
        snippet = "".join(content_lines[start_offset:end_offset])
        if snippet:
            snippets.append(snippet)
    return "\n".join(snippets)


def _get_evidence_numbered_lines(ev: Dict, repo_root: str = "dataset") -> List[tuple[int, str]]:
    rel = ev.get("file", "")
    if not rel:
        return []
    try:
        line_start = int(ev.get("line_start", 1))
        line_end = int(ev.get("line_end", 1))
    except (ValueError, TypeError):
        return []
    text = _read_evidence_from_disk(rel, line_start, line_end, repo_root)
    if not text:
        text = _read_evidence_from_index(rel, line_start, line_end)
    if not text:
        return []
    return [(line_start + idx, line) for idx, line in enumerate(text.splitlines())]


def _get_evidence_text(evidence_list: List[Dict], repo_root: str = "dataset") -> str:
    snippets = []
    for ev in evidence_list:
        rel = ev.get("file", "")
        if not rel:
            continue
        try:
            line_start = int(ev.get("line_start", 1))
            line_end = int(ev.get("line_end", 1))
        except (ValueError, TypeError):
            continue
        snippet = _read_evidence_from_disk(rel, line_start, line_end, repo_root)
        if not snippet:
            snippet = _read_evidence_from_index(rel, line_start, line_end)
        if snippet:
            snippets.append(f"--- {rel} (L{line_start}-{line_end}) ---\n" + snippet)
    return "\n".join(snippets)


def _narrowing_terms(question: str, predicted_answer: str) -> List[str]:
    text = f"{question}\n{predicted_answer}"
    terms = []
    terms.extend(re.findall(r"`([^`]{2,80})`", text))
    terms.extend(re.findall(r"\b[A-Za-z_][A-Za-z0-9_]*\.[A-Za-z_][A-Za-z0-9_\.]*\b", text))
    terms.extend(re.findall(r"\b[A-Za-z_][A-Za-z0-9_]{2,}\(\)", text))
    terms.extend(re.findall(r"\b[A-Z][A-Za-z0-9_]{2,}\b", text))
    stop = {
        "the", "and", "for", "with", "from", "into", "that", "this", "what",
        "which", "does", "how", "fastapi", "lambda", "class", "function",
        "method", "methods", "file", "files", "public", "directly", "async",
        "metadata", "answer", "question", "evidence", "returns", "return",
    }
    for token in re.findall(r"[A-Za-z_][A-Za-z0-9_]{2,}", text):
        if token.lower() not in stop:
            terms.append(token)
    out, seen = [], set()
    for term in terms:
        clean = term.strip().strip("`").rstrip("()")
        if not clean or clean.lower() in seen:
            continue
        seen.add(clean.lower())
        out.append(clean)
        if len(out) >= 32:
            break
    return out


def _line_relevance_score(line: str, terms: List[str]) -> int:
    score = 0
    for term in terms:
        if not term:
            continue
        pattern = rf"\b{re.escape(term)}\b" if re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", term) else re.escape(term)
        if re.search(pattern, line):
            score += 2 if any(ch in term for ch in "_.") or term[:1].isupper() else 1
    if re.search(r"\b(?:async\s+def|def|class)\s+[A-Za-z_][A-Za-z0-9_]*", line):
        score += 2
    if re.search(r"\bself\.[A-Za-z_][A-Za-z0-9_]*\s*=", line):
        score += 2
    return score


def narrow_evidence_spans(question: str, predicted_answer: str,
                          evidence_list: List[Dict], repo_root: str = "dataset") -> List[Dict]:
    """
    Heuristically narrow chunk-level citations to smaller subspans using only
    prediction-side information. This is applied uniformly to all methods.
    """
    terms = _narrowing_terms(question, predicted_answer)
    narrowed = []
    for ev in evidence_list:
        try:
            original_start = int(ev.get("line_start", 1))
            original_end = int(ev.get("line_end", original_start))
        except (ValueError, TypeError):
            narrowed.append(ev)
            continue
        numbered_lines = _get_evidence_numbered_lines(ev, repo_root)
        if not numbered_lines or original_end <= original_start:
            narrowed.append(ev)
            continue
        relevant = [
            line_no for line_no, line in numbered_lines
            if _line_relevance_score(line, terms) > 0
        ]
        if not relevant:
            narrowed.append(ev)
            continue
        new_start = max(original_start, min(relevant) - NARROWING_WINDOW)
        new_end = min(original_end, max(relevant) + NARROWING_WINDOW)
        if new_end < new_start:
            narrowed.append(ev)
            continue
        narrowed_ev = dict(ev)
        narrowed_ev["line_start"] = new_start
        narrowed_ev["line_end"] = new_end
        narrowed_ev["original_line_start"] = original_start
        narrowed_ev["original_line_end"] = original_end
        narrowed_ev["narrowing_applied"] = (new_start, new_end) != (original_start, original_end)
        narrowed.append(narrowed_ev)
    return narrowed


def llm_judge(question: str,
              predicted: str,
              gt_answer: str,
              reasoning_type: str,
              grading_notes: str,
              pred_evidence: List[Dict],
              with_citation: bool = False) -> tuple[Dict[str, Any], str]:
    if not predicted or not predicted.strip():
        return {
            "answer_score": 0.0,
            "answer_reason": "Empty prediction.",
            "citation_support": 0.0 if with_citation else None,
            "citation_reason": "No answer.",
            "judge_model_used": None,
            "judge_fallback_used": False,
        }, "ok"

    if with_citation:
        evidence_text = _get_evidence_text(pred_evidence)
        citation_block = f"""

Cited Evidence (text from predicted files):
{evidence_text if evidence_text.strip() else '(No readable evidence found on disk)'}

Scoring rubric for citation_support:
- 1.0 = The cited evidence directly supports the main claims in the answer.
- 0.5 = The evidence supports only part of the answer, or supports the right area/file but not the key claim directly.
- 0.0 = The evidence does not directly support the answer's main claims, or the answer goes materially beyond the evidence.
"""
        response_format = JudgeOutput
    else:
        citation_block = ""
        response_format = AnswerOnlyOutput

    prompt = f"""You are an expert grading judge evaluating a repository-QA system.

Question: {question}
Reasoning Type: {reasoning_type}
Grading Notes: {grading_notes}

Ground Truth Answer:
{gt_answer}

Predicted Answer:
{predicted}{citation_block}

Scoring rubric for answer_score:
- 1.0 = Correct and sufficiently complete. Must include the core mechanism required by the ground truth and grading notes. No major factual errors.
- 0.5 = Partially correct. Captures some relevant ideas but misses a required mechanism, required comparison side, required trace step, or exact implementation detail.
- 0.0 = Incorrect or unsupported. Misses the core mechanism, contradicts the ground truth, or answers at the wrong abstraction level.

Important:
- Do not award 1.0 for answers that are only generally plausible.
- If the question is implementation-focused, answers that only describe usage, API surface, or high-level behavior should not receive full credit.
- If the question is comparative, both sides must be substantively covered for full credit.
- If the question is trace-based, the answer must describe the actual flow across the relevant files/functions, not just summarize the general concept."""

    messages = [
        {"role": "system", "content": "You are a grading judge. Output JSON only."},
        {"role": "user", "content": prompt}
    ]

    def _run_single_judge(client: OpenAI, model_name: str, fallback_used: bool) -> Dict[str, Any]:
        resp = _call_llm_with_retry(
            client=client,
            model_name=model_name,
            messages=messages,
            response_format=response_format
        )
        parsed = resp.choices[0].message.parsed
        ans = quantize_judge_score(float(parsed.answer_score))
        if with_citation:
            cit = quantize_judge_score(float(parsed.citation_support))
            return {
                "answer_score": ans,
                "answer_reason": parsed.answer_reason,
                "citation_support": cit,
                "citation_reason": parsed.citation_reason,
                "judge_model_used": model_name,
                "judge_fallback_used": fallback_used,
            }
        return {
            "answer_score": ans,
            "answer_reason": getattr(parsed, "answer_reason", ""),
            "citation_support": None,
            "citation_reason": "",
            "judge_model_used": model_name,
            "judge_fallback_used": fallback_used,
        }

    try:
        return _run_single_judge(_primary_judge_client, JUDGE_MODEL_NAME, False), "ok_primary"
    except Exception as primary_error:
        if _is_transient_judge_error(primary_error):
            print(f"  [Judge fallback] primary judge unavailable ({primary_error}). Falling back to {FALLBACK_JUDGE_MODEL_NAME}.")
            try:
                return _run_single_judge(_fallback_judge_client, FALLBACK_JUDGE_MODEL_NAME, True), "ok_fallback"
            except Exception as fallback_error:
                status = "fallback_api_error"
                print(f"  [Judge error] {status}: {fallback_error}")
                return {
                    "answer_score": None,
                    "answer_reason": str(fallback_error),
                    "citation_support": None,
                    "citation_reason": "",
                    "judge_model_used": None,
                    "judge_fallback_used": True,
                }, status

        status = "api_error"
        print(f"  [Judge error] {status}: {primary_error}")
        return {
            "answer_score": None,
            "answer_reason": str(primary_error),
            "citation_support": None,
            "citation_reason": "",
            "judge_model_used": None,
            "judge_fallback_used": False,
        }, status


def score_record(record: Dict, gt_item: Dict,
                 run_judge: bool = True,
                 with_citation_judge: bool = False,
                 with_bertscore: bool = True,
                 bertscore_model: str = BERTSCORE_DEFAULT_MODEL,
                 bertscore_device: Optional[str] = BERTSCORE_DEFAULT_DEVICE,
                 bertscore_rescale: bool = False) -> Dict:
    ensure_required_fields(record, ["id", "question", "predicted_answer", "predicted_evidence"], "prediction record")

    error_type = normalize_error_type(record.get("error_type"))
    pred_answer = record.get("predicted_answer") or record.get("answer", "")
    pred_evidence = record.get("predicted_evidence") or record.get("evidence", [])
    scored_pred_evidence = narrow_evidence_spans(record.get("question", ""), pred_answer, pred_evidence)

    gt_answer = gt_item["answer"]
    gt_evidence = gt_item.get("evidence", [])
    reasoning_type = normalize_reasoning_type(gt_item)
    grading_notes = gt_item.get("grading_notes", "")

    if error_type in (ERROR_TIMEOUT, ERROR_MALFORMED_JSON, ERROR_TOOL):
        judge_out = {
            "answer_score": 0.0,
            "answer_reason": f"hard-fail:{error_type}",
            "citation_support": 0.0 if with_citation_judge else None,
            "citation_reason": "",
            "judge_model_used": None,
            "judge_fallback_used": False,
        }
        judge_status = "skipped_hard_fail"
        cit_fp = cit_fr = cit_ff1 = liou = cit_support_geom = 0.0
        containment_recall = span_precision = span_f2 = weighted_span = 0.0
    else:
        cit_fp = citation_file_precision(scored_pred_evidence, gt_evidence)
        cit_fr = citation_file_recall(scored_pred_evidence, gt_evidence)
        cit_ff1 = citation_file_f1(scored_pred_evidence, gt_evidence)
        liou = evidence_line_iou(scored_pred_evidence, gt_evidence)
        cit_support_geom = citation_support_score(scored_pred_evidence, gt_evidence)
        containment_recall = citation_containment_recall(scored_pred_evidence, gt_evidence)
        span_precision = citation_span_precision(scored_pred_evidence, gt_evidence)
        span_f2 = citation_span_fbeta(scored_pred_evidence, gt_evidence, beta=2.0)
        weighted_span = citation_weighted_span_score(scored_pred_evidence, gt_evidence, recall_weight=0.7)

        if not run_judge:
            judge_out = {
                "answer_score": None,
                "answer_reason": "judge_skipped",
                "citation_support": None,
                "citation_reason": "",
                "judge_model_used": None,
                "judge_fallback_used": False,
            }
            judge_status = "skipped"
        else:
            cache_keys = {
                "question": record["question"],
                "gt_answer": gt_answer,
                "predicted_answer": pred_answer,
                "judge_model_primary": JUDGE_MODEL_NAME,
                "judge_model_fallback": FALLBACK_JUDGE_MODEL_NAME,
                "judge_mode": "answer+citation" if with_citation_judge else "answer_only",
                "judge_prompt_version": JUDGE_PROMPT_VERSION,
                "scoring_version": SCORING_VERSION,
                "reasoning_type": reasoning_type,
                "grading_notes": grading_notes
            }
            if with_citation_judge:
                cache_keys["predicted_evidence"] = scored_pred_evidence

            cache_hash = hashlib.sha256(json.dumps(cache_keys, sort_keys=True).encode("utf-8")).hexdigest()
            if cache_hash in _judge_cache:
                judge_out = _judge_cache[cache_hash]
                judge_status = "cache_hit"
            else:
                _pacer.wait()
                judge_out, judge_status = llm_judge(
                    record["question"], pred_answer, gt_answer,
                    reasoning_type, grading_notes, scored_pred_evidence,
                    with_citation=with_citation_judge
                )
                if judge_status == "ok":
                    append_to_cache(cache_hash, judge_out)

    if with_bertscore:
        bert_out = compute_bertscore(
            pred_answer,
            gt_answer,
            model_type=bertscore_model,
            device=bertscore_device,
            rescale_with_baseline=bertscore_rescale,
        )
    else:
        bert_out = {
            "bertscore_precision": None,
            "bertscore_recall": None,
            "bertscore_f1": None,
            "bertscore_model": bertscore_model,
            "bertscore_status": "skipped",
            "bertscore_error": "",
        }

    return {
        "id": record["id"],
        "method": record.get("method", "unknown"),
        "dataset": record.get("dataset", gt_item.get("dataset", "unknown")),
        "difficulty": record.get("difficulty", gt_item.get("difficulty", "unknown")),
        "reasoning_type": reasoning_type,
        "question": record["question"],
        "predicted_answer": pred_answer,
        "predicted_evidence": pred_evidence,
        "scored_predicted_evidence": scored_pred_evidence,
        "citation_narrowing_applied": sum(1 for ev in scored_pred_evidence if ev.get("narrowing_applied")),
        "latency_sec": record.get("latency_sec", 0.0),
        "input_tokens": record.get("input_tokens", 0),
        "output_tokens": record.get("output_tokens", 0),
        "model_calls": record.get("model_calls", 1),
        "error_type": error_type,
        "raw_error": record.get("raw_error"),
        "judge_status": judge_status,
        "judge_model_requested": JUDGE_MODEL_NAME,
        "judge_model_fallback": FALLBACK_JUDGE_MODEL_NAME,
        "judge_model_used": judge_out.get("judge_model_used"),
        "judge_fallback_used": judge_out.get("judge_fallback_used", False),
        "answer_score": judge_out["answer_score"],
        "answer_reason": judge_out.get("answer_reason", ""),
        "citation_support_llm": judge_out["citation_support"],
        "citation_reason": judge_out.get("citation_reason", ""),
        "citation_file_precision": cit_fp,
        "citation_file_recall": cit_fr,
        "citation_file_f1": cit_ff1,
        "evidence_line_iou": liou,
        "citation_support_score": cit_support_geom,
        "citation_containment_recall": containment_recall,
        "citation_span_precision": span_precision,
        "citation_span_f2": span_f2,
        "citation_weighted_span_score": weighted_span,
        "citation_span_metric_basis": "scored_predicted_evidence",
        **bert_out,
    }


def load_ground_truth(qa_file: str) -> Dict[str, Dict]:
    with open(qa_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    return {item["id"]: item for item in data}


def score_predictions_file(predictions_file: str, gt_map: Dict[str, Dict],
                           output_dir: str, run_judge: bool = True,
                           with_citation_judge: bool = False,
                           subset_ids: Optional[set] = None,
                           with_bertscore: bool = True,
                           bertscore_model: str = BERTSCORE_DEFAULT_MODEL,
                           bertscore_device: Optional[str] = BERTSCORE_DEFAULT_DEVICE,
                           bertscore_rescale: bool = False) -> str:
    basename = os.path.basename(predictions_file).replace("predictions_", "scored_")
    if not basename.endswith(".jsonl"):
        basename = os.path.splitext(basename)[0] + ".jsonl"
    out_path = os.path.join(output_dir, basename)

    with open(predictions_file, "r", encoding="utf-8") as f:
        lines = [json.loads(l) for l in f if l.strip()]

    target_records = [rec for rec in lines if not subset_ids or rec["id"] in subset_ids]
    if not target_records:
        print(f"No records matching subset filter in {predictions_file}")
        return ""

    mode = "answer+citation" if with_citation_judge else "answer-only"
    if with_bertscore:
        mode += f"+bertscore:{bertscore_model}"
    print(f"\nScoring {len(target_records)} predictions from {os.path.basename(predictions_file)} [{mode}]...")

    scored_records = []
    for i, record in enumerate(target_records):
        qid = record["id"]
        if qid not in gt_map:
            print(f"  Warning: ID {qid} not in ground truth, skipping.")
            continue
        scored_records.append(
            score_record(
                record,
                gt_map[qid],
                run_judge=run_judge,
                with_citation_judge=with_citation_judge,
                with_bertscore=with_bertscore,
                bertscore_model=bertscore_model,
                bertscore_device=bertscore_device,
                bertscore_rescale=bertscore_rescale,
            )
        )
        if (i + 1) % 5 == 0:
            print(f"  {i + 1}/{len(target_records)} scored...")

    with open(out_path, "w", encoding="utf-8") as f:
        for rec in scored_records:
            f.write(json.dumps(rec) + "\n")

    print(f"Scored results saved to {out_path}")
    _print_summary(scored_records)
    return out_path


def _print_summary(records: List[Dict]):
    n = len(records)
    if n == 0:
        return

    def mean(vals):
        return sum(vals) / len(vals) if vals else 0.0

    method = records[0].get("method", "?")
    ans_scores = [r["answer_score"] for r in records if r.get("answer_score") is not None]
    cit_supp_scores = [r["citation_support_llm"] for r in records if r.get("citation_support_llm") is not None]
    cit_f1 = [r["citation_file_f1"] for r in records]
    liou = [r["evidence_line_iou"] for r in records]
    geom_support = [r["citation_support_score"] for r in records]
    containment = [r["citation_containment_recall"] for r in records]
    span_precision = [r["citation_span_precision"] for r in records]
    span_f2 = [r["citation_span_f2"] for r in records]
    weighted_span = [r["citation_weighted_span_score"] for r in records]
    bert_f1 = [r["bertscore_f1"] for r in records if r.get("bertscore_f1") is not None]

    statuses = [r["judge_status"] for r in records]
    n_ok_primary = statuses.count("ok_primary")
    n_ok_fallback = statuses.count("ok_fallback")
    n_hit = statuses.count("cache_hit")
    n_err = statuses.count("api_error") + statuses.count("fallback_api_error")
    n_fallback_used = sum(1 for r in records if r.get("judge_fallback_used"))

    print(f"\n{'=' * 55}")
    print(f"  Method {method} - {n} samples scored")
    print(f"  Judge Status API: {n_ok_primary} primary, {n_ok_fallback} fallback, Cache: {n_hit} hits, Errors: {n_err}")
    print(f"  Judge Fallback Used    : {n_fallback_used}")
    print(f"{'=' * 55}")
    print(f"  answer_score            : {mean(ans_scores):.3f}" if ans_scores else "  answer_score            : N/A")
    if cit_supp_scores:
        print(f"  citation_support_llm    : {mean(cit_supp_scores):.3f}")
    else:
        print("  citation_support_llm    : skipped (use --with-citation-judge)")
    print(f"  citation_file_f1        : {mean(cit_f1):.3f}")
    print(f"  containment_recall      : {mean(containment):.3f}")
    print(f"  span_precision          : {mean(span_precision):.3f}")
    print(f"  citation_span_f2        : {mean(span_f2):.3f}")
    print(f"  weighted_span_score     : {mean(weighted_span):.3f}")
    if bert_f1:
        bert_status_counts = {}
        for r in records:
            status = r.get("bertscore_status", "unknown")
            bert_status_counts[status] = bert_status_counts.get(status, 0) + 1
        print(f"  bertscore_f1           : {mean(bert_f1):.3f} {bert_status_counts}")
    else:
        print("  bertscore_f1           : unavailable/skipped (default on; check bertscore_status or --no-bertscore)")
    print(f"  evidence_line_iou       : {mean(liou):.3f} (legacy)")
    print(f"  citation_support_score  : {mean(geom_support):.3f} (legacy IoU support)")
    print(f"{'=' * 55}")


def main():
    parser = argparse.ArgumentParser(description="Score raw predictions from evaluate.py")
    parser.add_argument("--predictions", type=str, default="results/raw",
                        help="Path to JSONL predictions file or directory of JSONL files")
    parser.add_argument("--ground-truth", type=str, default="qa_dataset/seed_v0.json")
    parser.add_argument("--output", type=str, default="results/scored")
    parser.add_argument("--no-judge", action="store_true",
                        help="Skip LLM-as-judge entirely")
    parser.add_argument("--with-citation-judge", action="store_true",
                        help="Include citation_support in the single LLM call")
    parser.add_argument("--with-bertscore", action="store_true",
                        help="Deprecated: BERTScore is computed by default unless --no-bertscore is set")
    parser.add_argument("--no-bertscore", action="store_true",
                        help="Disable automatic BERTScore computation")
    parser.add_argument("--bertscore-model", default=BERTSCORE_DEFAULT_MODEL,
                        help="Hugging Face model for BERTScore, default from BERTSCORE_MODEL or roberta-large")
    parser.add_argument("--bertscore-device", default=BERTSCORE_DEFAULT_DEVICE,
                        help="Optional BERTScore device, e.g. cuda:0 or cpu")
    parser.add_argument("--bertscore-rescale", action="store_true",
                        help="Use BERTScore baseline rescaling when available")
    parser.add_argument("--rpm-limit", type=int, default=4,
                        help="Rate limit for the judge API in requests per minute")
    parser.add_argument("--subset", type=str, default=None,
                        help="Path to a text file containing issue IDs to evaluate")
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
        with open(args.subset, "r", encoding="utf-8") as f:
            subset_ids = {line.strip() for line in f if line.strip()}
        print(f"Loaded {len(subset_ids)} target IDs from subset filter.")

    if os.path.isdir(args.predictions):
        files = glob.glob(os.path.join(args.predictions, "predictions_*.jsonl"))
        if not files:
            print(f"No predictions_*.jsonl found in {args.predictions}")
            return
    else:
        files = [args.predictions]

    for path in sorted(files):
        score_predictions_file(
            path,
            gt_map,
            args.output,
            run_judge=not args.no_judge,
            with_citation_judge=args.with_citation_judge,
            subset_ids=subset_ids,
            with_bertscore=not args.no_bertscore,
            bertscore_model=args.bertscore_model,
            bertscore_device=args.bertscore_device,
            bertscore_rescale=args.bertscore_rescale,
        )

    print("\nAll scoring complete.")


if __name__ == "__main__":
    main()
