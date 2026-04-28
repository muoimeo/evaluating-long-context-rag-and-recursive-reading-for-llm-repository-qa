"""
evaluate.py — INFERENCE ONLY

This script runs QA methods and saves raw predictions. It does NOT score results.
Scoring is done separately by score.py, which can be re-run without touching the LLM.

Output:
  results/raw/predictions_{method}_{timestamp}.jsonl  — one JSON object per line
  results/raw/run_config_{timestamp}.json             — frozen copy of eval config
"""
import os
import json
import time
import argparse
import shutil
from datetime import datetime
from typing import Dict, Any, List

# Add src to path if running from root
import sys
sys.path.insert(0, os.path.dirname(__file__))

# EARLY ARG PARSE FOR DYNAMIC MODEL CONFIGURATION
_parser = argparse.ArgumentParser(add_help=False)
_parser.add_argument("--model", type=str, default=None)
_args, _ = _parser.parse_known_args()
if _args.model:
    os.environ["MODEL_NAME"] = _args.model

from config import OLLAMA_BASE_URL, API_KEY, MODEL_NAME
from tqdm import tqdm
from pipeline_schema import (
    ERROR_EMPTY,
    ERROR_MALFORMED_JSON,
    ERROR_NO_CITATION,
    ERROR_TIMEOUT,
    ERROR_TOOL,
    normalize_reasoning_type,
    resolve_iterative_top_k,
)

from method_a_longcontext import run_method_a
from method_b_rag import run_method_b
from method_c_iterative import run_method_c


def normalize_answer_text(answer: Any) -> str:
    """Coerce model outputs into a stable string for downstream evaluation."""
    if answer is None:
        return ""
    if isinstance(answer, str):
        return answer
    if isinstance(answer, dict):
        if isinstance(answer.get("answer"), str):
            return answer["answer"]
        try:
            return json.dumps(answer, ensure_ascii=False)
        except TypeError:
            return str(answer)
    if isinstance(answer, list):
        try:
            return json.dumps(answer, ensure_ascii=False)
        except TypeError:
            return str(answer)
    return str(answer)


def classify_error(result: Dict[str, Any]) -> str | None:
    """Classify method output into an error type, or None if ok."""
    if result.get("success"):
        answer = result.get("answer", "")
        if not answer or not str(answer).strip():
            return ERROR_EMPTY
        return None  # No error

    err = str(result.get("error", ""))
    if "timeout" in err.lower() or "timed out" in err.lower():
        return ERROR_TIMEOUT
    if "json" in err.lower() or "parse" in err.lower() or "decode" in err.lower():
        return ERROR_MALFORMED_JSON
    return ERROR_TOOL


def run_inference(qa_file: str, method_name: str, output_dir: str,
                  config_path: str, num_samples: int = None,
                  subset_ids: set = None) -> str:
    """Run inference for one method and save raw predictions."""
    with open(qa_file, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    if subset_ids:
        dataset = [q for q in dataset if q["id"] in subset_ids]
        print(f"Loaded {len(dataset)} questions using subset filter.")

    if num_samples:
        dataset = dataset[:num_samples]

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_file = os.path.join(output_dir, f"predictions_{method_name.lower()}_{timestamp}.jsonl")

    # Snapshot the eval config
    run_config = {}
    if os.path.isfile(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            run_config = json.load(f)
    run_config["run_timestamp"] = timestamp
    run_config["method"] = method_name
    run_config["qa_file"] = qa_file
    run_config["num_samples"] = num_samples or len(dataset)
    if method_name == "C":
        run_config["resolved_iterative_top_k"] = resolve_iterative_top_k(run_config, default=10)

    config_out = os.path.join(output_dir, f"run_config_{method_name.lower()}_{timestamp}.json")
    with open(config_out, "w", encoding="utf-8") as f:
        json.dump(run_config, f, indent=2)

    print(f"\nEvaluating Method {method_name} on {len(dataset)} questions...")
    print(f"Saving predictions to: {out_file}")

    with open(out_file, "w", encoding="utf-8") as fout:
        for i, record in enumerate(tqdm(dataset, desc=f"Method {method_name}")):
            question = record["question"]
            q_id = record["id"]
            reasoning_type = normalize_reasoning_type(record)

            row = {
                "id": q_id,
                "method": method_name,
                "dataset": record.get("dataset", "unknown"),
                "difficulty": record.get("difficulty", "unknown"),
                "reasoning_type": reasoning_type,
                "question": question,
            }

            try:
                if method_name == "A":
                    result = run_method_a(question)
                elif method_name == "B":
                    result = run_method_b(question)
                elif method_name == "C":
                    top_k = resolve_iterative_top_k(run_config, default=10)
                    result = run_method_c(question, top_k=top_k)
                else:
                    raise ValueError(f"Unknown method: {method_name}")

                error_type = classify_error(result)
                row.update({
                    "predicted_answer": normalize_answer_text(result.get("answer", "")),
                    "predicted_evidence": result.get("evidence", []),
                    "success": error_type is None,
                    "error_type": error_type,
                    "raw_error": result.get("error"),
                    "raw_warning": result.get("warning"),
                    "latency_sec": result.get("latency", 0.0),
                    "input_tokens": result.get("input_tokens", 0),
                    "output_tokens": result.get("output_tokens", 0),
                    "model_calls": result.get("model_calls", 1),
                    "run_config_file": config_out,
                })

            except Exception as e:
                row.update({
                    "predicted_answer": "",
                    "predicted_evidence": [],
                    "success": False,
                    "error_type": ERROR_MALFORMED_JSON,
                    "raw_error": str(e),
                    "raw_warning": None,
                    "latency_sec": 0.0,
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "model_calls": 1,
                    "run_config_file": config_out,
                })

            fout.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Done. {len(dataset)} predictions saved.")
    return out_file


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate QA Methods (inference only). Score separately with score.py")
    parser.add_argument("--qa-file", type=str, default="qa_dataset/seed_v0.json")
    parser.add_argument("--method", type=str, required=True, choices=["A", "B", "C", "all"])
    parser.add_argument("--samples", type=int, default=None,
                        help="Limit questions for fast testing")
    parser.add_argument("--output-dir", type=str, default="results/raw")
    parser.add_argument("--subset", type=str, default="",
                        help="Path to a text file with question IDs (one per line)")
    parser.add_argument("--config", type=str, default="configs/eval_config.json")
    parser.add_argument("--model", type=str, default=None,
                        help="Force override the answer_model from config")
    args = parser.parse_args()

    if os.path.basename(os.getcwd()) == "src":
        os.chdir("..")

    os.makedirs(args.output_dir, exist_ok=True)

    subset_ids = None
    if args.subset:
        with open(args.subset, "r", encoding="utf-8") as f:
            subset_ids = {line.strip() for line in f if line.strip()}
        print(f"Subset filter: {len(subset_ids)} question IDs loaded from {args.subset}")

    methods = ["A", "B", "C"] if args.method == "all" else [args.method]
    for m in methods:
        run_inference(
            qa_file=args.qa_file,
            method_name=m,
            output_dir=args.output_dir,
            config_path=args.config,
            num_samples=args.samples,
            subset_ids=subset_ids,
        )

    print("\nAll inference complete. Run score.py to compute metrics.")


if __name__ == "__main__":
    main()
