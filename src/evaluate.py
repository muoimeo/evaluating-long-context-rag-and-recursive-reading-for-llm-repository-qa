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

from config import OLLAMA_BASE_URL, API_KEY, MODEL_NAME
from tqdm import tqdm

from method_a_longcontext import run_method_a
from method_b_rag import run_method_b
from method_c_recursive import run_method_c


def classify_error(output: Dict) -> str:
    """Assign a fine-grained error_type tag based on what went wrong."""
    if not output.get("success", False):
        err = str(output.get("error", "")).lower()
        if "timeout" in err:
            return "timeout"
        if "json" in err or "parse" in err or "decode" in err:
            return "malformed_json_unrecovered"
        return "tool_error"
    answer = output.get("answer", "").strip()
    evidence = output.get("evidence", [])
    if not answer or answer.lower().startswith("i don't have"):
        return "empty_answer"
    if not evidence:
        return "no_citation"
    return None  # No error


def run_inference(qa_file: str, method_name: str, output_dir: str,
                  config_path: str, num_samples: int = None) -> str:
    """Run inference for one method and save raw predictions."""
    with open(qa_file, "r", encoding="utf-8") as f:
        dataset = json.load(f)
    if num_samples:
        dataset = dataset[:num_samples]

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_file = os.path.join(output_dir, f"predictions_{method_name.lower()}_{timestamp}.jsonl")

    # Load and snapshot the run config
    run_config = {}
    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            run_config = json.load(f)
    run_config["run_timestamp"] = timestamp
    run_config["method"] = method_name
    run_config["qa_file"] = qa_file
    run_config["num_samples"] = num_samples or len(dataset)

    config_out = os.path.join(output_dir, f"run_config_{method_name.lower()}_{timestamp}.json")
    with open(config_out, "w", encoding="utf-8") as f:
        json.dump(run_config, f, indent=2)

    print(f"\nEvaluating Method {method_name} on {len(dataset)} questions...")
    print(f"Saving predictions to: {out_file}")

    records = []
    with open(out_file, "w", encoding="utf-8") as out:
        for item in tqdm(dataset, desc=f"Method {method_name}"):
            question = item["question"]

            try:
                if method_name == "A":
                    output = run_method_a(question)
                elif method_name == "B":
                    output = run_method_b(question)
                elif method_name == "C":
                    output = run_method_c(question, top_k=run_config.get("top_k_rlm", 10))
                else:
                    raise ValueError(f"Unknown method: {method_name}")
            except Exception as e:
                output = {"success": False, "error": str(e),
                          "latency": 0.0, "input_tokens": 0,
                          "output_tokens": 0, "model_calls": 1}

            error_type = classify_error(output)

            record = {
                "id": item["id"],
                "method": method_name,
                "dataset": item.get("dataset", "unknown"),
                "difficulty": item.get("difficulty", "unknown"),
                "expected_reasoning_type": item.get("expected_reasoning_type", "unknown"),
                "question": question,
                "predicted_answer": output.get("answer", ""),
                "predicted_evidence": output.get("evidence", []),
                "success": output.get("success", False),
                "error_type": error_type,
                "latency_sec": output.get("latency", 0.0),
                "input_tokens": output.get("input_tokens", 0),
                "output_tokens": output.get("output_tokens", 0),
                "model_calls": output.get("model_calls", 1),
                "run_config_file": config_out
            }
            out.write(json.dumps(record) + "\n")
            records.append(record)

    print(f"Done. {len(records)} predictions saved.")
    return out_file


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate QA Methods (inference only). Score separately with score.py")
    parser.add_argument("--qa-file", type=str, default="qa_dataset/seed_v0.json")
    parser.add_argument("--method", type=str, required=True, choices=["A", "B", "C", "all"])
    parser.add_argument("--samples", type=int, default=None,
                        help="Limit questions for fast testing")
    parser.add_argument("--output-dir", type=str, default="results/raw")
    parser.add_argument("--config", type=str, default="configs/eval_config.json")
    args = parser.parse_args()

    if os.path.basename(os.getcwd()) == "src":
        os.chdir("..")

    os.makedirs(args.output_dir, exist_ok=True)

    methods = ["A", "B", "C"] if args.method == "all" else [args.method]
    for m in methods:
        run_inference(args.qa_file, m, args.output_dir, args.config, args.samples)

    print(f"\nAll inference complete. Run score.py to compute metrics.")


if __name__ == "__main__":
    main()
