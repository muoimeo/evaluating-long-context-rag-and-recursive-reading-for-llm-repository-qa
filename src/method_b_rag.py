import argparse
import time
from typing import Dict, Any, List

from openai import OpenAI

from vector_store import VectorStore
from config import TOP_K_RETRIEVAL, INDEX_PATH, OLLAMA_BASE_URL, API_KEY, MODEL_NAME
from reranker import rerank_and_select, source_first_filter

client = OpenAI(base_url=OLLAMA_BASE_URL, api_key=API_KEY)


def _dataset_of_path(file_path: str) -> str:
    fastapi_root_files = {"pyproject.toml", "README.md", "CONTRIBUTING.md"}
    if (
        file_path.startswith("fastapi/")
        or "docs/" in file_path
        or file_path.startswith("docs_src/")
        or file_path in fastapi_root_files
    ):
        return "fastapi"
    return "lambda"


def _question_mode(question: str) -> str:
    q = question.lower()
    if any(term in q for term in ["compare", "both ", "difference", "differences"]):
        return "compare"
    if any(term in q for term in ["trace", "flow", "through to", "from the"]):
        return "trace"
    return "extract"


def _format_chunks(results: List[Dict[str, Any]]) -> str:
    """Format retrieved chunks with file/line boundaries for synthesis."""
    context_blocks = []
    current_file = None
    for res in results:
        meta = res["metadata"]
        file_path = meta["file"]
        chunk_idx = meta["chunk_index"]
        total_chunks = meta["total_chunks"]
        line_start = meta.get("line_start", "?")
        chunk_text = res["document"]

        if file_path != current_file:
            if current_file is not None:
                context_blocks.append("\n")
            context_blocks.append(f"--- FILE: {file_path} ---")
            current_file = file_path

        context_blocks.append(f"[Chunk {chunk_idx + 1}/{total_chunks} | Lines {line_start}-{meta.get('line_end', '?')}]")
        lines = chunk_text.split("\n")
        for i, line in enumerate(lines):
            actual_line_num = line_start + i if isinstance(line_start, int) else i + 1
            context_blocks.append(f"{actual_line_num}: {line}")
    return "\n".join(context_blocks)


def build_rag_context(query: str, top_k: int = TOP_K_RETRIEVAL) -> List[Dict[str, Any]]:
    """Retrieve top-K chunks, re-rank by source-code priority, and format with line numbers."""
    vs = VectorStore()
    vs.validate_manifest(INDEX_PATH, raise_on_mismatch=True)
    
    # 1. Expand retrieval pool: fetch 4x more candidates (e.g. 20 chunks if top_k is 5)
    # This ensures core source code chunks have a chance to be retrieved even if 
    # highly-semantic tutorial docs try to crowd them out.
    results = vs.retrieve(query, top_k=top_k * 4)
    
    if not results:
        print("Warning: No results retrieved from VectorStore.")
        return []
    
    # Re-rank a slightly larger window, then apply the shared source-first
    # filter so docs do not consume the whole top_k budget.
    final_results = source_first_filter(rerank_and_select(results, top_k * 2, query))[:top_k]
    
    # Sort for display so it reads linearly: group by file, then by chunk index
    final_results.sort(key=lambda x: (x["metadata"]["file"], x["metadata"]["chunk_index"]))
    return final_results


def select_answer_evidence(results: List[Dict[str, Any]], question: str, max_chunks: int = 4) -> List[Dict[str, Any]]:
    """Pick a small, diverse evidence set; for compare questions, force both corpora when possible."""
    if not results:
        return []

    mode = _question_mode(question)
    selected: List[Dict[str, Any]] = []
    seen_ids = set()

    if mode == "compare":
        for dataset_name in ("fastapi", "lambda"):
            for res in results:
                if res["id"] in seen_ids:
                    continue
                if _dataset_of_path(res["metadata"]["file"]) != dataset_name:
                    continue
                selected.append(res)
                seen_ids.add(res["id"])
                break

    for res in results:
        if res["id"] in seen_ids:
            continue
        selected.append(res)
        seen_ids.add(res["id"])
        if len(selected) >= max_chunks:
            break

    return selected[:max_chunks]


def synthesize_answer(question: str, evidence_chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Question-aware synthesis with system-selected evidence instead of citation-heavy one-shot generation."""
    context = _format_chunks(evidence_chunks)
    mode = _question_mode(question)

    task_instruction = {
        "compare": (
            "This is a comparison question. You MUST cover both sides explicitly. "
            "State the mechanism on side A and side B, then the key difference."
        ),
        "trace": (
            "This is a trace question. Answer as a short ordered flow. "
            "Name the concrete files/functions/classes from the evidence."
        ),
        "extract": (
            "This is an extractive question. Give the direct implementation-grounded answer "
            "using the exact symbol/function/class names visible in the evidence."
        ),
    }[mode]

    system_prompt = (
        "You answer repository QA questions using ONLY the provided evidence.\n"
        "Be concrete, not generic.\n"
        "Prefer exact file names, function names, class names, and return values from the evidence.\n"
        "Do not give a vague intro like 'based on the context'.\n"
        "If the evidence only supports part of the question, answer that part and state the missing part explicitly.\n"
        f"{task_instruction}"
    )
    user_prompt = f"QUESTION:\n{question}\n\nEVIDENCE:\n{context}\n\nWrite a concise grounded answer."

    start_time = time.time()
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.0,
        max_tokens=900,
    )
    latency = time.time() - start_time
    answer_text = response.choices[0].message.content.strip()
    return {
        "success": True,
        "answer": answer_text,
        "evidence": [
            {
                "file": r["metadata"]["file"],
                "line_start": r["metadata"].get("line_start", 1),
                "line_end": r["metadata"].get("line_end", 1),
            }
            for r in evidence_chunks
        ],
        "latency": latency,
        "input_tokens": response.usage.prompt_tokens if response.usage else 0,
        "output_tokens": response.usage.completion_tokens if response.usage else 0,
        "model_calls": 1,
    }

def run_method_b(question: str, top_k: int = TOP_K_RETRIEVAL):
    """Run Method B (Standard RAG) on a given query."""
    print(f"Retrieving top {top_k} most relevant chunks from VectorStore...")
    results = build_rag_context(question, top_k)
    
    if not results:
        return {
            "success": False,
            "error": "No chunks retrieved.",
            "latency": 0.0,
            "input_tokens": 0,
            "output_tokens": 0,
            "model_calls": 0
        }

    selected_evidence = select_answer_evidence(results, question)

    print("\nSynthesizing answer from selected evidence...")
    result = synthesize_answer(question=question, evidence_chunks=selected_evidence)
    
    return result

def main():
    parser = argparse.ArgumentParser(description="Run Method B: Standard RAG")
    parser.add_argument("--query", type=str, required=True, help="Question to ask the repository")
    parser.add_argument("--top-k", type=int, default=TOP_K_RETRIEVAL, help="Number of chunks to retrieve")
    args = parser.parse_args()
    
    result = run_method_b(args.query, top_k=args.top_k)
    
    if result.get("success"):
        print("\n" + "="*50)
        print("ANSWER:")
        print(result["answer"])
        print("\nEVIDENCE:")
        for ev in result["evidence"]:
            print(f"  - {ev.get('file')} (Lines {ev.get('line_start')} to {ev.get('line_end')})")
        print("="*50)
        print(f"Metrics: Latency={result['latency']:.2f}s | Input Tokens={result['input_tokens']} | Output Tokens={result['output_tokens']}")
    else:
        print(f"\nError occurred: {result.get('error')}")

if __name__ == "__main__":
    main()
