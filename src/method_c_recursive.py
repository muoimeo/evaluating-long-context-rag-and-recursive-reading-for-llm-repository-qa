import os
import json
import argparse
import time
from typing import List, Dict, Any
from openai import OpenAI
from config import OLLAMA_BASE_URL, API_KEY, MODEL_NAME, INDEX_PATH
from vector_store import VectorStore

client = OpenAI(base_url=OLLAMA_BASE_URL, api_key=API_KEY)


def extract_from_chunk(query: str, file_path: str, chunk_text: str, line_start: int, line_end: int) -> Dict[str, Any]:
    """Sub-LLM call: Read one chunk and extract information relevant to the query."""
    
    # Add line numbers to help the model cite accurately
    numbered_lines = []
    for i, line in enumerate(chunk_text.split('\n')):
        numbered_lines.append(f"{line_start + i}: {line}")
    numbered_text = '\n'.join(numbered_lines)

    prompt = f"""You are reading a section of the file '{file_path}' (lines {line_start}-{line_end}).

--- CONTENT ---
{numbered_text}
--- END ---

QUERY: {query}

Extract information from this section that is useful for answering the query.
If this section contains NO relevant information, respond with exactly: {{"relevant": false, "confidence": 0.0}}
If it IS relevant, respond with:
{{
  "relevant": true,
  "confidence": <float 0.0-1.0, how confident you are this chunk helps answer the query>,
  "extracted_info": "<concise summary of what you found>",
  "key_lines": [<list of line numbers that are most important>]
}}
Respond ONLY with valid JSON."""

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=500
        )
        
        content = response.choices[0].message.content.strip()
        tokens_used = response.usage.prompt_tokens + response.usage.completion_tokens if response.usage else 0
        
        # Try to parse JSON from the response
        try:
            result = json.loads(content)
        except json.JSONDecodeError:
            # Try to extract JSON from markdown code blocks
            if "```" in content:
                json_str = content.split("```")[1]
                if json_str.startswith("json"):
                    json_str = json_str[4:]
                result = json.loads(json_str.strip())
            else:
                result = {"relevant": False}
        
        result["tokens_used"] = tokens_used
        result["file"] = file_path
        result["line_start"] = line_start
        result["line_end"] = line_end
        return result
        
    except Exception as e:
        return {"relevant": False, "error": str(e), "tokens_used": 0}


def synthesize_answer(query: str, buffer: List[Dict]) -> Dict[str, Any]:
    """Final LLM call: Combine all extracted info from the buffer into a final answer with citations."""
    
    gathered_info = []
    for i, item in enumerate(buffer):
        gathered_info.append(
            f"[Source {i+1}] File: {item['file']} (Lines {item['line_start']}-{item['line_end']})\n"
            f"  Info: {item.get('extracted_info', 'N/A')}\n"
            f"  Key lines: {item.get('key_lines', [])}"
        )
    
    info_text = "\n\n".join(gathered_info)
    
    prompt = f"""You have read through multiple sections of a code repository and documentation.
Here is the information gathered from reading:

{info_text}

---
QUERY: {query}

Based ONLY on the information gathered above, provide:
1. A clear, accurate answer to the query  
2. Exact citations with file paths and line ranges

Respond in this exact JSON format:
{{
  "answer": "<your answer>",
  "evidence": [
    {{"file": "<file path>", "line_start": <int>, "line_end": <int>}}
  ]
}}
Respond ONLY with valid JSON."""

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=2000
        )
        
        content = response.choices[0].message.content.strip()
        tokens_used = response.usage.prompt_tokens + response.usage.completion_tokens if response.usage else 0
        
        try:
            result = json.loads(content)
        except json.JSONDecodeError:
            if "```" in content:
                json_str = content.split("```")[1]
                if json_str.startswith("json"):
                    json_str = json_str[4:]
                result = json.loads(json_str.strip())
            else:
                return {"answer": content, "evidence": [], "tokens_used": tokens_used}
        
        result["tokens_used"] = tokens_used
        return result
        
    except Exception as e:
        return {"answer": f"Error: {e}", "evidence": [], "tokens_used": 0}


# Minimum confidence score to accept a chunk into the buffer.
# Chunks below this threshold are discarded to reduce noise/hallucination.
CONFIDENCE_THRESHOLD = 0.4

def run_method_c(question: str, top_k: int = 10) -> Dict[str, Any]:
    """
    Run Method C: Recursive Reading (RLM-style).
    
    Flow:
      query → retrieve candidate chunks → for each chunk: sub-LLM extracts info
      → confidence filter → store in buffer → final LLM synthesizes answer from buffer
    """
    print("=" * 50)
    print("Method C: Recursive Reading (RLM)")
    print("=" * 50)
    
    start_time = time.time()
    total_tokens = 0
    model_calls = 0
    
    # Step 1: Use vector store to find candidate chunks
    print(f"\nStep 1: Retrieving top-{top_k} candidate chunks...")
    vs = VectorStore()
    results = vs.retrieve(question, top_k=top_k)
    
    if not results:
        return {"success": False, "error": "No chunks retrieved.", "latency": time.time() - start_time}
    
    print(f"  Found {len(results)} candidate chunks.")
    
    # Step 2: Read each chunk with a sub-LLM call (the "recursive reading" step)
    print("\nStep 2: Reading chunks one-by-one with sub-LLM...")
    buffer = []
    
    for i, res in enumerate(results):
        meta = res['metadata']
        file_path = meta['file']
        chunk_idx = meta['chunk_index']
        line_start = meta.get('line_start', 1)
        line_end = meta.get('line_end', 1)
        chunk_text = res['document']
        
        print(f"  [{i+1}/{len(results)}] Reading {file_path} (lines {line_start}-{line_end})...", end=" ")
        
        extraction = extract_from_chunk(question, file_path, chunk_text, line_start, line_end)
        model_calls += 1
        total_tokens += extraction.get("tokens_used", 0)
        
        is_relevant = extraction.get("relevant", False)
        confidence = extraction.get("confidence", 0.0)
        
        if is_relevant and confidence >= CONFIDENCE_THRESHOLD:
            buffer.append(extraction)
            print(f"✓ RELEVANT (confidence={confidence:.2f})")
        elif is_relevant:
            print(f"✗ low confidence ({confidence:.2f} < {CONFIDENCE_THRESHOLD})")
        else:
            print("✗ not relevant")
    
    print(f"\n  Buffer: {len(buffer)} high-confidence chunks out of {len(results)} candidates.")
    
    if not buffer:
        return {
            "success": False, 
            "error": "No relevant information found in any chunk.",
            "latency": time.time() - start_time,
            "input_tokens": total_tokens,
            "output_tokens": 0,
            "model_calls": model_calls
        }
    
    # Step 3: Synthesize final answer from buffer
    print("\nStep 3: Synthesizing final answer from buffer...")
    synthesis = synthesize_answer(question, buffer)
    model_calls += 1
    total_tokens += synthesis.get("tokens_used", 0)
    
    latency = time.time() - start_time
    
    return {
        "success": True,
        "answer": synthesis.get("answer", ""),
        "evidence": synthesis.get("evidence", []),
        "latency": latency,
        "input_tokens": total_tokens,  # approximate total
        "output_tokens": 0,
        "model_calls": model_calls
    }


def main():
    parser = argparse.ArgumentParser(description="Run Method C: Recursive Reading (RLM)")
    parser.add_argument("--query", type=str, required=True, help="Question to ask")
    parser.add_argument("--top-k", type=int, default=10, help="Number of candidate chunks to read")
    args = parser.parse_args()
    
    result = run_method_c(args.query, top_k=args.top_k)
    
    if result.get("success"):
        print("\n" + "="*50)
        print("FINAL ANSWER:")
        print(result["answer"])
        print("\nEVIDENCE:")
        for ev in result["evidence"]:
            print(f"  - {ev.get('file')} (Lines {ev.get('line_start')} to {ev.get('line_end')})")
        print("="*50)
        print(f"Metrics: Latency={result['latency']:.2f}s | Total Tokens≈{result.get('input_tokens')} | Model Calls={result.get('model_calls')}")
    else:
        print(f"\nError occurred: {result.get('error')}")

if __name__ == "__main__":
    main()
