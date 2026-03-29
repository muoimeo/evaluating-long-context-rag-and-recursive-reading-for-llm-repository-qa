import os
import json
import argparse
import time
from typing import List, Dict, Any
from openai import OpenAI
from config import OLLAMA_BASE_URL, API_KEY, MODEL_NAME, TOP_K_RETRIEVAL
from vector_store import VectorStore

client = OpenAI(base_url=OLLAMA_BASE_URL, api_key=API_KEY)

# Maximum token budget for the reasoning state to prevent prompt bloat.
# The LLM is instructed to keep reasoning under this limit.
MAX_REASONING_TOKENS = 300


def read_and_reason(question: str, state: Dict, 
                    file_path: str, chunk_text: str, 
                    line_start: int, line_end: int,
                    chunk_num: int, total_chunks: int) -> Dict[str, Any]:
    """
    Core RLM step: Read ONE chunk while carrying forward structured reasoning state.
    
    The LLM sees the ORIGINAL code (with line numbers) plus the current structured state.
    It updates the state incrementally — no lossy summary stage.
    
    State is structured as:
      { "known_facts": [...], "reasoning": "...", "open_questions": [...] }
    to prevent drift and keep it compact.
    """
    
    # Add line numbers to help the model cite accurately
    numbered_lines = []
    for i, line in enumerate(chunk_text.split('\n')):
        numbered_lines.append(f"{line_start + i}: {line}")
    numbered_text = '\n'.join(numbered_lines)

    # Format the structured state for the prompt
    state_text = json.dumps(state, indent=2) if state["known_facts"] else "(Empty — this is the first chunk.)"

    prompt = f"""You are reading chunk {chunk_num}/{total_chunks} from a code repository to answer a question.

QUESTION: {question}

--- YOUR STRUCTURED STATE ---
{state_text}
--- END STATE ---

--- NEW CHUNK: {file_path} (Lines {line_start}-{line_end}) ---
{numbered_text}
--- END CHUNK ---

INSTRUCTIONS:
1. Read the new chunk carefully.
2. If this chunk contains useful information, update the state:
   - Add new facts to "known_facts" (keep each fact to ONE concise sentence)
   - Update "reasoning" with your current thinking (max 2-3 sentences, do NOT repeat known_facts)
   - Remove any "open_questions" that are now answered
3. Do NOT repeat earlier facts. Do NOT restate facts already in known_facts.
4. Keep the total state compact (under {MAX_REASONING_TOKENS} words).
5. If this chunk is NOT useful, return the state unchanged.

Respond in this exact JSON format:
{{
  "known_facts": ["<fact 1>", "<fact 2>", "..."],
  "reasoning": "<your current concise reasoning>",
  "open_questions": ["<remaining unknowns>"],
  "chunk_was_useful": <true or false>,
  "confident_enough_to_answer": <true if you can already fully answer the question, false otherwise>
}}
Respond ONLY with valid JSON."""

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=800
        )
        
        content = response.choices[0].message.content.strip()
        tokens_in = response.usage.prompt_tokens if response.usage else 0
        tokens_out = response.usage.completion_tokens if response.usage else 0
        
        # Parse JSON response
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
                # If we can't parse, return state unchanged
                result = {
                    "known_facts": state.get("known_facts", []),
                    "reasoning": state.get("reasoning", ""),
                    "open_questions": state.get("open_questions", []),
                    "chunk_was_useful": False,
                    "confident_enough_to_answer": False
                }
        
        result["tokens_in"] = tokens_in
        result["tokens_out"] = tokens_out
        return result
        
    except Exception as e:
        return {
            "known_facts": state.get("known_facts", []),
            "reasoning": state.get("reasoning", ""),
            "open_questions": state.get("open_questions", []),
            "chunk_was_useful": False,
            "confident_enough_to_answer": False,
            "tokens_in": 0,
            "tokens_out": 0,
            "error": str(e)
        }


def final_answer(question: str, state: Dict, citations: List[Dict]) -> Dict[str, Any]:
    """
    Final step: Convert the accumulated structured state into a final answer.
    
    Citations are system-tracked (grounded), not LLM-generated, so they are
    guaranteed to reference real chunks that were actually read.
    """
    
    # Format citations for reference
    cite_text = "\n".join(
        f"  - {c['file']} (Lines {c['line_start']}-{c['line_end']})" 
        for c in citations
    ) if citations else "  (none gathered)"
    
    state_text = json.dumps(state, indent=2)
    
    prompt = f"""Based on your reading of multiple code chunks, provide a final answer.

QUESTION: {question}

--- YOUR ACCUMULATED STATE ---
{state_text}
--- END STATE ---

--- GROUNDED CITATIONS (verified source locations) ---
{cite_text}
--- END CITATIONS ---

Provide your final answer using ONLY the known_facts and reasoning from your state.
Do NOT generate your own citations or evidence — the system will automatically attach the grounded citations listed above.

Respond in this exact JSON format:
{{
  "answer": "<your complete, accurate answer>"
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
        tokens_in = response.usage.prompt_tokens if response.usage else 0
        tokens_out = response.usage.completion_tokens if response.usage else 0
        
        try:
            result = json.loads(content)
        except json.JSONDecodeError:
            if "```" in content:
                json_str = content.split("```")[1]
                if json_str.startswith("json"):
                    json_str = json_str[4:]
                result = json.loads(json_str.strip())
            else:
                return {"answer": content, "evidence": citations, 
                        "tokens_in": tokens_in, "tokens_out": tokens_out}
        
        result["tokens_in"] = tokens_in
        result["tokens_out"] = tokens_out
        return result
        
    except Exception as e:
        return {"answer": f"Error: {e}", "evidence": [], 
                "tokens_in": 0, "tokens_out": 0}


def run_method_c(question: str, top_k: int = TOP_K_RETRIEVAL) -> Dict[str, Any]:
    """
    Run Method C: Recursive Language Model (RLM) — True Sequential Reading.
    
    Architecture (per RLM paper):
      retrieve chunks → state = {} → for chunk in chunks:
        state = LLM(state + chunk) → if confident: early stop
      → final_answer = LLM(state)
    
    Key design decisions:
      1. Structured state {known_facts, reasoning, open_questions} prevents drift
      2. Reasoning capped at ~300 words to prevent prompt bloat
      3. Citations are SYSTEM-TRACKED (grounded), not LLM-generated
      4. Chunks read in similarity order (best-match first) 
      5. Early-stop when LLM is confident enough to answer
    """
    print("=" * 50)
    print("Method C: Recursive Language Model (RLM)")
    print("=" * 50)
    
    start_time = time.time()
    total_tokens_in = 0
    total_tokens_out = 0
    model_calls = 0
    
    # Step 1: Retrieve candidate chunks
    print(f"\nStep 1: Retrieving top-{top_k} candidate chunks...")
    vs = VectorStore()
    results = vs.retrieve(question, top_k=top_k)
    
    if not results:
        return {
            "success": False, 
            "error": "No chunks retrieved.", 
            "latency": time.time() - start_time,
            "input_tokens": 0,
            "output_tokens": 0,
            "model_calls": 0
        }
    
    # Keep retrieval order (sorted by similarity score, best match first)
    # Do NOT re-sort by file/chunk_index — the retriever's ranking is the
    # best signal for which chunks to read first.
    print(f"  Found {len(results)} candidate chunks. Reading in similarity order...\n")
    
    # Step 2: Sequential reading — the core RLM loop
    # Structured state prevents drift; system-tracked citations prevent hallucination
    state = {
        "known_facts": [],
        "reasoning": "",
        "open_questions": [question]
    }
    grounded_citations = []  # System-tracked, NOT LLM-generated
    
    for i, res in enumerate(results):
        meta = res['metadata']
        file_path = meta['file']
        line_start = meta.get('line_start', 1)
        line_end = meta.get('line_end', 1)
        chunk_text = res['document']
        
        print(f"  [{i+1}/{len(results)}] Reading {file_path} (lines {line_start}-{line_end})...", end=" ")
        
        step_result = read_and_reason(
            question=question,
            state=state,
            file_path=file_path,
            chunk_text=chunk_text,
            line_start=line_start,
            line_end=line_end,
            chunk_num=i + 1,
            total_chunks=len(results)
        )
        
        model_calls += 1
        total_tokens_in += step_result.get("tokens_in", 0)
        total_tokens_out += step_result.get("tokens_out", 0)
        
        # Update the structured state — MERGE facts, don't overwrite
        new_facts = step_result.get("known_facts", [])
        for f in new_facts:
            if f not in state["known_facts"]:
                state["known_facts"].append(f)
        
        # Enforce token budget: keep only the most recent 20 facts
        if len(state["known_facts"]) > 20:
            state["known_facts"] = state["known_facts"][-20:]
        
        # Reasoning and open_questions can be replaced (they're current-state, not cumulative)
        state["reasoning"] = step_result.get("reasoning", state["reasoning"])
        state["open_questions"] = step_result.get("open_questions", state["open_questions"])
        
        useful = step_result.get("chunk_was_useful", False)
        
        # GROUNDED citation: if the LLM says chunk was useful, we record the
        # actual chunk metadata (file path + line range) — NOT what the LLM says.
        if useful:
            grounded_citations.append({
                "file": file_path,
                "line_start": line_start,
                "line_end": line_end
            })
        
        # Early-stop: if LLM is confident enough, skip remaining chunks
        confident = step_result.get("confident_enough_to_answer", False)
        print(f"{'✓ USEFUL' if useful else '— skipped'} | {'🎯 CONFIDENT' if confident else ''} ({len(grounded_citations)} citations)")
        
        # Early-stop: require at least 3 chunks read AND 2 citations to avoid premature answers
        if confident and len(grounded_citations) >= 2 and i >= 2:
            print(f"\n  ⚡ Early stop at chunk {i+1}/{len(results)} — model is confident.")
            break
    
    print(f"\n  Sequential reading complete. {len(grounded_citations)} grounded citations.")
    
    if not state["known_facts"] and not state["reasoning"].strip():
        return {
            "success": False,
            "error": "No relevant information found after reading all chunks.",
            "latency": time.time() - start_time,
            "input_tokens": total_tokens_in,
            "output_tokens": total_tokens_out,
            "model_calls": model_calls
        }
    
    # Step 3: Final answer from accumulated structured state
    print("\nStep 3: Generating final answer from structured state...")
    answer_result = final_answer(question, state, grounded_citations)
    model_calls += 1
    total_tokens_in += answer_result.get("tokens_in", 0)
    total_tokens_out += answer_result.get("tokens_out", 0)
    
    latency = time.time() - start_time
    
    # ALWAYS use grounded citations — never trust LLM-generated evidence
    # This completely prevents citation hallucination in the final output.
    evidence = grounded_citations
    
    return {
        "success": True,
        "answer": answer_result.get("answer", ""),
        "evidence": evidence,
        "latency": latency,
        "input_tokens": total_tokens_in,
        "output_tokens": total_tokens_out,
        "model_calls": model_calls
    }


def main():
    parser = argparse.ArgumentParser(description="Run Method C: Recursive Language Model (RLM)")
    parser.add_argument("--query", type=str, required=True, help="Question to ask")
    parser.add_argument("--top-k", type=int, default=TOP_K_RETRIEVAL, help="Number of candidate chunks to read")
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
        print(f"Metrics: Latency={result['latency']:.2f}s | Tokens(in)={result.get('input_tokens')} | Tokens(out)={result.get('output_tokens')} | Model Calls={result.get('model_calls')}")
    else:
        print(f"\nError occurred: {result.get('error')}")

if __name__ == "__main__":
    main()