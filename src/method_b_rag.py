import argparse
from vector_store import VectorStore
from llm_client import generate_answer
from config import TOP_K_RETRIEVAL

def build_rag_context(query: str, top_k: int = TOP_K_RETRIEVAL) -> str:
    """Retrieve top-K chunks, re-rank by source-code priority, and format with line numbers."""
    vs = VectorStore()
    
    # 1. Expand retrieval pool: fetch 4x more candidates (e.g. 20 chunks if top_k is 5)
    # This ensures core source code chunks have a chance to be retrieved even if 
    # highly-semantic tutorial docs try to crowd them out.
    results = vs.retrieve(query, top_k=top_k * 4)
    
    if not results:
        print("Warning: No results retrieved from VectorStore.")
        return ""
    
    # Re-rank: Apply a soft penalty/bonus multiplier to the semantic distance score.
    # ChromaDB returns distance: lower score means more similar.
    def adjust_score(res):
        fp = res['metadata']['file']
        score = res['score']
        
        penalty = 1.0 # Default multiplier
        
        # 1. CORE SOURCE IMPLEMENTATION (Bonus / Priority 0)
        if fp.startswith("fastapi/") and fp.endswith(".py"):
            penalty = 0.8  # Strong bonus
        elif fp.startswith("sample-apps/blank-python") and fp.endswith(".py"):
            penalty = 0.8  # Strong bonus for specific target app
        elif "sample-apps/" in fp and (fp.endswith(".py") or fp.endswith(".js") or fp.endswith(".go")):
            penalty = 0.9  # Mild bonus for other sample apps
            
        # 2. DOCUMENTATION & TUTORIALS (Penalty / Priority 4)
        elif "docs_src/" in fp or "docs/en/" in fp or "docs/zh/" in fp:
            penalty = 1.6  # Heavy penalty for FastAPI tutorials
        elif "ExampleCS/" in fp or fp.endswith(".cs"):
            penalty = 1.5  # Penalize C# (we're focusing on Python Lambda usually)
            
        # 3. AWS Lambda pure documentation
        elif fp.endswith(".md"):
            penalty = 1.2
            
        # 4. CONFIG FILES (Heavy Penalty)
        elif fp.endswith((".yml", ".yaml", ".json", ".txt", ".ini")):
            penalty = 1.7
            
        return score * penalty
    
    # Sort by the adjusted distance score (lowest is best)
    results.sort(key=lambda r: adjust_score(r))
    
    # Take only top_k after re-ranking
    final_results = results[:top_k]
    
    # Sort for display so it reads linearly: group by file, then by chunk index
    final_results.sort(key=lambda x: (x['metadata']['file'], x['metadata']['chunk_index']))
    
    context_blocks = []
    current_file = None
    for res in final_results:
        meta = res['metadata']
        file_path = meta['file']
        chunk_idx = meta['chunk_index']
        total_chunks = meta['total_chunks']
        line_start = meta.get('line_start', '?')
        line_end = meta.get('line_end', '?')
        chunk_text = res['document']
        
        if file_path != current_file:
            if current_file is not None:
                context_blocks.append("\n")
            context_blocks.append(f"--- FILE: {file_path} ---")
            current_file = file_path
            
        context_blocks.append(f"[Chunk {chunk_idx + 1}/{total_chunks} | Lines {line_start}-{line_end}]")
        
        # Number the lines within this chunk for accurate citation
        lines = chunk_text.split('\n')
        for i, line in enumerate(lines):
            actual_line_num = line_start + i if isinstance(line_start, int) else i + 1
            context_blocks.append(f"{actual_line_num}: {line}")
        
    return "\n".join(context_blocks)

def run_method_b(question: str, top_k: int = TOP_K_RETRIEVAL):
    """Run Method B (Standard RAG) on a given query."""
    print(f"Retrieving top {top_k} most relevant chunks from VectorStore...")
    context = build_rag_context(question, top_k)
    
    print("\nSending context + query to LLM...")
    result = generate_answer(prompt_context=context, question=question)
    
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
