import json
import random
import argparse
from llm_client import generate_answer
from config import INDEX_PATH

# Deterministic seed for reproducibility across evaluation runs.
# This ensures that the round-robin file ordering is identical every time,
# which is critical for fair comparison and paper reproducibility.
RANDOM_SEED = 42

# Context window budget. Set to 50k tokens (not 100k) to:
#   1. Be realistic — most commercial LLMs cap at 32k-128k
#   2. Be FAIR vs RAG (top_k=5 ≈ 2500 tokens) and the iterative reader
#      (top_k=10 ≈ 5000 tokens)
#   3. Leave buffer for system prompt + question + output generation
MAX_CONTEXT_TOKENS = 50000


def build_long_context(index_path=INDEX_PATH, max_context_tokens=MAX_CONTEXT_TOKENS):
    """Read actual source files in a fair, shuffled order to avoid repo ordering bias."""
    try:
        with open(index_path, 'r', encoding='utf-8') as f:
            docs = json.load(f)
    except FileNotFoundError:
        return ""
    
    # Deterministic shuffle for reproducibility
    random.seed(RANDOM_SEED)
    
    # Get unique files and group by dataset source
    fastapi_files = []
    lambda_files = []
    
    seen = set()
    for d in docs:
        fp = d['file']
        if fp in seen:
            continue
        seen.add(fp)
        if fp.startswith("fastapi/"):
            fastapi_files.append(fp)
        else:
            lambda_files.append(fp)
    
    # Shuffle within each group to avoid alphabetical bias
    random.shuffle(fastapi_files)
    random.shuffle(lambda_files)
    
    # Round-robin across repos to ensure FAIR representation
    # This prevents one repo from dominating when we hit the token limit
    interleaved = []
    max_len = max(len(fastapi_files), len(lambda_files))
    for i in range(max_len):
        if i < len(fastapi_files):
            interleaved.append(fastapi_files[i])
        if i < len(lambda_files):
            interleaved.append(lambda_files[i])
    
    context_lines = []
    total_estimated_tokens = 0
    files_added = 0
    
    for file_path in interleaved:
        # Resolve to disk path
        if file_path.startswith("fastapi/"):
            disk_path = f"dataset/repos/{file_path}"
        else:
            disk_path = f"dataset/docs/aws-lambda-developer-guide/{file_path}"
            
        try:
            with open(disk_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
            # Rough token estimate (chars / 4)
            est_tokens = len(content) // 4 
            if total_estimated_tokens + est_tokens > max_context_tokens:
                continue  # Skip this file but keep trying smaller ones
            
            lines = content.split('\n')
            total_lines = len(lines)
            
            # Clear file boundary markers so LLM can distinguish files
            context_lines.append(f"===== FILE START =====")
            context_lines.append(f"PATH: {file_path}")
            context_lines.append(f"LINES: 1-{total_lines}")
            context_lines.append(f"---------------------")
            
            # Add line numbers to help LLM cite lines accurately
            for i, line in enumerate(lines, 1):
                context_lines.append(f"{i}: {line}")
                
            context_lines.append(f"===== FILE END =====")
            context_lines.append("")  # blank line separator
            
            total_estimated_tokens += est_tokens
            files_added += 1
            
        except Exception:
            continue
    
    print(f"Loaded {files_added} files (~{total_estimated_tokens} tokens) via round-robin interleaving.")
    return "\n".join(context_lines)

def run_method_a(question: str):
    """Run the Long-Context baseline on a given question."""
    print("Building long context (round-robin, shuffled, seed=42)...")
    context = build_long_context()
    print(f"Context built. Length: {len(context)} characters.")
    
    print("\nSending to LLM...")
    result = generate_answer(prompt_context=context, question=question)
    
    return result

def main():
    parser = argparse.ArgumentParser(description="Run Method A: Long-Context Baseline")
    parser.add_argument("--query", type=str, required=True, help="Question to ask the repository")
    args = parser.parse_args()
    
    result = run_method_a(args.query)
    
    if result["success"]:
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
