import os
import json
import argparse
import tiktoken
from tqdm import tqdm
from config import CHUNK_SIZE, CHUNK_OVERLAP, DATASET_DIRS, INDEX_PATH

# Use a generic tokenizer for chunking (cl100k_base used by GPT models).
# For Qwen, this serves as a very good approximation metric for chunk boundaries.
enc = tiktoken.get_encoding("cl100k_base")

def get_file_language(file_path):
    ext = os.path.splitext(file_path)[1].lower()
    mapping = {
        '.py': 'python',
        '.js': 'javascript',
        '.ts': 'typescript',
        '.go': 'go',
        '.java': 'java',
        '.json': 'json',
        '.yml': 'yaml',
        '.yaml': 'yaml',
        '.sh': 'shell',
        '.md': 'markdown'
    }
    return mapping.get(ext, 'text')

def should_ingest(file_path):
    ext = os.path.splitext(file_path)[1].lower()
    valid_exts = {'.py', '.js', '.ts', '.go', '.java', '.json', '.yml', '.yaml', '.sh', '.md'}
    if ext not in valid_exts:
        return False
        
    # Exclude virtual environments, git history, and compiled caches
    parts = file_path.replace("\\", "/").split("/")
    if any(p in ('.git', 'venv', '__pycache__', 'node_modules', '.pytest_cache') for p in parts):
        return False
    
    # CRITICAL: Exclude test directories, .github etc.
    # We DO NOT exclude docs/ here because the AWS Lambda dataset IS documentation,
    # and sometimes FastAPI docs are needed. We handle docs vs source bias via
    # re-ranking in method_b_rag.py instead of hard filtering.
    excluded_dirs = {'tests', 'scripts', '.github'}
    if any(p in excluded_dirs for p in parts):
        return False
    
    # Also exclude test files by name pattern
    basename = os.path.basename(file_path)
    if basename.startswith('test_') or basename.endswith('.test.py'):
        return False
        
    return True

def chunk_by_lines(content, chunk_size, chunk_overlap=CHUNK_OVERLAP):
    """Split text into chunks by lines, tracking exact line_start and line_end.
    Includes an overlap of ~chunk_overlap tokens to preserve boundary context."""
    lines = content.split('\n')
    chunks = []
    
    i = 0
    while i < len(lines):
        current_chunk_lines = []
        current_tokens = 0
        chunk_start_line = i + 1  # 1-indexed
        
        # Build chunk forward
        while i < len(lines):
            line = lines[i]
            line_tokens = len(enc.encode(line))
            
            # If adding this line exceeds size (and we already have lines), break to flush
            if current_tokens + line_tokens > chunk_size and current_chunk_lines:
                break
                
            current_chunk_lines.append(line)
            current_tokens += line_tokens
            i += 1
            
        chunks.append({
            "text": '\n'.join(current_chunk_lines),
            "line_start": chunk_start_line,
            "line_end": chunk_start_line + len(current_chunk_lines) - 1
        })
        
        if i >= len(lines):
            break
            
        # Backtrack `i` to create overlap for the next chunk
        overlap_tokens = 0
        overlap_lines = 0
        # Walk back from the current line (`i-1`) upwards
        for j in range(i - 1, max(-1, chunk_start_line - 2), -1):
            t = len(enc.encode(lines[j]))
            if overlap_tokens + t > chunk_overlap:
                break
            overlap_tokens += t
            overlap_lines += 1
            
        # Ensure we always advance at least 1 line to prevent infinite loops
        if overlap_lines > 0 and overlap_lines < len(current_chunk_lines):
            i = i - overlap_lines

    return chunks

def ingest_directory(base_dir):
    documents = []
    
    # Walk through all files in the directory tree
    for root, _, files in os.walk(base_dir):
        for file in files:
            file_path = os.path.join(root, file)
            if not should_ingest(file_path):
                continue
                
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
            except UnicodeDecodeError:
                continue # Skip binary files that slip through extension filter
                
            token_count = len(enc.encode(content))
            total_lines = content.count('\n') + 1
            
            # Use forward slashes for cross-platform consistency in metadata
            # We strip the leading 'dataset/' part so the path matches our Q&A seed definitions
            rel_path = os.path.relpath(file_path, start="dataset").replace("\\", "/")
            # Specifically for AWS Lambda, paths in seed avoid the top-level directory names:
            if rel_path.startswith("docs/aws-lambda-developer-guide/"):
                rel_path = rel_path.replace("docs/aws-lambda-developer-guide/", "")
            elif rel_path.startswith("repos/fastapi/"):
                rel_path = rel_path.replace("repos/fastapi/", "")
                
            # CRITICAL FIX: Prepend the file path to the content so the embedding model
            # can mathematically associate the text with its source file.
            prepended_content = f"FILE: {rel_path}\n{content}"
            token_count = len(enc.encode(prepended_content))
            
            # If the file fits in one chunk, use it whole
            if token_count <= CHUNK_SIZE:
                documents.append({
                    "file": rel_path,
                    "language": get_file_language(file_path),
                    "total_tokens": token_count,
                    "content": prepended_content, # Store with prepended FILE path
                    "chunk_index": 0,
                    "total_chunks": 1,
                    "line_start": 1,
                    "line_end": total_lines
                })
            else:
                # Line-aware chunking preserving exact line numbers
                chunks = chunk_by_lines(content, CHUNK_SIZE)
                for idx, chunk_info in enumerate(chunks):
                    # Prepend file path to EACH chunk
                    chunk_text_with_path = f"FILE: {rel_path}\n{chunk_info['text']}"
                    
                    documents.append({
                        "file": rel_path,
                        "language": get_file_language(file_path),
                        "total_tokens": len(enc.encode(chunk_text_with_path)),
                        "content": chunk_text_with_path, # Store with prepended FILE path
                        "chunk_index": idx,
                        "total_chunks": len(chunks),
                        "line_start": chunk_info["line_start"],
                        "line_end": chunk_info["line_end"]
                    })
                    
    return documents

def main():
    parser = argparse.ArgumentParser(description="Ingest dataset files into a chunked JSON index.")
    parser.add_argument("--output", type=str, default=INDEX_PATH, help="Output JSON file path")
    args = parser.parse_args()
    
    # Run from the project root. If executed from src/, adjust paths.
    if os.path.basename(os.getcwd()) == "src":
        os.chdir("..")
        
    all_documents = []
    
    if not os.path.exists("dataset"):
        print("Error: 'dataset' directory not found. Please run from project root.")
        return

    print("Starting ingestion process...")
    for d_dir in DATASET_DIRS:
        if not os.path.exists(d_dir):
            print(f"Warning: Dataset directory {d_dir} not found. Skipping.")
            continue
            
        print(f"Ingesting {d_dir}...")
        docs = ingest_directory(d_dir)
        all_documents.extend(docs)
        
    print(f"Total chunks created: {len(all_documents)}")
    
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(all_documents, f, indent=2)
        
    print(f"Ingestion complete. Chunked index saved to {args.output}")

if __name__ == "__main__":
    main()
