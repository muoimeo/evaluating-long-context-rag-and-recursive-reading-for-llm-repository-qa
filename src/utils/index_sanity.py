"""
Sanity checks for the retrieval corpus and vector index.
"""
import os
import json
import argparse

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config import INDEX_PATH
from vector_store import VectorStore


def scan_index(index_path: str):
    with open(index_path, "r", encoding="utf-8") as f:
        docs = json.load(f)
    banned = [d["file"] for d in docs if "/tests/" in f"/{d['file']}/" or d["file"].startswith("tests/")]
    print(f"Indexed chunks: {len(docs)}")
    print(f"Banned test-path chunks found: {len(banned)}")
    if banned:
        for path in sorted(set(banned))[:20]:
            print(f"  BAD: {path}")
    return banned


def run_queries(queries, top_k: int):
    vs = VectorStore()
    print(f"Manifest status: {'ok' if vs.validate_manifest(INDEX_PATH, raise_on_mismatch=False) else 'missing_or_mismatch'}")
    for query in queries:
        print(f"\nQuery: {query}")
        results = vs.retrieve(query, top_k=top_k)
        for idx, res in enumerate(results, start=1):
            meta = res["metadata"]
            print(f"  {idx}. {meta['file']} [{meta.get('line_start')}-{meta.get('line_end')}] score={res['score']:.4f}")


def main():
    parser = argparse.ArgumentParser(description="Sanity check retrieval index and vector store")
    parser.add_argument("--index", type=str, default=INDEX_PATH)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--query", action="append", default=[], help="Optional retrieval sanity query. Can be repeated.")
    args = parser.parse_args()

    if os.path.basename(os.getcwd()) == "src":
        os.chdir("..")

    banned = scan_index(args.index)
    if not args.query:
        args.query = [
            "FastAPI routing request body validation",
            "AWS Lambda blank python lambda_handler return AccountUsage",
        ]
    run_queries(args.query, top_k=args.top_k)

    if banned:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
