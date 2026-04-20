import os
import json
import argparse
import hashlib
import chromadb
from chromadb.utils import embedding_functions
from tqdm import tqdm
from config import INDEX_PATH, VECTOR_DB_PATH, TOP_K_RETRIEVAL

class VectorStore:
    def __init__(self, db_path=VECTOR_DB_PATH):
        self.db_path = db_path
        # Use BAAI/bge-small-en-v1.5
        self.embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="BAAI/bge-small-en-v1.5")
        
        # Initialize ChromaDB persistent client
        self.client = chromadb.PersistentClient(path=self.db_path)
        
        # Get or create the collection
        self.collection_name = "repo_qa_docs"
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            embedding_function=self.embedding_fn
        )
        self.manifest_path = os.path.join(self.db_path, "index_manifest.json")

    def _index_manifest(self, index_path: str) -> dict:
        with open(index_path, "rb") as f:
            payload = f.read()
        return {
            "index_path": os.path.abspath(index_path),
            "index_sha256": hashlib.sha256(payload).hexdigest(),
            "collection_name": self.collection_name,
        }

    def write_manifest(self, index_path: str):
        os.makedirs(self.db_path, exist_ok=True)
        with open(self.manifest_path, "w", encoding="utf-8") as f:
            json.dump(self._index_manifest(index_path), f, indent=2)

    def validate_manifest(self, index_path: str, raise_on_mismatch: bool = False) -> bool:
        if not os.path.exists(self.manifest_path):
            if raise_on_mismatch:
                raise RuntimeError("Vector index manifest missing. Rebuild the vector store cleanly.")
            return False
        with open(self.manifest_path, "r", encoding="utf-8") as f:
            current = json.load(f)
        expected = self._index_manifest(index_path)
        ok = (current.get("index_sha256") == expected.get("index_sha256") and 
              os.path.normcase(current.get("index_path", "")) == os.path.normcase(expected.get("index_path", "")))
        if not ok and raise_on_mismatch:
            raise RuntimeError(
                "Vector index manifest mismatch. Rebuild the vector store cleanly.\n"
                f"Expected: {expected}\nFound: {current}"
            )
        return ok
        
    def build(self, index_path=INDEX_PATH, batch_size=100, clean: bool = False):
        """Build the vector index from the chunked JSON file."""
        print(f"Loading indexed documents from {index_path}...")
        try:
            with open(index_path, 'r', encoding='utf-8') as f:
                documents = json.load(f)
        except FileNotFoundError:
            print(f"Error: {index_path} not found. Run 'python src/ingest.py' first.")
            return

        if clean:
            try:
                self.client.delete_collection(self.collection_name)
            except Exception:
                pass
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                embedding_function=self.embedding_fn
            )

        print(f"Loaded {len(documents)} chunks. Preparing for insertion...")
        
        ids = []
        texts = []
        metadatas = []
        
        for i, doc in enumerate(documents):
            # Create a unique ID for each chunk: "filepath_chunkindex"
            doc_id = f"{doc['file']}_{doc['chunk_index']}"
            ids.append(doc_id)
            texts.append(doc['content'])
            
            # Metadata must be simple key-value pairs (str, int, float)
            meta = {
                "file": doc['file'],
                "chunk_index": doc['chunk_index'],
                "total_chunks": doc['total_chunks'],
                "language": doc['language'],
                "line_start": doc.get('line_start', 1),
                "line_end": doc.get('line_end', 1)
            }
            metadatas.append(meta)
            
        print(f"Inserting {len(ids)} chunks into ChromaDB at {self.db_path}")
        
        # Insert in batches to avoid memory/API limits
        for i in tqdm(range(0, len(ids), batch_size), desc="Embedding batches"):
            batch_ids = ids[i:i + batch_size]
            batch_texts = texts[i:i + batch_size]
            batch_metadatas = metadatas[i:i + batch_size]
            
            # upsert will insert new items and update existing items with matching IDs
            self.collection.upsert(
                ids=batch_ids,
                documents=batch_texts,
                metadatas=batch_metadatas
            )
        self.write_manifest(index_path)
        print(f"Index built successfully. Collection '{self.collection_name}' has {self.collection.count()} items.")
        
    def retrieve(self, query, top_k=TOP_K_RETRIEVAL):
        """Retrieve top_k most similar chunks for the query."""
        results = self.collection.query(
            query_texts=[query],
            n_results=top_k
        )
        
        # Format the results into a cleaner list of dicts
        formatted_results = []
        if not results['ids'] or not results['ids'][0]:
            return formatted_results
            
        for i in range(len(results['ids'][0])):
            formatted_results.append({
                "id": results['ids'][0][i],
                "score": results['distances'][0][i] if 'distances' in results and results['distances'] else 0.0,
                "document": results['documents'][0][i],
                "metadata": results['metadatas'][0][i]
            })
            
        return formatted_results

def main():
    parser = argparse.ArgumentParser(description="Manage the vector store for RAG.")
    parser.add_argument("--build", action="store_true", help="Build/update the vector database from the indexed JSON")
    parser.add_argument("--clean", action="store_true", help="Recreate the collection before building")
    parser.add_argument("--check-manifest", action="store_true", help="Fail if the vector index manifest mismatches the current index file")
    parser.add_argument("--query", type=str, help="Test a retrieve query")
    parser.add_argument("--top-k", type=int, default=5, help="Number of results to retrieve for test query")
    args = parser.parse_args()
    
    # Run from the project root
    if os.path.basename(os.getcwd()) == "src":
        os.chdir("..")
        
    vs = VectorStore()
    
    if args.build:
        vs.build(clean=args.clean)
    if args.check_manifest:
        ok = vs.validate_manifest(INDEX_PATH, raise_on_mismatch=False)
        print("Manifest OK" if ok else "Manifest mismatch or missing")
        
    if args.query:
        print(f"\nSearching for: '{args.query}'")
        print("-" * 50)
        results = vs.retrieve(args.query, top_k=args.top_k)
        for i, res in enumerate(results):
            meta = res['metadata']
            print(f"\n[Result {i+1}] Score: {res['score']:.4f} | File: {meta['file']} (Chunk {meta['chunk_index']}/{meta['total_chunks']})")
            # Print first 200 chars of the document as preview
            preview = res['document'][:200].replace('\n', ' ') + "..."
            print(f"Content: {preview}")
            
    if not args.build and not args.query:
        # Just show DB stats if no action specified
        print(f"Connected to ChromaDB at {vs.db_path}")
        print(f"Collection '{vs.collection_name}' currently holds {vs.collection.count()} items.")

if __name__ == "__main__":
    main()
