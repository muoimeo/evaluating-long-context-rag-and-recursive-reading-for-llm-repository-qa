import os
from dotenv import load_dotenv

# Load environment variables from .env file if it exists
load_dotenv()

import json

# Try to load model from eval_config
eval_config_path = os.path.join(os.path.dirname(__file__), "..", "configs", "eval_config.json")
eval_model = "qwen2.5:7b"
try:
    if os.path.exists(eval_config_path):
        with open(eval_config_path, "r", encoding="utf-8") as f:
            eval_cfg = json.load(f)
            eval_model = eval_cfg.get("global", {}).get("answer_model", "qwen2.5:7b")
except Exception:
    pass

# --- LLM Configuration (Ollama) ---
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
MODEL_NAME = os.getenv("MODEL_NAME", eval_model)
API_KEY = os.getenv("API_KEY", "ollama")

# --- Ingestion & RAG Configuration ---
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 512))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 64))

DATASET_DIRS = [
    os.path.join("dataset", "repos", "fastapi"),
    os.path.join("dataset", "docs", "aws-lambda-developer-guide")
]

INDEX_PATH = os.getenv("INDEX_PATH", "docs_index.json")
VECTOR_DB_PATH = os.getenv("VECTOR_DB_PATH", "chroma_db")

# --- Retrieval Configuration ---
TOP_K_RETRIEVAL = int(os.getenv("TOP_K_RETRIEVAL", 5))

# --- Judge Model Configuration ---
# Judge should be a stronger / more stable model than the answer model.
# Falls back to the answer model if not explicitly set.
JUDGE_BASE_URL = os.getenv("JUDGE_BASE_URL", OLLAMA_BASE_URL)
JUDGE_MODEL_NAME = os.getenv("JUDGE_MODEL_NAME", MODEL_NAME)
JUDGE_API_KEY = os.getenv("JUDGE_API_KEY", API_KEY)