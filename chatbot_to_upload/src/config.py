import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Project Root
ROOT_DIR = Path(__file__).parent.parent
DATA_DIR = ROOT_DIR / "data"
VECTOR_DB_DIR = ROOT_DIR / "chroma_db"

# Model Configuration
LLM_BACKEND = os.getenv("LLM_BACKEND", "local").lower()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL_NAME = os.getenv("GEMINI_MODEL_NAME", "gemini-2.0-flash-lite")

LLM_MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
EMBEDDING_BACKEND = os.getenv("EMBEDDING_BACKEND", "local").lower()
EMBEDDING_MODEL_ID = "intfloat/multilingual-e5-large"
GEMINI_EMBEDDING_MODEL = "models/text-embedding-004"

# Quantization Settings
USE_4BIT = True
BNB_4BIT_COMPUTE_DTYPE = "float16" # or "bfloat16" if supported
BNB_4BIT_QUANT_TYPE = "nf4"

# Retrieval Settings
TOP_K_RETRIEVAL = 5

# Advanced Retrieval Features
QUERY_REWRITE_ENABLED = os.getenv("QUERY_REWRITE_ENABLED", "true").lower() == "true"
BM25_ENABLED = os.getenv("BM25_ENABLED", "true").lower() == "true"
RERANKER_ENABLED = os.getenv("RERANKER_ENABLED", "false").lower() == "true"
COMPRESSION_ENABLED = os.getenv("COMPRESSION_ENABLED", "false").lower() == "true"

COMPRESSION_THRESHOLD = float(os.getenv("COMPRESSION_THRESHOLD", "0.4"))
RERANKER_MODEL_ID = os.getenv("RERANKER_MODEL_ID", "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1")
RERANKER_TOP_N = int(os.getenv("RERANKER_TOP_N", "4"))
RRF_K = int(os.getenv("RRF_K", "60"))
BM25_INDEX_PATH = ROOT_DIR / os.getenv("BM25_INDEX_PATH", "data/bm25_index.pkl")

# ChromaDB Settings
COLLECTION_NAME = "brainlight_rag"
