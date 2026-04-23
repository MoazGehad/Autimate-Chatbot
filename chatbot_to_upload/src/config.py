import os
from pathlib import Path

# Project Root
ROOT_DIR = Path(__file__).parent.parent
DATA_DIR = ROOT_DIR / "data"
VECTOR_DB_DIR = ROOT_DIR / "chroma_db"

# Model Configuration
LLM_MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
EMBEDDING_MODEL_ID = "intfloat/multilingual-e5-large"

# Quantization Settings
USE_4BIT = True
BNB_4BIT_COMPUTE_DTYPE = "float16" # or "bfloat16" if supported
BNB_4BIT_QUANT_TYPE = "nf4"

# Retrieval Settings
TOP_K_RETRIEVAL = 5

# ChromaDB Settings
COLLECTION_NAME = "brainlight_rag"
