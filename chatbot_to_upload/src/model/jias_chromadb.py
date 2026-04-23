import os
from dotenv import load_dotenv
from langchain_community.embeddings import HuggingFaceEmbeddings
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch
import chromadb
from chromadb.config import Settings

load_dotenv()

# Initialize ChromaDB (persistent local storage)
chroma_client = chromadb.PersistentClient(
    path="./chroma_db",
    settings=Settings(anonymized_telemetry=False)
)

# Get or create collections for different languages (1024-dim for e5-large)
arabic_collection = chroma_client.get_or_create_collection(
    name="arabic",
    metadata={"hnsw:space": "cosine"}
)

english_collection = chroma_client.get_or_create_collection(
    name="english",
    metadata={"hnsw:space": "cosine"}
)

# Embedding model (High-quality multilingual embeddings for A100)
# multilingual-e5-large: 1024-dim, excellent Arabic support
embeddings_model = HuggingFaceEmbeddings(
    model_name="intfloat/multilingual-e5-large",
    model_kwargs={'device': 'cuda'},
    encode_kwargs={'normalize_embeddings': True}
)

# LLM Model - Upgraded for A100 GPU
# Using Qwen2.5-14B-Instruct with 8-bit quantization for better quality
# A100 can handle this easily (14B @ 8-bit ≈ 14GB VRAM)
model_name = "Qwen/Qwen2.5-14B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_name)

# 8-bit Quantization Config (better quality than 4-bit, still efficient)
quantization_config = BitsAndBytesConfig(
    load_in_8bit=True,
    llm_int8_threshold=6.0,
)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=quantization_config,
    device_map="auto",   
    trust_remote_code=True
)

from sentence_transformers import CrossEncoder
# Initialize Reranker (Multilingual, excellent for Arabic)
reranker = CrossEncoder("BAAI/bge-reranker-v2-m3")

