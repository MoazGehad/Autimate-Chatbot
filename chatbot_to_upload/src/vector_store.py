import os
import chromadb
from chromadb.config import Settings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from src.config import VECTOR_DB_DIR, COLLECTION_NAME, EMBEDDING_MODEL_ID
from src.logging_config import get_logger, log_latency
import torch

logger = get_logger(__name__)

def get_embedding_function():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    encode_kwargs = {'normalize_embeddings': True}
    model_kwargs = {'device': device, 'trust_remote_code': True}
    
    # Load in float16 as requested for memory optimization
    if device == "cuda":
        model_kwargs['model_kwargs'] = {'torch_dtype': torch.float16}

    logger.info(f"Loading embedding model {EMBEDDING_MODEL_ID} on {device}")
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_ID,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )

def get_vector_store():
    embedding_function = get_embedding_function()
    
    vector_store = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embedding_function,
        persist_directory=str(VECTOR_DB_DIR),
    )
    return vector_store

def get_document_count() -> int:
    """Returns the total number of documents in the Chroma collection."""
    try:
        vector_store = get_vector_store()
        return vector_store._collection.count()
    except Exception as e:
        logger.error(f"Failed to get document count: {e}")
        return 0

def add_documents_to_db(documents):
    if not documents:
        logger.warning("No documents provided to add_documents_to_db")
        return
    
    vector_store = get_vector_store()
    
    # Add documents in batches to avoid hitting limits if any
    batch_size = 100
    total_batches = (len(documents) - 1) // batch_size + 1
    
    with log_latency(logger, "add_documents_to_db"):
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i+batch_size]
            vector_store.add_documents(documents=batch)
            logger.info(f"Added batch {i//batch_size + 1}/{total_batches} ({len(batch)} docs)")
    
    # Persist is automatic in newer Chroma versions but good to be aware
