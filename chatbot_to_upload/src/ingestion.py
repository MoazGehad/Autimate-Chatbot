import sys
import os
from pathlib import Path

# Ensure the root directory (chatbot_to_upload) is in sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import os
import re
import hashlib
import pickle
from typing import List, Dict, Any
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_core.documents import Document
from src.logging_config import get_logger, log_latency
from src.config import BM25_ENABLED, BM25_INDEX_PATH

logger = get_logger(__name__)

def clean_arabic_text(text: str) -> str:
    if not isinstance(text, str):
        return text
    # Remove /uniXXXX and uniXXXX patterns
    text = re.sub(r'/uni[0-9A-F]{4}', '', text, flags=re.IGNORECASE)
    text = re.sub(r'uni[0-9A-F]{4}', '', text, flags=re.IGNORECASE)
    # Remove Tatweel (Kashida)
    text = re.sub(r'[\u0640]', '', text)
    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def clean_text(text: str) -> str:
    """General cleaning with specific Arabic handling if detected."""
    # Basic cleanup
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Check if likely Arabic (simple heuristic)
    if re.search(r'[\u0600-\u06FF]', text):
        text = clean_arabic_text(text)
        
    return text

def generate_chunk_id(content: str, metadata: dict) -> str:
    """Generates a stable hash ID for a chunk based on its content."""
    hash_input = f"{content}_{metadata.get('source', '')}".encode('utf-8')
    return hashlib.sha256(hash_input).hexdigest()

def load_documents(directory: str) -> List[Document]:
    documents = []
    if not os.path.exists(directory):
        logger.error(f"Directory {directory} does not exist.")
        return []

    logger.info(f"Loading documents from {directory}")
    with log_latency(logger, "load_documents"):
        for root, _, files in os.walk(directory):
            for file in files:
                file_path = os.path.join(root, file)
                try:
                    if file.endswith(".pdf"):
                        loader = PyPDFLoader(file_path)
                        docs = loader.load()
                        documents.extend(docs)
                    elif file.endswith(".txt"):
                        loader = TextLoader(file_path, encoding="utf-8")
                        docs = loader.load()
                        documents.extend(docs)
                    elif file.endswith(".json"):
                        # Load specific JSON format: [{"content": "...", "metadata": ...}]
                        try:
                            with open(file_path, 'r', encoding='utf-8') as f:
                                data = json.load(f)
                                if isinstance(data, list):
                                    for item in data:
                                        if "content" in item:
                                            content = item.get("content", "")
                                            # Handle mixed types if any
                                            if not isinstance(content, str):
                                                content = str(content)
                                                
                                            metadata = item.get("metadata", {})
                                            metadata["source"] = file_path
                                            
                                            doc = Document(page_content=content, metadata=metadata)
                                            documents.append(doc)
                        except json.JSONDecodeError:
                            logger.error(f"Error decoding JSON {file_path}")
                except Exception as e:
                    logger.error(f"Error loading {file_path}: {e}")

        # Post-process cleaning
        for doc in documents:
            doc.page_content = clean_text(doc.page_content)
            doc.metadata["source"] = doc.metadata.get("source", "")
            
    logger.info(f"Loaded {len(documents)} documents")
    return documents

def split_documents(documents: List[Document], chunk_size: int = 1000, chunk_overlap: int = 200) -> List[Document]:
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        # Optimized separators for Arabic to avoid improper cuts
        separators=["\n\n", "\n", " ", "،", ".", ""],
        length_function=len
    )
    
    with log_latency(logger, "split_documents"):
        chunks = text_splitter.split_documents(documents)
        
        # Add chunk IDs
        for chunk in chunks:
            chunk.metadata["chunk_id"] = generate_chunk_id(chunk.page_content, chunk.metadata)
            
    logger.info(f"Split documents into {len(chunks)} chunks")
    return chunks

def build_and_save_bm25_index(chunks: List[Document]):
    """Builds and saves a BM25 index from document chunks."""
    try:
        from rank_bm25 import BM25Okapi
        
        logger.info(f"Building BM25 index for {len(chunks)} chunks")
        with log_latency(logger, "build_bm25"):
            tokenized_corpus = [chunk.page_content.split(" ") for chunk in chunks]
            bm25 = BM25Okapi(tokenized_corpus)
            
            # Save the index and the chunks together so we can retrieve them later
            os.makedirs(os.path.dirname(BM25_INDEX_PATH), exist_ok=True)
            with open(BM25_INDEX_PATH, 'wb') as f:
                pickle.dump({'bm25': bm25, 'chunks': chunks}, f)
                
            logger.info(f"BM25 index saved to {BM25_INDEX_PATH}")
    except ImportError:
        logger.error("rank_bm25 is not installed. Run 'pip install rank-bm25'")
    except Exception as e:
        logger.error(f"Failed to build/save BM25 index: {e}")

if __name__ == "__main__":
    from src.config import DATA_DIR, ROOT_DIR
    
    target_dir = str(DATA_DIR)
    processed_dir = os.path.join(ROOT_DIR, "processed", "cleaned_datasets")
    
    if os.path.exists(processed_dir) and (not os.path.exists(target_dir) or not os.listdir(target_dir)):
        logger.info(f"Using processed/cleaned_datasets directory: {processed_dir}")
        target_dir = processed_dir
    
    docs = load_documents(target_dir)
    
    if docs:
        chunks = split_documents(docs)
        
        # Build BM25 Index if enabled
        if BM25_ENABLED:
            build_and_save_bm25_index(chunks)
            
        # Ingest into ChromaDB
        from src.vector_store import add_documents_to_db
        add_documents_to_db(chunks)
    else:
        logger.warning("No documents found. Please add files to 'data/' or 'processed/cleaned_datasets'.")
