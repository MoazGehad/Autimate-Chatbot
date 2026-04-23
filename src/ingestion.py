import os
import re
from typing import List, Dict, Any
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader, JSONLoader
from langchain.docstore.document import Document
from tqdm import tqdm

def clean_arabic_text(text: str) -> str:
    if not isinstance(text, str):
        return text
    # Remove /uniXXXX and uniXXXX patterns
    text = re.sub(r'/uni[0-9A-F]{4}', '', text, flags=re.IGNORECASE)
    text = re.sub(r'uni[0-9A-F]{4}', '', text, flags=re.IGNORECASE)
    # Remove Tatweel (Kashida)
    text = re.sub(r'[\u0640]', '', text)
    
    # Normalize Alifs
    text = re.sub(r'[أإآ]', 'ا', text)
    # Normalize Hamzas (optional, depending on strictness, but often helps retrieval)
    # text = re.sub(r'[ؤئ]', 'ء', text) 
    
    # Normalize Taa Marbuta to Haa (optional, but standard in some search)
    # text = re.sub(r'ة', 'h', text) # simple transliteration or keep as is. 
    # Better to keep 'ة' as 'ة' or 'ه' depending on strategy. 
    # Let's just fix common issues like double spaces and weird chars.
    
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

def load_documents(directory: str) -> List[Document]:
    documents = []
    if not os.path.exists(directory):
        print(f"Directory {directory} does not exist.")
        return []

    for root, _, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            try:
                if file.endswith(".pdf"):
                    # Using PyPDFLoader for now, consider PyMuPDF if Arabic extraction is poor
                    loader = PyPDFLoader(file_path)
                    docs = loader.load()
                    documents.extend(docs)
                elif file.endswith(".txt"):
                    loader = TextLoader(file_path, encoding="utf-8")
                    docs = loader.load()
                    documents.extend(docs)
                elif file.endswith(".json"):
                    # Load JSON documents if they follow a specific schema
                    # Assuming list of dicts with 'content' key
                    try:
                        loader = JSONLoader(
                            file_path=file_path,
                            jq_schema='.[]',
                            content_key='content',
                            text_content=False
                        )
                        docs = loader.load()
                        documents.extend(docs)
                    except Exception as json_err:
                        print(f"Error loading JSON {file_path}: {json_err}")
            except Exception as e:
                print(f"Error loading {file_path}: {e}")

    # Post-process cleaning
    for doc in documents:
        doc.page_content = clean_text(doc.page_content)
        # Add metadata if needed
        doc.metadata["source"] = doc.metadata.get("source", "")
        # Ensure source is string for compatibility
        if isinstance(doc.metadata.get("source"), (list, dict)):
             doc.metadata["source"] = str(doc.metadata["source"])
        
    return documents

def split_documents(documents: List[Document], chunk_size: int = 1000, chunk_overlap: int = 200) -> List[Document]:
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        # Optimized separators for Arabic to avoid improper cuts
        # Prioritize paragraph breaks, then sentences, then phrases
        separators=["\n\n", "\n", ".\s", "؟\s", "!\s", "،", " ", ""],
        length_function=len
    )
    return text_splitter.split_documents(documents)

if __name__ == "__main__":
    # Test run
    from src.config import DATA_DIR
    print(f"Loading from {DATA_DIR}...")
    docs = load_documents(str(DATA_DIR))
    print(f"Loaded {len(docs)} documents.")
    chunks = split_documents(docs)
    print(f"Split into {len(chunks)} chunks.")
