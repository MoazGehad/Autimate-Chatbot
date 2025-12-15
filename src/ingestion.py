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
                    # Simple JSON loading assuming 'content' field or similar structure
                    # For complex JSON tailored to this project, we might need custom logic
                    # referencing the old json_reader.py if needed.
                    # For now, let's treat it generically or skip if it's metadata.
                     pass 
            except Exception as e:
                print(f"Error loading {file_path}: {e}")

    # Post-process cleaning
    for doc in documents:
        doc.page_content = clean_text(doc.page_content)
        # Add metadata if needed
        doc.metadata["source"] = doc.metadata.get("source", "")
        
    return documents

def split_documents(documents: List[Document], chunk_size: int = 1000, chunk_overlap: int = 200) -> List[Document]:
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        # Optimized separators for Arabic to avoid improper cuts
        separators=["\n\n", "\n", " ", "،", ".", ""],
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
