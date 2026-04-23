from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from contextlib import asynccontextmanager
from src.retriever import Retriever
from src.generator import Generator
from src.ingestion import load_documents, split_documents
from src.vector_store import add_documents_to_db
from src.config import DATA_DIR
import uvicorn
import os

# Global variables for RAG components
rag_components = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Load model and retriever
    print("Loading RAG components...")
    try:
        rag_components["retriever"] = Retriever()
        rag_components["generator"] = Generator()
        print("RAG components loaded successfully.")
    except Exception as e:
        print(f"Error loading RAG components: {e}")
        # We might continue without them for health check, or fail hard.
        # Failing hard is better for T4 singular focus.
        raise e
    
    yield
    
    # Shutdown: Clean up handled by torch/python mostly
    print("Shutting down...")
    rag_components.clear()

app = FastAPI(title="BrainLight RAG Chatbot", lifespan=lifespan)

class ChatRequest(BaseModel):
    message: str

class ChatResponse(BaseModel):
    response: str
    sources: list

from fastapi.responses import RedirectResponse

@app.get("/")
def root():
    return RedirectResponse(url="/docs")

@app.get("/health")
def health_check():
    return {"status": "healthy", "gpu": "T4-Optimized"}

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    if "generator" not in rag_components:
        raise HTTPException(status_code=503, detail="System initializing")
    
    query = request.message
    retriever = rag_components["retriever"]
    generator = rag_components["generator"]
    
    # 1. Retrieve
    # Arabic/Dialect handling is implicit in the multilingual model and prompt
    retrieved_docs = retriever.get_relevant_documents(query)
    
    # 2. Generate
    response = generator.generate_response(query, retrieved_docs)
    
    return ChatResponse(
        response=response,
        sources=[doc.page_content[:200] for doc in retrieved_docs]
    )

@app.post("/ingest")
async def trigger_ingestion():
    """Trigger ingestion manually."""
    try:
        docs = load_documents(str(DATA_DIR))
        chunks = split_documents(docs)
        add_documents_to_db(chunks)
        return {"status": "Ingestion complete", "chunks_added": len(chunks)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("src.main:app", host="0.0.0.0", port=7860, reload=True)
