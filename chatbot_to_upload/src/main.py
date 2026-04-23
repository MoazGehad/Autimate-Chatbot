import sys
import os
from pathlib import Path

# Ensure the root directory (chatbot_to_upload) is in sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import RedirectResponse, StreamingResponse
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator
from contextlib import asynccontextmanager
import asyncio
import json

from src.retriever import Retriever
from src.generator import Generator
from src.ingestion import load_documents, split_documents, build_and_save_bm25_index
from src.vector_store import add_documents_to_db, get_document_count
from src.config import DATA_DIR, LLM_BACKEND, BM25_ENABLED
from src.llm_provider import get_llm_provider
from src.logging_config import get_logger, log_latency

logger = get_logger(__name__)

# Global variables for RAG components
rag_components = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Load model and retriever
    logger.info("Loading RAG components...")
    try:
        provider = get_llm_provider(LLM_BACKEND)
        rag_components["llm_provider"] = provider
        rag_components["retriever"] = Retriever(llm_provider=provider)
        rag_components["generator"] = Generator(provider=provider)
        logger.info("RAG components loaded successfully.")
    except Exception as e:
        logger.error(f"Error loading RAG components: {e}")
        raise e
    
    yield
    
    # Shutdown: Clean up handled by torch/python mostly
    logger.info("Shutting down...")
    rag_components.clear()

app = FastAPI(title="BrainLight RAG Chatbot", lifespan=lifespan)

# Custom exception handler for validation errors to return Arabic messages
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    errors = exc.errors()
    error_msg = "خطأ في البيانات المدخلة"
    if errors and len(errors) > 0:
        error_msg = errors[0].get("msg", error_msg)
    return JSONResponse(
        status_code=422,
        content={"detail": error_msg},
    )

class ChatRequest(BaseModel):
    message: str = Field(..., description="The user query")

    @field_validator('message')
    @classmethod
    def validate_message(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("الرسالة لا يمكن أن تكون فارغة")
        if len(v) > 500:
            raise ValueError("الرسالة طويلة جداً (الحد الأقصى 500 حرف)")
        return v

class ChatResponse(BaseModel):
    response: str
    sources: list

@app.get("/")
def root():
    return RedirectResponse(url="/docs")

@app.get("/health")
def health_check():
    doc_count = get_document_count()
    status = "healthy" if "generator" in rag_components else "initializing"
    return {
        "status": status,
        "active_llm_backend": LLM_BACKEND,
        "chroma_document_count": doc_count,
        "components_loaded": list(rag_components.keys())
    }

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    if "generator" not in rag_components:
        raise HTTPException(status_code=503, detail="System initializing")
    
    query = request.message
    retriever = rag_components["retriever"]
    generator = rag_components["generator"]
    
    logger.info(f"Received chat request: {query}")
    
    with log_latency(logger, "chat_endpoint"):
        # 1. Retrieve
        retrieved_docs = await retriever.get_relevant_documents(query)
        
        # 2. Generate
        response = await generator.generate_response(query, retrieved_docs)
    
    return ChatResponse(
        response=response,
        sources=[doc.page_content[:200] for doc in retrieved_docs]
    )

@app.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    """Streaming endpoint (only supported for Gemini backend currently)."""
    if "generator" not in rag_components:
        raise HTTPException(status_code=503, detail="System initializing")
        
    provider = rag_components["llm_provider"]
    if LLM_BACKEND != "gemini":
        raise HTTPException(status_code=400, detail="Streaming is only supported with Gemini backend.")
        
    query = request.message
    retriever = rag_components["retriever"]
    
    logger.info(f"Received streaming chat request: {query}")
    
    retrieved_docs = await retriever.get_relevant_documents(query)
    context_str = "\n\n".join([doc.page_content for doc in retrieved_docs])
    
    async def event_generator():
        async for chunk in provider.generate_stream(query, context_str):
            yield f"data: {json.dumps({'text': chunk}, ensure_ascii=False)}\n\n"
        yield "data: [DONE]\n\n"
            
    return StreamingResponse(event_generator(), media_type="text/event-stream")


@app.post("/ingest")
async def trigger_ingestion():
    """Trigger ingestion manually."""
    logger.info("Starting ingestion process")
    try:
        loop = asyncio.get_running_loop()
        
        # Run heavy I/O in executor
        def run_ingestion():
            docs = load_documents(str(DATA_DIR))
            if not docs:
                return 0
            chunks = split_documents(docs)
            if BM25_ENABLED:
                build_and_save_bm25_index(chunks)
            add_documents_to_db(chunks)
            return len(chunks)
            
        chunks_added = await loop.run_in_executor(None, run_ingestion)
        
        logger.info(f"Ingestion complete. Added {chunks_added} chunks.")
        return {"status": "Ingestion complete", "chunks_added": chunks_added}
    except Exception as e:
        logger.error(f"Ingestion failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.main:app", host="0.0.0.0", port=7860, reload=True)
