# Changelog

This document outlines the modifications made to the BrainLight RAG Chatbot repository to implement multi-provider LLM support, advanced retrieval techniques, and code quality improvements.

## 1. Multi-Provider LLM Support

*   **[NEW] `src/llm_provider.py`**: Created an abstraction layer (`BaseLLMProvider`) to manage LLM interactions.
    *   Implemented `QwenProvider` to wrap the existing local model logic, executing generation in a thread pool to avoid blocking the event loop.
    *   Implemented `GeminiProvider` using the `google-generativeai` SDK to provide an alternative, cost-effective LLM backend (defaults to `gemini-2.0-flash-lite`).
    *   Included language detection (Arabic vs. English) and dynamic system prompting within the providers.
*   **[MODIFIED] `src/generator.py`**: Cleaned up ~180 lines of legacy commented-out code. Refactored the `Generator` class to act as a thin facade over the injected `BaseLLMProvider`.

## 2. Advanced Retrieval Enhancements

*   **[MODIFIED] `src/retriever.py`**: Completely rewritten to support a multi-stage retrieval pipeline, controlled by feature flags:
    *   **Query Rewriting (Step 2a)**: Uses the active LLM provider to generate 2-3 semantically distinct sub-queries to improve recall.
    *   **Hybrid Retrieval (Step 2b)**: Combines dense search (ChromaDB) with sparse search (BM25Okapi). Results are fused using Reciprocal Rank Fusion (RRF) and deduplicated via chunk hashes.
    *   **Cross-Encoder Reranking (Step 2c)**: Optionally reranks the fused top candidates using `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1` (chosen for better Arabic support).
    *   **Contextual Compression (Step 2d)**: Splits retrieved chunks into sentences and filters out those with low cosine similarity to the query, reducing prompt token count.
*   **[MODIFIED] `src/ingestion.py`**:
    *   Removed legacy code.
    *   Added `generate_chunk_id` to create stable SHA-256 hashes for each chunk based on content and metadata (used for deduplication during hybrid retrieval).
    *   Added logic to build and pickle a `rank_bm25` index during the ingestion process if `BM25_ENABLED` is true.

## 3. Code Quality & Robustness

*   **[NEW] `src/logging_config.py`**: Introduced structured JSON logging. Provides a `get_logger` factory and a `log_latency` context manager to automatically track the execution time of critical operations (e.g., retrieval, generation, database additions).
*   **[MODIFIED] `src/config.py`**: Transitioned from hardcoded values to environment variables using `python-dotenv`. Added flags and parameters for all new features (`LLM_BACKEND`, `GEMINI_API_KEY`, `BM25_ENABLED`, `COMPRESSION_THRESHOLD`, etc.).
*   **[MODIFIED] `src/main.py`**:
    *   **Input Validation**: Replaced bare string passing with Pydantic V2 `ChatRequest` model. Added `field_validator`s to reject empty messages and messages over 500 characters, returning proper Arabic error messages via a custom `RequestValidationError` handler.
    *   **Health Endpoint**: Updated `/health` to return dynamic information, including the total ChromaDB document count, the active LLM backend, and the initialization status.
    *   **Asynchronous Execution**: Wrapped blocking I/O calls (like model inference and the ingestion process) in `asyncio.get_running_loop().run_in_executor()` to prevent the FastAPI event loop from stalling.
    *   **Streaming**: Added a `/chat/stream` endpoint specifically to support streaming Server-Sent Events (SSE) when using the Gemini backend.
*   **[MODIFIED] `src/vector_store.py`**: Added `get_document_count()` for the health check and wrapped `add_documents` with structured logging.
*   **[MODIFIED] `requirements.txt`**: Added necessary dependencies: `google-generativeai`, `rank-bm25`, `python-json-logger`, and `python-dotenv`.
*   **[NEW] `.env.example`**: Created a template environment file documenting all configuration options.
