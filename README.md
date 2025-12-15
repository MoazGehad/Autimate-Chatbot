# BrainLight RAG Chatbot (T4 Optimized)

Optimized for **NVIDIA T4 (16GB VRAM)** using 4-bit quantization and efficient retrieval.

## Features
- **Model**: `Qwen/Qwen2.5-7B-Instruct` (4-bit NF4).
- **Embeddings**: `intfloat/multilingual-e5-large` (Float16).
- **Database**: ChromaDB (Local).
- **API**: FastAPI.

## Setup

1. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Ingest Data**
   Place your PDF/JSON/TXT files in `data/`.
   ```bash
   # Run the simple ingestion script (edit src/ingestion.py main block or use the API)
   python src/ingestion.py
   # OR via API
   # POST /ingest
   ```

3. **Run Chatbot**
   ```bash
   python src/main.py
   ```
   Access API at `http://localhost:8000/docs`.

## Structure
- `src/config.py`: Configuration.
- `src/ingestion.py`: Loading & Chunking.
- `src/vector_store.py`: ChromaDB setup.
- `src/retriever.py`: Semantic retrieval logic.
- `src/generator.py`: LLM loading & generation.
- `src/main.py`: API Server.

## Optimization Notes
- **VRAM Usage**: Optimized to fit within 16GB.
    - Qwen 7B (4-bit) ~5.5GB
    - E5-Large (Float16) ~1-2GB
    - Context & Overhead ~4-6GB
- **Language**: Specific prompts for Arabic/Egyptian dialect.
