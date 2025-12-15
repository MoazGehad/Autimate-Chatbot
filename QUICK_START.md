# 🚀 Quick Start Guide

## Lightning.ai Deployment (Recommended)

### 1️⃣ Install Dependencies
```bash
pip install -r src/model/requirements.txt
```

### 2️⃣ Populate ChromaDB
```bash
cd src/vector_db
python app_chromadb.py
```

### 3️⃣ Test Chatbot (Optional)
```bash
cd ../model
python chatbot_chromadb.py
```

### 4️⃣ Start API Server
```bash
python api_chromadb.py
```

### 5️⃣ Expose Port 8000
- Open ports in Lightning Studio
- Copy public URL

### 6️⃣ Test API
```bash
# Health check
curl https://YOUR-URL/

# English test
curl -X POST https://YOUR-URL/chat \
  -H "Content-Type: application/json" \
  -d '{"query": "how to help autistic child"}'

# Arabic test
curl -X POST https://YOUR-URL/chat \
  -H "Content-Type: application/json" \
  -d '{"query": "كيف أساعد طفلي المصاب بالتوحد"}'
```

---

## 📁 Key Files

- **`src/model/chatbot_chromadb.py`** - Main chatbot with language detection
- **`src/model/api_chromadb.py`** - FastAPI server
- **`src/model/jias_chromadb.py`** - Model initialization
- **`src/vector_db/app_chromadb.py`** - Data ingestion

---

## ⚡ Language Detection

- **Arabic query** → Arabic response
- **English query** → English response
- Automatic detection based on character set

---

See `CHROMADB_DEPLOYMENT.md` for detailed instructions.
