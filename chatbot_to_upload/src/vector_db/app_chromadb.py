import os
import json
import chromadb
from chromadb.config import Settings
from langchain_community.embeddings import HuggingFaceEmbeddings
from dotenv import load_dotenv

load_dotenv()

# Initialize ChromaDB
chroma_client = chromadb.PersistentClient(
    path="../../chroma_db",
    settings=Settings(anonymized_telemetry=False)
)

# Initialize embedding model (High-quality for A100)
print("Loading embedding model (multilingual-e5-large - 1024 dimensions)...")
embeddings_model = HuggingFaceEmbeddings(
    model_name="intfloat/multilingual-e5-large",
    model_kwargs={'device': 'cuda'},
    encode_kwargs={'normalize_embeddings': True}
)
print("Embedding model loaded!")

# Get or create collections
arabic_collection = chroma_client.get_or_create_collection(
    name="arabic",
    metadata={"hnsw:space": "cosine"}
)

english_collection = chroma_client.get_or_create_collection(
    name="english",
    metadata={"hnsw:space": "cosine"}
)

def flatten_metadata(meta):
    """Convert all metadata values to strings for ChromaDB compatibility"""
    flat = {}
    for k, v in meta.items():
        if isinstance(v, (dict, list)):
            flat[k] = str(v)
        else:
            flat[k] = str(v) if v is not None else ""
    return flat

def upsert_to_chromadb(collection, path, batch_size=100):
    """Load data from JSON and upsert to ChromaDB collection"""
    print(f"Processing {path} for collection '{collection.name}'...")
    
    try:
        with open(path, "r", encoding="utf-8") as f:
            records = json.load(f)
    except FileNotFoundError:
        print(f"File not found: {path}")
        return

    ids = []
    documents = []
    metadatas = []
    embeddings = []
    
    for i, rec in enumerate(records):
        content = rec.get("content", "")
        if not content:
            continue
            
        # Generate ID
        record_id = rec.get("id", f"{collection.name}-{i}")
        
        # Prepare Metadata
        metadata = rec.get("metadata", {})
        if isinstance(metadata, dict):
            metadata = flatten_metadata(metadata)
        
        # Generate Embedding
        try:
            vector = embeddings_model.embed_query(content)
            
            ids.append(record_id)
            documents.append(content)
            metadatas.append(metadata)
            embeddings.append(vector)
            
        except Exception as e:
            print(f"Error embedding record {i}: {e}")

        # Upsert in batches
        if len(ids) >= batch_size:
            collection.upsert(
                ids=ids,
                documents=documents,
                metadatas=metadatas,
                embeddings=embeddings
            )
            print(f"Upserted batch of {len(ids)} to {collection.name}")
            ids, documents, metadatas, embeddings = [], [], [], []

    # Upsert remaining
    if ids:
        collection.upsert(
            ids=ids,
            documents=documents,
            metadatas=metadatas,
            embeddings=embeddings
        )
        print(f"Upserted final batch of {len(ids)} to {collection.name}")

# ---------- Upsert all files ----------
base_path = "../../processed/cleaned_datasets"

print("\n--- Populating Arabic Collection ---")
upsert_to_chromadb(arabic_collection, f"{base_path}/arabic_documents.json")
upsert_to_chromadb(arabic_collection, f"{base_path}/arabic_qa_combined.json")

print("\n--- Populating English Collection ---")
upsert_to_chromadb(english_collection, f"{base_path}/english_documents.json")
upsert_to_chromadb(english_collection, f"{base_path}/english_qa_combined.json")

print("\n--- Collections Stats ---")
print(f"Arabic collection: {arabic_collection.count()} documents")
print(f"English collection: {english_collection.count()} documents")

print("\nChromaDB population complete!")
