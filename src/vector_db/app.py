import os
import json
import time
from pinecone import Pinecone, ServerlessSpec
from langchain_community.embeddings import HuggingFaceEmbeddings
from dotenv import load_dotenv

load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
index_name = os.getenv("PINECONE_INDEX_NAME", "brain-light-chatbot")

# Initialize Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)

if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=1024,
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        )
    )

index = pc.Index(index_name)

# Initialize Embedding Model
print("Loading embedding model...")
embeddings_model = HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-large")

def flatten_metadata(meta):
    clean = {}
    for k, v in meta.items():
        if isinstance(v, dict):
            for kk, vv in v.items():
                clean[f"{k}_{kk}"] = str(vv)
        elif isinstance(v, list):
            clean[k] = ",".join([str(x) for x in v])
        else:
            clean[k] = str(v)
    return clean

def upsert_from_file(index, namespace, path, batch_size=50):
    print(f"Processing {path} for namespace '{namespace}'...")
    
    try:
        with open(path, "r", encoding="utf-8") as f:
            records = json.load(f)
    except FileNotFoundError:
        print(f"File not found: {path}")
        return

    vectors_to_upsert = []
    
    for i, rec in enumerate(records):
        content = rec.get("content", "")
        if not content:
            continue
            
        # Generate ID
        record_id = rec.get("id", f"{namespace}-{i}")
        
        # Prepare Metadata
        metadata = rec.get("metadata", {})
        if isinstance(metadata, dict):
            metadata = flatten_metadata(metadata)
        
        # Add content to metadata so we can retrieve it later
        metadata["text"] = content
        
        # Generate Embedding
        try:
            vector = embeddings_model.embed_query(content)
            
            vectors_to_upsert.append({
                "id": record_id,
                "values": vector,
                "metadata": metadata
            })
        except Exception as e:
            print(f"Error embedding record {i}: {e}")

        # Upsert in batches
        if len(vectors_to_upsert) >= batch_size:
            index.upsert(vectors=vectors_to_upsert, namespace=namespace)
            print(f"Upserted batch of {len(vectors_to_upsert)} to {namespace}")
            vectors_to_upsert = []

    # Upsert remaining
    if vectors_to_upsert:
        index.upsert(vectors=vectors_to_upsert, namespace=namespace)
        print(f"Upserted final batch of {len(vectors_to_upsert)} to {namespace}")

# ---------- Upsert all files ----------
# Adjust paths relative to src/vector_db/
base_path = "../../processed/cleaned_datasets"

upsert_from_file(index, "arabic", f"{base_path}/arabic_documents.json")
upsert_from_file(index, "arabic", f"{base_path}/arabic_qa_combined.json")
upsert_from_file(index, "english", f"{base_path}/english_documents.json")
upsert_from_file(index, "english", f"{base_path}/english_qa_combined.json")
upsert_from_file(index, "general", f"{base_path}/general_conversations.json")

print("Index stats:")
print(index.describe_index_stats())

