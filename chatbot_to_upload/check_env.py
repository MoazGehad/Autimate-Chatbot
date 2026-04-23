import sys
import traceback
print(f"Python: {sys.executable}")
try:
    from pinecone import Pinecone
    print("pinecone imported")
    pc = Pinecone(api_key="test")
    print("Pinecone client initialized")
except Exception:
    traceback.print_exc()

