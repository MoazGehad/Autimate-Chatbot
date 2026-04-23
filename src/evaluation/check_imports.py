
print("Starting imports check...")
try:
    import ragas
    print(f"Ragas imported: {ragas.__version__}")
except ImportError as e:
    print(f"Ragas import failed: {e}")
except Exception as e:
    print(f"Ragas import error: {e}")

try:
    import datasets
    print(f"Datasets imported: {datasets.__version__}")
except ImportError as e:
    print(f"Datasets import failed: {e}")

try:
    from langchain_community.document_loaders import PyPDFLoader
    print("Langchain import success")
except ImportError as e:
    print(f"Langchain import failed: {e}")

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
try:
    from src.generator import get_llm_pipeline
    print("src.generator import success")
except Exception as e:
    print(f"src.generator import failed: {e}")

print("Imports check done.")
