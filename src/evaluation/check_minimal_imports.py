
print("Checking minimal imports...")
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
except Exception as e:
    print(f"Datasets import error: {e}")

print("Minimal imports check done.")
