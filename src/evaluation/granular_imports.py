
print("Checking granular imports...")
try:
    import huggingface_hub
    print(f"huggingface_hub: {huggingface_hub.__version__}")
except ImportError as e:
    print(f"huggingface_hub failed: {e}")

try:
    from huggingface_hub import hf_api
    print("huggingface_hub.hf_api imported")
except ImportError as e:
    print(f"huggingface_hub.hf_api failed: {e}")
except Exception as e:
    print(f"huggingface_hub.hf_api error: {e}")

try:
    import datasets
    print(f"datasets: {datasets.__version__}")
except ImportError as e:
    print(f"datasets failed: {e}")
except Exception as e:
    print(f"datasets error: {e}")

try:
    import ragas
    print(f"ragas: {ragas.__version__}")
except ImportError as e:
    print(f"ragas failed: {e}")
except Exception as e:
    print(f"ragas error: {e}")
