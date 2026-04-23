
print("Checking datasets dependencies...")
try:
    import numpy
    print(f"numpy: {numpy.__version__}")
except ImportError as e:
    print(f"numpy failed: {e}")

try:
    import pandas
    print(f"pandas: {pandas.__version__}")
except ImportError as e:
    print(f"pandas failed: {e}")

try:
    import pyarrow
    print(f"pyarrow: {pyarrow.__version__}")
except ImportError as e:
    print(f"pyarrow failed: {e}")

try:
    import fsspec
    print(f"fsspec: {fsspec.__version__}")
except ImportError as e:
    print(f"fsspec failed: {e}")

try:
    import requests
    print(f"requests: {requests.__version__}")
except ImportError as e:
    print(f"requests failed: {e}")

print("Dependency check done.")
