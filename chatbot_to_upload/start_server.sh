#!/bin/bash

# Ensure we are in the project root
cd "$(dirname "$0")"

# Check GPU
if command -v nvidia-smi &> /dev/null
then
    echo "GPU detected:"
    nvidia-smi
else
    echo "WARNING: No GPU detected. Model loading might fail or be extremely slow."
fi

# Install dependencies if modules not found (basic check)
# In Lightning "Studio", envs persist, but good to ensure.
echo "Installing dependencies..."
pip install -r requirements.txt

# Start the server
echo "Starting FastAPI server on port 7860..."
python -m src.main
