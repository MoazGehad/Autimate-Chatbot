# Running Evaluation on Lightning.ai Studio

This guide explains how to run the RAG evaluation pipeline on your real model and data using Lightning.ai Studio.

## Prerequisites

1.  **Lightning Studio**: An active Lightning Studio instance (preferably with GPU like T4 or A10G for faster inference, though CPU works for evaluation but is slow).
2.  **Code**: Your local code must be synced or uploaded to the Studio.
3.  **Data**: Your `data/` folder (PDFs) or a `test_dataset.json` must be present in the Studio.

## Step-by-Step Instructions

### 1. Upload Code and Data

If you haven't already, upload your project to the Studio.
- Ensure the `src/evaluation` folder is included.
- Ensure your `requirements.txt` is updated (it should include `ragas`, `datasets`, and `numpy<2`).
- If you want to generate a test set from your real data, upload your PDF files to `data/`.

### 2. Set Up Environment

Open a terminal in Lightning Studio and run:

```bash
# Update dependencies
pip install -r requirements.txt
```

This will install `ragas`, `datasets`, and ensure `numpy` is compatible.

### 3. Generate Test Data (Optional)

If you don't have a `test_dataset.json` yet, you can generate one from your uploaded documents:

```bash
# This uses the local LLM to generate Q&A pairs from your data
python src/evaluation/generate_testset.py
```

This will create `src/evaluation/generated_testset.json`.

### 4. Run Evaluation

You have two options for running the evaluation:

#### Option A: Standard Ragas Evaluation (Recommended if it works)
This uses the widespread `ragas` library metrics.

```bash
python src/evaluation/evaluate.py
```

*Note: If you encounter `numpy`/`pyarrow` errors (like on Windows), switch to Option B.*

#### Option B: Custom Evaluation (Robust Fallback)
This uses our custom script that mimics Ragas but has fewer dependencies.

```bash
python src/evaluation/custom_evaluate.py
```

### 5. View Results

The results will be saved as CSV files:
- `src/evaluation/evaluation_results.csv` (Option A)
- `src/evaluation/custom_evaluation_results.csv` (Option B)

You can download these files from the Studio file browser to your local machine for analysis.

## troubleshooting

- **Out of Memory (OOM)**: If the evaluation crashes, try reducing the `batch_size` in `src/vector_store.py` or restart the Studio with a larger GPU.
- **Dependency Errors**: If `ragas` fails to import, try running `pip install "numpy<2" ragas datasets --force-reinstall`.
