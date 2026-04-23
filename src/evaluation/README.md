# Evaluation Pipeline

This directory contains scripts for evaluating the RAG chatbot using the `ragas` framework.

## Files

- `evaluate.py`: The main script to run the evaluation. It loads the RAG pipeline, runs it on the test dataset, and computes metrics like Faithfulness, Answer Relevance, and Context Precision.
- `generate_testset.py`: A script to generate a synthetic test dataset from your documents using the local LLM.
- `test_dataset.json`: A manually created or synthetic test dataset containing questions and ground truths.

## Usage

1. **Install Dependencies**:
   Ensure you have installed the required packages:
   ```bash
   pip install ragas datasets
   ```

2. **Run Evaluation**:
   ```bash
   python src/evaluation/evaluate.py
   ```
   This will output the evaluation results to the console and save them to `src/evaluation/evaluation_results.csv`.

3. **Generate Testset** (Optional):
   To generate a larger testset from your documents:
   ```bash
   python src/evaluation/generate_testset.py
   ```

## Metrics

- **Faithfulness**: Measures how well the answer is derived from the retrieved context. (Detects hallucinations)
- **Context Precision**: Measures if the retrieved chunks are relevant to the question.
- **Answer Relevance**: Measures how relevant the answer is to the question.
