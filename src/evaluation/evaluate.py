
import os
import sys
import json
import pandas as pd
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevance,
    context_precision,
)
from ragas.llms import LangchainLLM
from ragas.embeddings import LangchainEmbeddings

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from src.generator import get_llm_pipeline
from src.retriever import Retriever
from src.vector_store import get_embedding_function

def evaluate_rag():
    print("Loading RAG components...")
    retriever = Retriever()
    llm_pipeline, tokenizer = get_llm_pipeline()
    embedding_function = get_embedding_function()
    
    # Wrap for Ragas
    ragas_llm = LangchainLLM(llm=llm_pipeline)
    ragas_embeddings = LangchainEmbeddings(embeddings=embedding_function)
    
    # Load test dataset
    print("Loading test dataset...")
    with open("src/evaluation/test_dataset.json", "r", encoding="utf-8") as f:
        test_data = json.load(f)
    
    questions = [item["question"] for item in test_data]
    ground_truths = [item["ground_truth"] for item in test_data]
    
    answers = []
    contexts = []
    
    print("Running RAG pipeline on test data...")
    # Initialize Generator once
    from src.generator import Generator
    # We can pass the already loaded llm_pipeline if we wanted, or just let Generator load it and use it.
    # Since we called get_llm_pipeline() above, let's pass it to Generator to avoid double loading.
    generator = Generator(llm=llm_pipeline, tokenizer=tokenizer)
    
    # Update ragas_llm to use the generator's llm just in case
    ragas_llm = LangchainLLM(llm=generator.llm)
    
    for query in questions:
        print(f"Processing: {query}")
        # Retrieval
        docs = retriever.get_relevant_documents(query)
        retrieved_contexts = [doc.page_content for doc in docs]
        contexts.append(retrieved_contexts)
        
        # Generation
        # Generator.generate_response returns a string response
        response = generator.generate_response(query, docs)
        answers.append(response)
        
    # Construct Dataset
    data = {
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": ground_truths
    }
    
    dataset = Dataset.from_dict(data)
    
    print("Starting Evaluation...")
    result = evaluate(
        dataset=dataset,
        metrics=[
            faithfulness,
            answer_relevance,
            context_precision,
        ],
        llm=ragas_llm,
        embeddings=ragas_embeddings,
    )
    
    print("\nEvaluation Results:")
    print(result)
    
    # Save results
    df = result.to_pandas()
    df.to_csv("src/evaluation/evaluation_results.csv", index=False)
    print("Results saved to src/evaluation/evaluation_results.csv")

if __name__ == "__main__":
    evaluate_rag()
