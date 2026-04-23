
import os
import sys
import json
# import pandas as pd # Removed to avoid dependency hell
from tqdm import tqdm
import re

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from src.generator import Generator, get_llm_pipeline
from src.retriever import Retriever

class RAGEvaluator:
    def __init__(self):
        print("Initializing RAG Evaluator...")
        self.llm, self.tokenizer = get_llm_pipeline()
        self.generator = Generator(llm=self.llm, tokenizer=self.tokenizer)
        self.retriever = Retriever()
        
    def evaluate_faithfulness(self, question, answer, context):
        """
        Rate if the answer is derived from the context.
        Returns: Score 0.0 to 1.0
        """
        system_prompt = """You are an expert evaluator for a RAG system.
Your task is to check if the generated answer is faithful to the retrieved context.
If the answer contains information not present in the context, it is a hallucination (Score 0).
If the answer is fully supported by the context, it is faithful (Score 1).
Return a JSON object with "score" (0.0 to 1.0) and "reason".
"""
        user_prompt = f"""Context:
{context}

Question: {question}
Answer: {answer}

Evaluate faithfulness."""

        response = self._call_llm(system_prompt, user_prompt)
        return self._parse_score(response)

    def evaluate_answer_relevance(self, question, answer):
        """
        Rate if the answer addresses the question.
        Returns: Score 0.0 to 1.0
        """
        system_prompt = """You are an expert evaluator.
Your task is to check if the answer gives a relevant response to the question.
Ignore if the answer is correct or not, just check if it addresses the prompt.
Return a JSON object with "score" (0.0 to 1.0) and "reason".
"""
        user_prompt = f"""Question: {question}
Answer: {answer}

Evaluate relevance."""

        response = self._call_llm(system_prompt, user_prompt)
        return self._parse_score(response)

    def evaluate_context_precision(self, question, context, ground_truth):
        """
        Rate if the context snippet contains the information needed to answer the question (based on ground truth).
        Returns: Score 0.0 to 1.0
        """
        system_prompt = """You are an expert evaluator.
Your task is to check if the retrieved context contains the information present in the ground truth answer.
Return a JSON object with "score" (0.0 to 1.0) and "reason".
"""
        user_prompt = f"""Question: {question}
Ground Truth: {ground_truth}
Retrieved Context: {context}

Evaluate context precision."""
        
        response = self._call_llm(system_prompt, user_prompt)
        return self._parse_score(response)

    def _call_llm(self, system, user):
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user}
        ]
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        # We use the pipeline directly to get raw text
        return self.llm.invoke(prompt)

    def _parse_score(self, response_text):
        try:
            # Try to find JSON block
            match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if match:
                data = json.loads(match.group(0))
                return data.get("score", 0.0), data.get("reason", "")
            else:
                # Fallback simple parsing if model didn't output JSON
                if "Score 1" in response_text or "1.0" in response_text:
                    return 1.0, response_text
                return 0.0, response_text
        except:
            return 0.0, "Failed to parse"

    def run_evaluation(self, test_file="src/evaluation/test_dataset.json"):
        print(f"Loading test data from {test_file}...")
        with open(test_file, "r", encoding="utf-8") as f:
            test_data = json.load(f)
            
        results = []
        
        print(f"Evaluating {len(test_data)} examples...")
        for item in tqdm(test_data):
            question = item["question"]
            ground_truth = item.get("ground_truth", "")
            
            # 1. Pipeline execution
            docs = self.retriever.get_relevant_documents(question)
            context_text = "\n\n".join([d.page_content for d in docs])
            generated_answer = self.generator.generate_response(question, docs)
            
            # 2. Evaluation
            faith_score, faith_reason = self.evaluate_faithfulness(question, generated_answer, context_text)
            rel_score, rel_reason = self.evaluate_answer_relevance(question, generated_answer)
            prec_score, prec_reason = self.evaluate_context_precision(question, context_text, ground_truth)
            
            results.append({
                "question": question,
                "answer": generated_answer,
                "context": context_text,
                "ground_truth": ground_truth,
                "faithfulness": faith_score,
                "faithfulness_reason": faith_reason,
                "answer_relevance": rel_score,
                "relevance_reason": rel_reason,
                "context_precision": prec_score,
                "precision_reason": prec_reason
            })
            
        # Save results using csv module
        import csv
        keys = results[0].keys()
        with open("src/evaluation/custom_evaluation_results.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(results)
        print("Results saved to src/evaluation/custom_evaluation_results.csv")
        
        # Calculate averages
        avg_faith = sum(r['faithfulness'] for r in results) / len(results) if results else 0
        avg_rel = sum(r['answer_relevance'] for r in results) / len(results) if results else 0
        avg_prec = sum(r['context_precision'] for r in results) / len(results) if results else 0
        
        print("\nAverage Scores:")
        print(f"Faithfulness: {avg_faith:.2f}")
        print(f"Answer Relevance: {avg_rel:.2f}")
        print(f"Context Precision: {avg_prec:.2f}")

if __name__ == "__main__":
    evaluator = RAGEvaluator()
    evaluator.run_evaluation()
