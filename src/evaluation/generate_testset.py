
import os
import sys
from ragas.testset.generator import TestsetGenerator
from ragas.testset.evolutions import simple, reasoning, multi_context
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from src.generator import get_llm_pipeline
from src.vector_store import get_embedding_function
from ragas.llms import LangchainLLM
from ragas.embeddings import LangchainEmbeddings
from src.config import DATA_DIR
from src.ingestion import load_documents

def generate_testset(num_questions=10):
    print("Loading LLM and Embeddings...")
    llm_pipeline, _ = get_llm_pipeline()
    embedding_function = get_embedding_function()
    
    ragas_llm = LangchainLLM(llm=llm_pipeline)
    ragas_embeddings = LangchainEmbeddings(embeddings=embedding_function)
    
    print("Loading documents...")
    # Load documents using ingestion logic
    # We might want to use larger chunks for question generation context
    documents = load_documents(str(DATA_DIR))
    
    # Ragas TestsetGenerator handles its own node parsing usually, but let's provide documents
    # The documents should be Langchain Documents
    
    print("Initializing Testset Generator...")
    generator = TestsetGenerator.from_langchain(
        generator_llm=ragas_llm,
        critic_llm=ragas_llm,
        embeddings=ragas_embeddings,
    )
    
    print(f"Generating {num_questions} test questions...")
    testset = generator.generate_with_langchain_docs(
        documents,
        test_size=num_questions,
        distributions={simple: 0.5, reasoning: 0.25, multi_context: 0.25},
    )
    
    # Save testset
    output_file = "src/evaluation/generated_testset.json"
    df = testset.to_pandas()
    df.to_json(output_file, orient="records", force_ascii=False, indent=2)
    print(f"Testset saved to {output_file}")
    
    # Also save as CSV for easier viewing
    csv_file = "src/evaluation/generated_testset.csv"
    df.to_csv(csv_file, index=False)
    print(f"Testset saved to {csv_file}")

if __name__ == "__main__":
    generate_testset()
