from src.vector_store import get_vector_store
from src.config import TOP_K_RETRIEVAL

class Retriever:
    def __init__(self):
        self.vector_store = get_vector_store()
        
    def get_relevant_documents(self, query: str, k: int = TOP_K_RETRIEVAL):
        """
        Retrieve documents relevant to the query.
        Adds 'query: ' prefix as recommended for e5 models.
        """
        # E5 models expect 'query: ' prefix for asymmetric retrieval
        search_query = f"query: {query}"
        
        # Search
        # chroma stores raw text, so we return the documents directly
        # search_kwargs={"k": k}
        
        results = self.vector_store.similarity_search_with_score(search_query, k=k)
        
        # Results are (Document, score) tuples.
        # Check score metric. Chroma L2 distance: lower is better. 
        # Cosine: higher is better (if normalized). Default is L2.
        # But e5 is usually cosine similarity optimized. 
        # For now, return list of documents.
        
        return [doc for doc, score in results]

if __name__ == "__main__":
    retriever = Retriever()
    docs = retriever.get_relevant_documents("What is autism?  ") # What is autism?
    for doc in docs:
        print(f"Content: {doc.page_content[:500]}...")
        
