import os
import pickle
import re
import numpy as np
from typing import List, Dict, Tuple
from langchain_core.documents import Document

from src.vector_store import get_vector_store, get_embedding_function
from src.config import (
    TOP_K_RETRIEVAL, 
    QUERY_REWRITE_ENABLED, 
    BM25_ENABLED, 
    RERANKER_ENABLED,
    COMPRESSION_ENABLED,
    COMPRESSION_THRESHOLD,
    RERANKER_MODEL_ID,
    RERANKER_TOP_N,
    RRF_K,
    BM25_INDEX_PATH
)
from src.logging_config import get_logger, log_latency
from src.llm_provider import BaseLLMProvider

logger = get_logger(__name__)

class Retriever:
    def __init__(self, llm_provider: BaseLLMProvider = None):
        self.vector_store = get_vector_store()
        self.llm_provider = llm_provider
        self.embedding_function = get_embedding_function()
        
        # Load BM25 Index if enabled
        self.bm25_index = None
        self.bm25_chunks = None
        if BM25_ENABLED:
            try:
                if os.path.exists(BM25_INDEX_PATH):
                    with open(BM25_INDEX_PATH, 'rb') as f:
                        data = pickle.load(f)
                        self.bm25_index = data['bm25']
                        self.bm25_chunks = data['chunks']
                    logger.info("Loaded BM25 index successfully")
                else:
                    logger.warning(f"BM25 index not found at {BM25_INDEX_PATH}. Please run ingestion.")
            except Exception as e:
                logger.error(f"Error loading BM25 index: {e}")

        # Lazy load reranker
        self.reranker_model = None

    def _get_reranker(self):
        if self.reranker_model is None and RERANKER_ENABLED:
            from sentence_transformers import CrossEncoder
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info(f"Loading cross-encoder model: {RERANKER_MODEL_ID} on {device}")
            with log_latency(logger, "load_reranker"):
                self.reranker_model = CrossEncoder(RERANKER_MODEL_ID, device=device)
        return self.reranker_model

    async def get_relevant_documents(self, query: str, k: int = TOP_K_RETRIEVAL) -> List[Document]:
        """Orchestrates the advanced retrieval pipeline."""
        with log_latency(logger, "full_retrieval_pipeline"):
            
            # Step 2a: Query Rewriting
            queries = [query]
            if QUERY_REWRITE_ENABLED and self.llm_provider:
                logger.info("Executing Query Rewrite")
                try:
                    rewritten = await self.llm_provider.generate_query_rewrite(query)
                    if rewritten:
                        queries.extend(rewritten)
                        # Deduplicate queries just in case
                        queries = list(dict.fromkeys(queries))
                        logger.info(f"Expanded to {len(queries)} queries: {queries}")
                except Exception as e:
                    logger.error(f"Query rewriting failed: {e}")

            all_results: Dict[str, Document] = {}
            rrf_scores: Dict[str, float] = {}

            # Retrieve for all queries
            for q in queries:
                # E5 models expect 'query: ' prefix
                search_query = f"query: {q}"
                
                # Dense Retrieval
                dense_docs = []
                try:
                    results = self.vector_store.similarity_search_with_score(search_query, k=k)
                    dense_docs = [doc for doc, _ in results]
                except Exception as e:
                    logger.error(f"Dense retrieval failed for query '{q}': {e}")

                # Sparse Retrieval (Step 2b)
                sparse_docs = []
                if BM25_ENABLED and self.bm25_index:
                    try:
                        tokenized_query = q.split(" ")
                        # get top k documents
                        sparse_docs = self.bm25_index.get_top_n(tokenized_query, self.bm25_chunks, n=k)
                    except Exception as e:
                        logger.error(f"BM25 retrieval failed for query '{q}': {e}")

                # Merge and score (RRF)
                # Dense RRF
                for rank, doc in enumerate(dense_docs):
                    chunk_id = doc.metadata.get('chunk_id', doc.page_content)
                    if chunk_id not in all_results:
                        all_results[chunk_id] = doc
                    rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0.0) + (1.0 / (RRF_K + rank + 1))
                
                # Sparse RRF
                for rank, doc in enumerate(sparse_docs):
                    chunk_id = doc.metadata.get('chunk_id', doc.page_content)
                    if chunk_id not in all_results:
                        all_results[chunk_id] = doc
                    rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0.0) + (1.0 / (RRF_K + rank + 1))

            # Sort by RRF score
            sorted_candidates = [
                all_results[chunk_id] 
                for chunk_id, _ in sorted(rrf_scores.items(), key=lambda item: item[1], reverse=True)
            ]

            logger.info(f"Found {len(sorted_candidates)} unique candidates after fusion")

            # Step 2c: Cross-Encoder Reranking
            final_docs = sorted_candidates[:max(k, 10)] # Only rerank top N candidates to save time
            
            if RERANKER_ENABLED:
                reranker = self._get_reranker()
                if reranker:
                    logger.info("Executing Cross-Encoder Reranking")
                    try:
                        pairs = [[query, doc.page_content] for doc in final_docs]
                        scores = reranker.predict(pairs)
                        
                        # Sort by score
                        scored_docs = list(zip(final_docs, scores))
                        scored_docs.sort(key=lambda x: x[1], reverse=True)
                        
                        # Take top N
                        final_docs = [doc for doc, score in scored_docs[:RERANKER_TOP_N]]
                    except Exception as e:
                        logger.error(f"Reranking failed: {e}")
                        final_docs = final_docs[:RERANKER_TOP_N]
            else:
                final_docs = final_docs[:RERANKER_TOP_N]

            # Step 2d: Contextual Compression
            if COMPRESSION_ENABLED:
                logger.info("Executing Contextual Compression")
                compressed_docs = []
                query_emb = None
                
                try:
                    query_emb = np.array(self.embedding_function.embed_query(query))
                    
                    for doc in final_docs:
                        # Split by sentences (simple heuristic for Arabic/English)
                        sentences = re.split(r'(?<=[.!?؟])\s+', doc.page_content)
                        sentences = [s for s in sentences if s.strip()]
                        
                        if not sentences:
                            continue
                            
                        # Embed sentences
                        sentence_embs = self.embedding_function.embed_documents(sentences)
                        
                        kept_sentences = []
                        for i, emb in enumerate(sentence_embs):
                            # Cosine similarity
                            emb_np = np.array(emb)
                            sim = np.dot(query_emb, emb_np) / (np.linalg.norm(query_emb) * np.linalg.norm(emb_np))
                            if sim >= COMPRESSION_THRESHOLD:
                                kept_sentences.append(sentences[i])
                        
                        if kept_sentences:
                            compressed_content = " ".join(kept_sentences)
                            compressed_docs.append(Document(page_content=compressed_content, metadata=doc.metadata))
                    
                    if compressed_docs:
                        final_docs = compressed_docs
                except Exception as e:
                    logger.error(f"Compression failed: {e}")

            logger.info(f"Returning {len(final_docs)} final documents")
            return final_docs
