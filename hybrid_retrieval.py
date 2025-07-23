"""
Hybrid retrieval combining dense (FAISS) and sparse (BM25) search.
"""
from typing import List, Dict, Any, Tuple, Optional
import logging
from dataclasses import dataclass
import math
from collections import Counter, defaultdict

from langchain_core.documents import Document
from rank_bm25 import BM25Okapi
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class RetrievalResult:
    """Container for retrieval results with metadata."""
    document: Document
    dense_score: float = 0.0
    sparse_score: float = 0.0
    combined_score: float = 0.0
    retrieval_source: str = "hybrid"
    rank: int = 0


class BM25Retriever:
    """BM25 sparse retrieval implementation."""
    
    def __init__(
        self,
        documents: List[Document],
        k1: float = 1.5,
        b: float = 0.75
    ):
        self.documents = documents
        self.k1 = k1
        self.b = b
        
        # Tokenize documents
        self.tokenized_docs = [
            self._tokenize(doc.page_content) for doc in documents
        ]
        
        # Initialize BM25
        self.bm25 = BM25Okapi(self.tokenized_docs, k1=k1, b=b)
        
        logger.info(f"Initialized BM25 with {len(documents)} documents")
    
    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization - can be enhanced with better NLP."""
        import re
        # Simple word tokenization with lowercasing
        tokens = re.findall(r'\b\w+\b', text.lower())
        return tokens
    
    def search(self, query: str, k: int = 10) -> List[Tuple[Document, float]]:
        """
        Search using BM25 scoring.
        
        Args:
            query: Search query
            k: Number of results to return
            
        Returns:
            List of (document, score) tuples
        """
        query_tokens = self._tokenize(query)
        scores = self.bm25.get_scores(query_tokens)
        
        # Get top-k results
        top_indices = np.argsort(scores)[::-1][:k]
        
        results = []
        for idx in top_indices:
            if scores[idx] > 0:  # Only include positive scores
                results.append((self.documents[idx], float(scores[idx])))
        
        logger.debug(f"BM25 found {len(results)} results")
        return results


class HybridRetriever:
    """Hybrid retrieval combining dense and sparse methods."""
    
    def __init__(
        self,
        dense_retriever,  # Your FAISS-based retriever
        sparse_retriever: BM25Retriever,
        alpha: float = 0.7,  # Weight for dense scores
        beta: float = 0.3,   # Weight for sparse scores
        rerank_top_k: int = 20
    ):
        self.dense_retriever = dense_retriever
        self.sparse_retriever = sparse_retriever
        self.alpha = alpha
        self.beta = beta
        self.rerank_top_k = rerank_top_k
        
        # Validate weights
        if abs(alpha + beta - 1.0) > 1e-6:
            logger.warning(f"Weights don't sum to 1.0: alpha={alpha}, beta={beta}")
    
    def search(
        self, 
        query: str, 
        k: int = 6,
        rerank: bool = True,
        min_dense_score: float = 0.0,
        min_sparse_score: float = 0.0
    ) -> List[RetrievalResult]:
        """
        Hybrid search combining dense and sparse retrieval.
        
        Args:
            query: Search query
            k: Final number of results to return
            rerank: Whether to rerank combined results
            min_dense_score: Minimum dense retrieval score
            min_sparse_score: Minimum sparse retrieval score
            
        Returns:
            List of ranked retrieval results
        """
        # Get results from both retrievers
        dense_results = self.dense_retriever.search(
            query, 
            k=self.rerank_top_k
        )
        sparse_results = self.sparse_retriever.search(
            query, 
            k=self.rerank_top_k
        )
        
        # Create document ID mapping
        doc_to_scores = defaultdict(lambda: {'dense': 0.0, 'sparse': 0.0, 'doc': None})
        
        # Process dense results
        for doc, score in dense_results:
            if score >= min_dense_score:
                doc_id = self._get_doc_id(doc)
                doc_to_scores[doc_id]['dense'] = score
                doc_to_scores[doc_id]['doc'] = doc
        
        # Process sparse results
        for doc, score in sparse_results:
            if score >= min_sparse_score:
                doc_id = self._get_doc_id(doc)
                doc_to_scores[doc_id]['sparse'] = score
                if doc_to_scores[doc_id]['doc'] is None:
                    doc_to_scores[doc_id]['doc'] = doc
        
        # Combine scores and create results
        results = []
        for doc_id, scores in doc_to_scores.items():
            if scores['doc'] is not None:
                # Normalize scores (simple min-max normalization)
                normalized_dense = self._normalize_score(
                    scores['dense'], 
                    [s['dense'] for s in doc_to_scores.values()]
                )
                normalized_sparse = self._normalize_score(
                    scores['sparse'], 
                    [s['sparse'] for s in doc_to_scores.values()]
                )
                
                # Combined score
                combined_score = (
                    self.alpha * normalized_dense + 
                    self.beta * normalized_sparse
                )
                
                result = RetrievalResult(
                    document=scores['doc'],
                    dense_score=scores['dense'],
                    sparse_score=scores['sparse'],
                    combined_score=combined_score,
                    retrieval_source="hybrid"
                )
                results.append(result)
        
        # Sort by combined score
        results.sort(key=lambda x: x.combined_score, reverse=True)
        
        # Add ranking information
        for i, result in enumerate(results):
            result.rank = i + 1
        
        # Apply reranking if enabled
        if rerank and len(results) > k:
            results = self._rerank_results(query, results[:k*2])
        
        logger.info(f"Hybrid search returned {len(results[:k])} results")
        return results[:k]
    
    def _get_doc_id(self, doc: Document) -> str:
        """Generate unique ID for document."""
        # Use content hash as ID
        import hashlib
        content_hash = hashlib.md5(doc.page_content.encode()).hexdigest()
        return content_hash
    
    def _normalize_score(self, score: float, all_scores: List[float]) -> float:
        """Normalize score using min-max normalization."""
        valid_scores = [s for s in all_scores if s > 0]
        if not valid_scores or len(valid_scores) == 1:
            return 1.0 if score > 0 else 0.0
        
        min_score = min(valid_scores)
        max_score = max(valid_scores)
        
        if max_score == min_score:
            return 1.0 if score > 0 else 0.0
        
        return (score - min_score) / (max_score - min_score)
    
    def _rerank_results(
        self, 
        query: str, 
        results: List[RetrievalResult]
    ) -> List[RetrievalResult]:
        """Rerank results using additional criteria."""
        # Simple reranking based on query term frequency
        query_terms = set(query.lower().split())
        
        for result in results:
            doc_terms = set(result.document.page_content.lower().split())
            term_overlap = len(query_terms.intersection(doc_terms))
            
            # Boost score based on term overlap
            overlap_bonus = term_overlap / len(query_terms) * 0.1
            result.combined_score += overlap_bonus
        
        # Re-sort
        results.sort(key=lambda x: x.combined_score, reverse=True)
        return results
    
    def get_retrieval_stats(self, results: List[RetrievalResult]) -> Dict[str, Any]:
        """Get statistics about retrieval results."""
        if not results:
            return {}
        
        dense_scores = [r.dense_score for r in results]
        sparse_scores = [r.sparse_score for r in results]
        combined_scores = [r.combined_score for r in results]
        
        return {
            'total_results': len(results),
            'dense_stats': {
                'mean': np.mean(dense_scores),
                'max': np.max(dense_scores),
                'min': np.min(dense_scores)
            },
            'sparse_stats': {
                'mean': np.mean(sparse_scores),
                'max': np.max(sparse_scores),
                'min': np.min(sparse_scores)
            },
            'combined_stats': {
                'mean': np.mean(combined_scores),
                'max': np.max(combined_scores),
                'min': np.min(combined_scores)
            }
        }


class QueryExpander:
    """Query expansion for improved retrieval."""
    
    def __init__(self, llm):
        self.llm = llm
    
    def expand_query(self, query: str) -> List[str]:
        """
        Expand query with synonyms and related terms.
        
        Args:
            query: Original query
            
        Returns:
            List of expanded query terms
        """
        expansion_prompt = f"""
        Given the query: "{query}"
        
        Generate 3-5 related terms or synonyms that could help find relevant information.
        Focus on technical terms, alternate phrasings, and domain-specific vocabulary.
        
        Return only the terms, one per line.
        
        Query: {query}
        Related terms:
        """
        
        try:
            response = self.llm.invoke(expansion_prompt)
            expanded_terms = [
                term.strip() 
                for term in response.content.split('\n') 
                if term.strip()
            ]
            
            # Combine original query with expanded terms
            all_terms = [query] + expanded_terms
            logger.debug(f"Expanded query to {len(all_terms)} terms")
            return all_terms
            
        except Exception as e:
            logger.error(f"Query expansion failed: {e}")
            return [query]


# Usage example
def main():
    """Example usage of hybrid retrieval."""
    from langchain_core.documents import Document
    
    # Sample documents
    docs = [
        Document(page_content="Machine learning algorithms for classification"),
        Document(page_content="Deep neural networks and backpropagation"),
        Document(page_content="Natural language processing with transformers"),
        Document(page_content="Computer vision and convolutional networks"),
    ]
    
    # Initialize sparse retriever
    sparse_retriever = BM25Retriever(docs)
    
    # Initialize hybrid retriever (assuming you have a dense retriever)
    # hybrid_retriever = HybridRetriever(dense_retriever, sparse_retriever)
    
    # Search
    query = "neural networks"
    sparse_results = sparse_retriever.search(query, k=3)
    
    print(f"BM25 results for '{query}':")
    for doc, score in sparse_results:
        print(f"Score: {score:.3f} - {doc.page_content[:50]}...")


if __name__ == "__main__":
    main()