"""
Quality tests for retrieval system using evaluation metrics.
"""

import pytest
from typing import List, Dict, Any
from unittest.mock import Mock, patch

from tests.utils.evaluation_metrics import EvaluationMetrics, evaluate_answer_quality
from tests.utils.test_corpus import create_test_corpus, GROUND_TRUTH_QUERIES, get_queries_by_type
from src.core.pipeline import RAGPipeline
from langchain_core.documents import Document


class TestRetrievalQuality:
    """Test suite for evaluating retrieval quality with metrics"""
    
    @pytest.fixture
    def quality_pipeline(self, test_config):
        """Pipeline configured for quality testing"""
        
        with patch('src.core.pipeline.create_llm_client') as mock_create_llm, \
             patch('src.core.pipeline.create_embedding_client') as mock_create_embed, \
             patch('src.core.pipeline.create_vector_store') as mock_create_store:
            
            # Setup deterministic mocks for quality testing
            mock_llm = Mock()
            mock_embed = Mock()
            mock_store = Mock()
            
            mock_create_llm.return_value = mock_llm
            mock_create_embed.return_value = mock_embed
            mock_create_store.return_value = mock_store
            
            # Create deterministic embeddings based on content
            test_docs = create_test_corpus()
            doc_embeddings = {}
            
            def create_embedding(text: str) -> List[float]:
                import hashlib
                import numpy as np
                text_hash = int(hashlib.md5(text.encode()).hexdigest()[:8], 16)
                np.random.seed(text_hash % (2**31))
                return np.random.normal(0, 1, 768).tolist()
            
            # Pre-compute embeddings for test documents
            for doc in test_docs:
                doc_embeddings[doc.metadata["document_id"]] = create_embedding(doc.page_content)
            
            mock_embed.embed_documents.side_effect = lambda texts: [create_embedding(text) for text in texts]
            mock_embed.embed_query.side_effect = lambda text: create_embedding(text)
            
            # Implement realistic similarity search
            stored_docs = []
            stored_embeddings = []
            
            def mock_add_documents(documents, embeddings):
                stored_docs.extend(documents)
                stored_embeddings.extend(embeddings)
            
            def mock_similarity_search(query_embedding, k=6):
                import numpy as np
                
                if not stored_docs or not stored_embeddings:
                    return []
                
                # Calculate similarity scores
                query_vec = np.array(query_embedding)
                similarities = []
                
                for i, doc_embedding in enumerate(stored_embeddings):
                    doc_vec = np.array(doc_embedding)
                    # Cosine similarity
                    similarity = np.dot(query_vec, doc_vec) / (np.linalg.norm(query_vec) * np.linalg.norm(doc_vec))
                    similarities.append((similarity, i))
                
                # Sort by similarity and return top k
                similarities.sort(reverse=True)
                return [stored_docs[i] for _, i in similarities[:k]]
            
            mock_store.add_documents.side_effect = mock_add_documents
            mock_store.similarity_search.side_effect = mock_similarity_search
            mock_store.get_document_count.return_value = len(stored_docs)
            
            # Load test corpus into pipeline
            pipeline = RAGPipeline(test_config)
            test_embeddings = [create_embedding(doc.page_content) for doc in test_docs]
            pipeline.vector_store.add_documents(test_docs, test_embeddings)
            
            return pipeline
    
    def test_precision_at_k_calculation(self):
        """Test precision@k metric calculation"""
        
        # Create test documents with known relevance
        retrieved_docs = [
            Document(page_content="Relevant doc 1", metadata={"document_id": "relevant_1"}),
            Document(page_content="Irrelevant doc", metadata={"document_id": "irrelevant_1"}),
            Document(page_content="Relevant doc 2", metadata={"document_id": "relevant_2"}),
            Document(page_content="Irrelevant doc 2", metadata={"document_id": "irrelevant_2"}),
        ]
        
        relevant_ids = ["relevant_1", "relevant_2"]
        
        # Test precision@2
        result = EvaluationMetrics.precision_at_k(retrieved_docs, relevant_ids, k=2)
        assert result.score == 0.5  # 1 relevant out of 2 retrieved
        assert result.details["relevant_retrieved"] == 1
        assert result.details["k"] == 2
        
        # Test precision@4  
        result = EvaluationMetrics.precision_at_k(retrieved_docs, relevant_ids, k=4)
        assert result.score == 0.5  # 2 relevant out of 4 retrieved
        assert result.details["relevant_retrieved"] == 2
    
    def test_recall_at_k_calculation(self):
        """Test recall@k metric calculation"""
        
        retrieved_docs = [
            Document(page_content="Relevant doc 1", metadata={"document_id": "relevant_1"}),
            Document(page_content="Irrelevant doc", metadata={"document_id": "irrelevant_1"}),
            Document(page_content="Relevant doc 2", metadata={"document_id": "relevant_2"}),
        ]
        
        relevant_ids = ["relevant_1", "relevant_2", "relevant_3"]  # 3 total relevant
        
        # Test recall@3 
        result = EvaluationMetrics.recall_at_k(retrieved_docs, relevant_ids, k=3)
        assert result.score == 2/3  # Found 2 out of 3 relevant docs
        assert result.details["relevant_found"] == 2
        assert result.details["total_relevant"] == 3
    
    def test_mrr_calculation(self):
        """Test Mean Reciprocal Rank calculation"""
        
        # First relevant document at position 2
        retrieved_docs = [
            Document(page_content="Irrelevant", metadata={"document_id": "irrelevant_1"}),
            Document(page_content="Relevant", metadata={"document_id": "relevant_1"}),
            Document(page_content="Also relevant", metadata={"document_id": "relevant_2"}),
        ]
        
        relevant_ids = ["relevant_1", "relevant_2"]
        
        result = EvaluationMetrics.mean_reciprocal_rank(retrieved_docs, relevant_ids)
        assert result.score == 0.5  # 1/2 (first relevant at rank 2)
        assert result.details["first_relevant_rank"] == 2
        
        # Test when no relevant documents found
        irrelevant_docs = [
            Document(page_content="Irrelevant", metadata={"document_id": "irrelevant_1"}),
        ]
        
        result = EvaluationMetrics.mean_reciprocal_rank(irrelevant_docs, relevant_ids)
        assert result.score == 0.0
        assert result.details["first_relevant_rank"] is None
    
    def test_factual_query_retrieval_precision(self, quality_pipeline):
        """Test retrieval precision for factual queries"""
        
        factual_queries = get_queries_by_type("factual")
        precision_scores = []
        
        for query in factual_queries[:3]:  # Test first 3 factual queries
            # Perform retrieval (mock query since we're testing retrieval only)
            query_embedding = quality_pipeline.embedding_client.embed_query(query.question)
            retrieved_docs = quality_pipeline.vector_store.similarity_search(query_embedding, k=5)
            
            # Calculate precision
            result = EvaluationMetrics.precision_at_k(retrieved_docs, query.relevant_doc_ids, k=5)
            precision_scores.append(result.score)
            
            # For factual queries, we expect reasonable precision
            assert result.score >= 0.2, f"Low precision for factual query: {query.question}"
        
        # Average precision should be reasonable
        avg_precision = sum(precision_scores) / len(precision_scores)
        assert avg_precision >= 0.3, f"Overall factual query precision too low: {avg_precision:.3f}"
    
    def test_technical_query_retrieval_quality(self, quality_pipeline):
        """Test retrieval quality for technical queries"""
        
        technical_queries = get_queries_by_type("technical")
        f1_scores = []
        
        for query in technical_queries[:2]:  # Test first 2 technical queries
            query_embedding = quality_pipeline.embedding_client.embed_query(query.question)
            retrieved_docs = quality_pipeline.vector_store.similarity_search(query_embedding, k=6)
            
            # Calculate F1@6
            result = EvaluationMetrics.f1_at_k(retrieved_docs, query.relevant_doc_ids, k=6)
            f1_scores.append(result.score)
            
            # Technical queries should still achieve some F1 score
            assert result.score >= 0.1, f"Very low F1 for technical query: {query.question}"
        
        # Check that we have some positive F1 scores
        positive_f1_count = sum(1 for score in f1_scores if score > 0.2)
        assert positive_f1_count >= 1, "No technical queries achieved reasonable F1 scores"
    
    def test_retrieval_consistency(self, quality_pipeline):
        """Test consistency of retrieval across similar queries"""
        
        # Test with similar questions about authors
        similar_queries = [
            "Who are the authors of the Transformer paper?",
            "Who wrote the Attention Is All You Need paper?",
            "What are the authors of the Transformer research?"
        ]
        
        retrieved_doc_sets = []
        
        for question in similar_queries:
            query_embedding = quality_pipeline.embedding_client.embed_query(question)
            retrieved_docs = quality_pipeline.vector_store.similarity_search(query_embedding, k=3)
            
            # Get document IDs of retrieved docs
            doc_ids = set(doc.metadata.get("document_id", "") for doc in retrieved_docs)
            retrieved_doc_sets.append(doc_ids)
        
        # Check overlap between retrieved document sets
        if len(retrieved_doc_sets) >= 2:
            overlap_12 = len(retrieved_doc_sets[0].intersection(retrieved_doc_sets[1]))
            total_unique = len(retrieved_doc_sets[0].union(retrieved_doc_sets[1]))
            
            if total_unique > 0:
                consistency_score = overlap_12 / min(len(retrieved_doc_sets[0]), len(retrieved_doc_sets[1]))
                # Similar queries should retrieve some overlapping documents
                assert consistency_score >= 0.3, f"Low retrieval consistency: {consistency_score:.3f}"
    
    def test_retrieval_relevance_filtering(self, quality_pipeline):
        """Test that retrieval filters out clearly irrelevant content"""
        
        # Query about Transformer architecture
        transformer_query = "How does the Transformer architecture work?"
        query_embedding = quality_pipeline.embedding_client.embed_query(transformer_query)
        retrieved_docs = quality_pipeline.vector_store.similarity_search(query_embedding, k=5)
        
        # Check that retrieved docs are somewhat relevant
        transformer_related_count = 0
        for doc in retrieved_docs:
            content = doc.page_content.lower()
            if any(term in content for term in ["transformer", "attention", "neural", "architecture"]):
                transformer_related_count += 1
        
        relevance_rate = transformer_related_count / len(retrieved_docs) if retrieved_docs else 0
        
        # At least some documents should be relevant to the query
        assert relevance_rate >= 0.4, f"Too many irrelevant documents retrieved: {relevance_rate:.3f}"
    
    def test_diverse_query_types_retrieval(self, quality_pipeline):
        """Test retrieval performance across different query types"""
        
        query_type_performance = {}
        
        for query_type in ["factual", "conceptual", "technical"]:
            type_queries = get_queries_by_type(query_type)
            if not type_queries:
                continue
                
            precision_scores = []
            
            for query in type_queries[:2]:  # Test 2 queries per type
                query_embedding = quality_pipeline.embedding_client.embed_query(query.question)
                retrieved_docs = quality_pipeline.vector_store.similarity_search(query_embedding, k=5)
                
                precision_result = EvaluationMetrics.precision_at_k(
                    retrieved_docs, query.relevant_doc_ids, k=5
                )
                precision_scores.append(precision_result.score)
            
            if precision_scores:
                avg_precision = sum(precision_scores) / len(precision_scores)
                query_type_performance[query_type] = avg_precision
        
        # Each query type should achieve some minimum performance
        for query_type, performance in query_type_performance.items():
            assert performance >= 0.1, f"Very low performance for {query_type} queries: {performance:.3f}"
        
        # At least one query type should perform reasonably well
        max_performance = max(query_type_performance.values()) if query_type_performance else 0
        assert max_performance >= 0.25, f"No query type achieved reasonable performance: {max_performance:.3f}"
    
    def test_retrieval_with_empty_results(self, quality_pipeline):
        """Test retrieval metrics when no documents are retrieved"""
        
        # Mock empty retrieval
        with patch.object(quality_pipeline.vector_store, 'similarity_search', return_value=[]):
            
            query = "What is machine learning?"
            query_embedding = quality_pipeline.embedding_client.embed_query(query)
            retrieved_docs = quality_pipeline.vector_store.similarity_search(query_embedding, k=5)
            
            relevant_ids = ["relevant_1", "relevant_2"]
            
            # Test metrics with empty retrieval
            precision_result = EvaluationMetrics.precision_at_k(retrieved_docs, relevant_ids, k=5)
            recall_result = EvaluationMetrics.recall_at_k(retrieved_docs, relevant_ids, k=5)
            mrr_result = EvaluationMetrics.mean_reciprocal_rank(retrieved_docs, relevant_ids)
            
            assert precision_result.score == 0.0
            assert recall_result.score == 0.0
            assert mrr_result.score == 0.0
            
            assert precision_result.details["retrieved"] == 0
            assert recall_result.details["relevant_found"] == 0
            assert mrr_result.details["first_relevant_rank"] is None
    
    def test_retrieval_performance_benchmarks(self, quality_pipeline):
        """Test retrieval against performance benchmarks"""
        
        # Define minimum acceptable performance thresholds
        performance_thresholds = {
            "min_precision_at_3": 0.15,   # At least 15% precision@3
            "min_recall_at_5": 0.20,      # At least 20% recall@5  
            "min_mrr": 0.25,              # At least 0.25 MRR
            "min_queries_above_threshold": 0.30  # 30% of queries should meet thresholds
        }
        
        all_results = []
        queries_above_threshold = 0
        
        # Test on a sample of queries
        test_queries = GROUND_TRUTH_QUERIES[:5]  # Test first 5 queries
        
        for query in test_queries:
            query_embedding = quality_pipeline.embedding_client.embed_query(query.question)
            retrieved_docs = quality_pipeline.vector_store.similarity_search(query_embedding, k=5)
            
            precision_3 = EvaluationMetrics.precision_at_k(retrieved_docs, query.relevant_doc_ids, k=3)
            recall_5 = EvaluationMetrics.recall_at_k(retrieved_docs, query.relevant_doc_ids, k=5)
            mrr = EvaluationMetrics.mean_reciprocal_rank(retrieved_docs, query.relevant_doc_ids)
            
            query_result = {
                "question": query.question,
                "precision_at_3": precision_3.score,
                "recall_at_5": recall_5.score,
                "mrr": mrr.score
            }
            all_results.append(query_result)
            
            # Check if query meets thresholds
            if (precision_3.score >= performance_thresholds["min_precision_at_3"] and
                recall_5.score >= performance_thresholds["min_recall_at_5"] and
                mrr.score >= performance_thresholds["min_mrr"]):
                queries_above_threshold += 1
        
        # Calculate overall performance
        if all_results:
            avg_precision = sum(r["precision_at_3"] for r in all_results) / len(all_results)
            avg_recall = sum(r["recall_at_5"] for r in all_results) / len(all_results)
            avg_mrr = sum(r["mrr"] for r in all_results) / len(all_results)
            
            threshold_rate = queries_above_threshold / len(all_results)
            
            # Log performance for debugging
            print(f"\nRetrieval Performance Summary:")
            print(f"Average Precision@3: {avg_precision:.3f}")
            print(f"Average Recall@5: {avg_recall:.3f}")
            print(f"Average MRR: {avg_mrr:.3f}")
            print(f"Queries above threshold: {threshold_rate:.3f}")
            
            # At least some queries should meet basic performance thresholds
            assert threshold_rate >= performance_thresholds["min_queries_above_threshold"], \
                f"Too few queries met performance thresholds: {threshold_rate:.3f}"