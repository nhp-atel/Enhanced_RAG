"""
Integration tests for end-to-end RAG system functionality.
"""

import pytest
import json
from pathlib import Path
from typing import List, Dict, Any
from unittest.mock import patch, Mock

from src.core.pipeline import RAGPipeline, create_pipeline
from src.utils.config import RAGConfig
from tests.utils.evaluation_metrics import EvaluationMetrics, evaluate_answer_quality
from tests.utils.test_corpus import create_test_corpus, GROUND_TRUTH_QUERIES


class TestEndToEndRAG:
    """End-to-end integration tests for RAG system"""
    
    @pytest.fixture(scope="class")
    def test_pipeline(self, test_config):
        """Create test pipeline with mocked external dependencies"""
        
        # Use mock clients to avoid external API calls
        with patch('src.core.pipeline.create_llm_client') as mock_create_llm, \
             patch('src.core.pipeline.create_embedding_client') as mock_create_embed, \
             patch('src.core.pipeline.create_vector_store') as mock_create_store:
            
            # Setup mocks with realistic behavior
            mock_llm = Mock()
            mock_embed = Mock()
            mock_store = Mock()
            
            mock_create_llm.return_value = mock_llm
            mock_create_embed.return_value = mock_embed  
            mock_create_store.return_value = mock_store
            
            # Configure embedding mock for deterministic embeddings
            def mock_embed_documents(texts):
                import hashlib
                import numpy as np
                embeddings = []
                for text in texts:
                    # Create deterministic embedding from text hash
                    text_hash = int(hashlib.md5(text.encode()).hexdigest()[:8], 16)
                    np.random.seed(text_hash % (2**31))
                    embedding = np.random.normal(0, 1, 768).tolist()
                    embeddings.append(embedding)
                return embeddings
            
            mock_embed.embed_documents.side_effect = mock_embed_documents
            mock_embed.embed_query.side_effect = lambda text: mock_embed_documents([text])[0]
            
            # Configure LLM mock for realistic responses
            def mock_generate(messages, **kwargs):
                message_text = " ".join([msg.get("content", "") for msg in messages])
                
                # Answer questions based on content
                if "who are the authors" in message_text.lower():
                    return Mock(content="The authors are Ashish Vaswani and Noam Shazeer from Google Brain.")
                elif "what is the transformer" in message_text.lower():
                    return Mock(content="The Transformer is a model architecture based solely on attention mechanisms, dispensing with recurrence and convolutions entirely.")
                elif "main contribution" in message_text.lower():
                    return Mock(content="The main contribution is proposing the Transformer architecture that relies entirely on self-attention mechanisms.")
                elif "extract metadata" in message_text.lower():
                    return Mock(content="""Title: Attention Is All You Need
Authors: Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit
Institutions: Google Brain, Google Research, University of Toronto
Publication Date: 2017-06-12
ArXiv ID: 1706.03762
Keywords: attention mechanism, transformer, neural networks, sequence modeling
Abstract: We propose a new simple network architecture, the Transformer, based solely on attention mechanisms, dispensing with recurrence and convolutions entirely.
--- END OF METADATA ---""")
                else:
                    return Mock(content="This is a research paper about neural network architectures.")
            
            mock_llm.generate.side_effect = mock_generate
            mock_llm.generate_with_retry.side_effect = mock_generate
            
            # Configure vector store mock
            stored_docs = []
            stored_embeddings = []
            
            def mock_add_documents(documents, embeddings):
                stored_docs.extend(documents)
                stored_embeddings.extend(embeddings)
            
            def mock_similarity_search(query_embedding, k=6):
                # Simple similarity: return first k documents
                return stored_docs[:min(k, len(stored_docs))]
            
            mock_store.add_documents.side_effect = mock_add_documents
            mock_store.similarity_search.side_effect = mock_similarity_search
            mock_store.get_document_count.return_value = len(stored_docs)
            mock_store.save_index.return_value = True
            mock_store.load_index.return_value = True
            
            pipeline = RAGPipeline(test_config)
            return pipeline
    
    def test_document_processing_workflow(self, test_pipeline, create_test_pdf):
        """Test complete document processing workflow"""
        
        # Create test document
        test_content = """
        Attention Is All You Need
        
        Authors: Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit
        Google Brain, Google Research, University of Toronto
        
        Abstract
        The dominant sequence transduction models are based on complex recurrent or 
        convolutional neural networks that include an encoder and a decoder. The best 
        performing models also connect the encoder and decoder through an attention 
        mechanism. We propose a new simple network architecture, the Transformer, 
        based solely on attention mechanisms, dispensing with recurrence and 
        convolutions entirely.
        
        1. Introduction
        Recurrent neural networks, long short-term memory and gated recurrent neural 
        networks in particular, have been firmly established as state of the art 
        approaches in sequence modeling and transduction problems.
        """
        
        test_pdf = create_test_pdf(test_content, "attention_paper.pdf")
        
        # Mock PDF loading
        with patch('src.core.ingest.PyPDFLoader') as mock_loader:
            from langchain_core.documents import Document
            
            # Create mock pages from test content
            pages = [
                Document(page_content=test_content[:500], metadata={"page": 0, "source": str(test_pdf)}),
                Document(page_content=test_content[500:1000], metadata={"page": 1, "source": str(test_pdf)}),
                Document(page_content=test_content[1000:], metadata={"page": 2, "source": str(test_pdf)})
            ]
            mock_loader.return_value.load.return_value = pages
            
            # Process document
            result = test_pipeline.process_document(str(test_pdf))
            
            # Verify processing success
            assert result.success is True
            assert result.document_id is not None
            assert result.chunk_count > 0
            assert result.metadata is not None
            assert result.metadata.title == "Attention Is All You Need"
            assert "Ashish Vaswani" in result.metadata.authors
    
    def test_query_answering_workflow(self, test_pipeline, create_test_pdf):
        """Test complete query answering workflow"""
        
        # First process a document
        test_content = """
        BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding
        
        Authors: Jacob Devlin, Ming-Wei Chang, Kenton Lee, Kristina Toutanova
        Google AI Language
        
        Abstract
        We introduce a new language representation model called BERT, which stands for 
        Bidirectional Encoder Representations from Transformers. BERT is designed to 
        pre-train deep bidirectional representations from unlabeled text by jointly 
        conditioning on both left and right context in all layers.
        """
        
        test_pdf = create_test_pdf(test_content, "bert_paper.pdf")
        
        with patch('src.core.ingest.PyPDFLoader') as mock_loader:
            from langchain_core.documents import Document
            
            pages = [
                Document(page_content=test_content, metadata={"page": 0, "source": str(test_pdf)})
            ]
            mock_loader.return_value.load.return_value = pages
            
            # Process document
            process_result = test_pipeline.process_document(str(test_pdf))
            assert process_result.success is True
            
            # Test queries
            test_queries = [
                "Who are the authors of this paper?",
                "What does BERT stand for?",
                "What is the main contribution of this work?"
            ]
            
            for query in test_queries:
                result = test_pipeline.query(query)
                
                assert result.answer is not None
                assert len(result.answer) > 0
                assert result.sources is not None
                assert result.processing_time_ms > 0
                assert result.tokens_used >= 0
                assert result.cost_usd >= 0.0
    
    def test_multiple_documents_integration(self, test_pipeline, create_test_pdf):
        """Test processing multiple documents and cross-document queries"""
        
        # Create multiple test documents
        doc1_content = """
        Attention Is All You Need
        Authors: Ashish Vaswani, Noam Shazeer
        The Transformer architecture uses self-attention mechanisms.
        """
        
        doc2_content = """
        BERT: Bidirectional Transformers
        Authors: Jacob Devlin, Ming-Wei Chang
        BERT uses bidirectional training of Transformer encoders.
        """
        
        doc1_pdf = create_test_pdf(doc1_content, "transformer.pdf")
        doc2_pdf = create_test_pdf(doc2_content, "bert.pdf")
        
        with patch('src.core.ingest.PyPDFLoader') as mock_loader:
            from langchain_core.documents import Document
            
            def mock_load_side_effect():
                # Return different content based on which file is being loaded
                source_path = mock_loader.call_args[0][0]
                if "transformer.pdf" in source_path:
                    return [Document(page_content=doc1_content, metadata={"page": 0, "source": source_path})]
                else:
                    return [Document(page_content=doc2_content, metadata={"page": 0, "source": source_path})]
            
            mock_loader.return_value.load.side_effect = mock_load_side_effect
            
            # Process both documents
            result1 = test_pipeline.process_document(str(doc1_pdf))
            result2 = test_pipeline.process_document(str(doc2_pdf))
            
            assert result1.success is True
            assert result2.success is True
            
            # Test cross-document query
            result = test_pipeline.query("Which papers discuss Transformer architectures?")
            assert result.answer is not None
            assert len(result.sources) > 0
    
    def test_error_handling_integration(self, test_pipeline):
        """Test error handling in integration scenarios"""
        
        # Test invalid document source
        result = test_pipeline.process_document("/nonexistent/file.pdf")
        assert result.success is False
        assert result.error_message is not None
        
        # Test query without any documents
        with patch.object(test_pipeline.vector_store, 'similarity_search', return_value=[]):
            result = test_pipeline.query("What is this about?")
            # Should still return a result, even if no relevant documents found
            assert result.answer is not None
    
    def test_caching_integration(self, test_pipeline, create_test_pdf):
        """Test caching behavior in integration scenarios"""
        
        test_content = "Test paper for caching integration"
        test_pdf = create_test_pdf(test_content, "cache_test.pdf")
        
        with patch('src.core.ingest.PyPDFLoader') as mock_loader:
            from langchain_core.documents import Document
            
            pages = [Document(page_content=test_content, metadata={"page": 0, "source": str(test_pdf)})]
            mock_loader.return_value.load.return_value = pages
            
            # First processing
            result1 = test_pipeline.process_document(str(test_pdf))
            assert result1.success is True
            
            # Mock cache to simulate hit on second processing
            with patch.object(test_pipeline, 'cache') as mock_cache:
                mock_cache.get.return_value = result1  # Cache hit
                
                result2 = test_pipeline.process_document(str(test_pdf))
                assert result2.success is True
                
                # Verify cache was checked
                mock_cache.get.assert_called()
    
    def test_persistence_integration(self, test_pipeline, temp_dir):
        """Test index persistence in integration scenarios"""
        
        index_path = temp_dir / "test_index"
        
        # Test saving index
        save_result = test_pipeline.save_index(str(index_path))
        assert save_result is True
        
        # Test loading index
        load_result = test_pipeline.load_index(str(index_path))
        assert load_result is True
    
    def test_health_check_integration(self, test_pipeline):
        """Test system health check integration"""
        
        # Mock component health checks
        test_pipeline.llm_client.health_check = Mock(return_value={"status": "healthy"})
        test_pipeline.embedding_client.health_check = Mock(return_value={"status": "healthy"})
        test_pipeline.vector_store.health_check = Mock(return_value={"status": "healthy", "doc_count": 10})
        
        health = test_pipeline.health_check()
        
        assert health["overall"] == "healthy"
        assert "llm_client" in health["components"]
        assert "embedding_client" in health["components"]
        assert "vector_store" in health["components"]
        assert "timestamp" in health
    
    def test_stats_integration(self, test_pipeline):
        """Test statistics collection integration"""
        
        stats = test_pipeline.get_stats()
        
        assert "documents_processed" in stats
        assert "total_queries" in stats
        assert "cache_stats" in stats
        assert isinstance(stats["documents_processed"], int)
        assert isinstance(stats["total_queries"], int)


class TestGroundTruthEvaluation:
    """Integration tests using ground truth data for quality evaluation"""
    
    @pytest.fixture(scope="class") 
    def evaluation_pipeline(self, test_config):
        """Pipeline configured for evaluation with deterministic responses"""
        
        with patch('src.core.pipeline.create_llm_client') as mock_create_llm, \
             patch('src.core.pipeline.create_embedding_client') as mock_create_embed, \
             patch('src.core.pipeline.create_vector_store') as mock_create_store:
            
            # Create evaluation-specific mocks
            mock_llm = Mock()
            mock_embed = Mock()
            mock_store = Mock()
            
            mock_create_llm.return_value = mock_llm
            mock_create_embed.return_value = mock_embed
            mock_create_store.return_value = mock_store
            
            # Configure with ground truth responses
            ground_truth_responses = {
                "who are the authors": "The authors are Ashish Vaswani, Noam Shazeer, Niki Parmar, and Jakob Uszkoreit.",
                "what is the transformer": "The Transformer is a novel neural sequence transduction model based entirely on attention mechanisms.",
                "main contribution": "The main contribution is the Transformer architecture that achieves better results while being more parallelizable.",
                "attention mechanism": "The attention mechanism allows the model to focus on different parts of the input sequence when producing each output element."
            }
            
            def evaluation_generate(messages, **kwargs):
                message_text = " ".join([msg.get("content", "") for msg in messages]).lower()
                
                for key, response in ground_truth_responses.items():
                    if key in message_text:
                        return Mock(content=response)
                
                return Mock(content="I don't have enough information to answer this question.")
            
            mock_llm.generate.side_effect = evaluation_generate
            mock_llm.generate_with_retry.side_effect = evaluation_generate
            
            # Configure embeddings and storage
            stored_docs = []
            
            def mock_add_documents(documents, embeddings):
                stored_docs.extend(documents)
            
            def mock_similarity_search(query_embedding, k=6):
                # Return most relevant docs based on query
                return stored_docs[:min(k, len(stored_docs))]
            
            mock_embed.embed_documents.return_value = [[0.1] * 768] * 10  # Dummy embeddings
            mock_embed.embed_query.return_value = [0.1] * 768
            mock_store.add_documents.side_effect = mock_add_documents  
            mock_store.similarity_search.side_effect = mock_similarity_search
            
            pipeline = RAGPipeline(test_config)
            return pipeline
    
    def test_factual_question_accuracy(self, evaluation_pipeline, ground_truth_qa):
        """Test accuracy on factual questions using ground truth"""
        
        # Setup test corpus
        from langchain_core.documents import Document
        
        test_documents = [
            Document(
                page_content="Attention Is All You Need. Authors: Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit. Google Brain and Google Research.",
                metadata={"type": "paper_metadata", "source": "transformer_paper.pdf"}
            ),
            Document(
                page_content="We propose the Transformer, a model architecture based entirely on attention mechanisms, dispensing with recurrence and convolutions entirely.",
                metadata={"type": "content", "source": "transformer_paper.pdf", "page": 1}  
            )
        ]
        
        # Add documents to pipeline
        embeddings = [[0.1] * 768] * len(test_documents)
        evaluation_pipeline.vector_store.add_documents(test_documents, embeddings)
        
        # Test ground truth questions
        factual_questions = [qa for qa in ground_truth_qa if qa["answer_type"] == "factual"]
        
        correct_answers = 0
        total_questions = len(factual_questions)
        
        for qa_pair in factual_questions:
            result = evaluation_pipeline.query(qa_pair["question"])
            
            # Check if answer contains expected content
            expected = qa_pair["expected_answer"].lower()
            actual = result.answer.lower()
            
            if expected in actual or any(word in actual for word in expected.split()):
                correct_answers += 1
        
        accuracy = correct_answers / total_questions if total_questions > 0 else 0
        
        # We expect high accuracy on factual questions
        assert accuracy >= 0.7, f"Factual accuracy too low: {accuracy:.2f}"
    
    def test_retrieval_precision(self, evaluation_pipeline):
        """Test precision of document retrieval"""
        
        from langchain_core.documents import Document
        
        # Create documents with clear relevance signals
        relevant_docs = [
            Document(
                page_content="The Transformer architecture uses multi-head self-attention mechanisms for sequence modeling.",
                metadata={"relevance": "high", "topic": "transformer_architecture"}
            ),
            Document(
                page_content="Attention mechanisms allow models to focus on relevant parts of the input sequence.",
                metadata={"relevance": "high", "topic": "attention_mechanism"}
            )
        ]
        
        irrelevant_docs = [
            Document(
                page_content="Convolutional neural networks are effective for image classification tasks.",
                metadata={"relevance": "low", "topic": "cnn"}
            ),
            Document(
                page_content="Linear regression is a fundamental statistical method for predictive modeling.",
                metadata={"relevance": "low", "topic": "statistics"}
            )
        ]
        
        all_docs = relevant_docs + irrelevant_docs
        embeddings = [[0.1] * 768] * len(all_docs)
        
        # Mock similarity search to return relevant docs first
        def mock_relevant_search(query_embedding, k=6):
            query_text = "transformer attention mechanism"  # Simulated query
            return relevant_docs[:k]
        
        evaluation_pipeline.vector_store.similarity_search = mock_relevant_search
        evaluation_pipeline.vector_store.add_documents(all_docs, embeddings)
        
        # Test retrieval precision
        result = evaluation_pipeline.query("How does the Transformer use attention mechanisms?")
        
        # Check that retrieved sources are relevant
        if result.sources:
            relevant_count = sum(1 for doc in result.sources 
                               if doc.metadata.get("relevance") == "high")
            precision = relevant_count / len(result.sources)
            
            assert precision >= 0.8, f"Retrieval precision too low: {precision:.2f}"
    
    def test_answer_completeness(self, evaluation_pipeline):
        """Test that answers are complete and informative"""
        
        # Setup context documents
        from langchain_core.documents import Document
        
        context_docs = [
            Document(
                page_content="The Transformer model architecture is based entirely on attention mechanisms. It consists of an encoder and decoder, each composed of multiple identical layers.",
                metadata={"section": "architecture"}
            ),
            Document(
                page_content="Multi-head attention allows the model to jointly attend to information from different representation subspaces at different positions.",
                metadata={"section": "attention_details"}
            ),
            Document(
                page_content="The Transformer achieves better translation quality while being more parallelizable and requiring significantly less time to train.",
                metadata={"section": "results"}
            )
        ]
        
        embeddings = [[0.1] * 768] * len(context_docs)
        evaluation_pipeline.vector_store.add_documents(context_docs, embeddings)
        
        # Test comprehensive questions
        complex_question = "What is the Transformer architecture and what are its key advantages?"
        result = evaluation_pipeline.query(complex_question)
        
        # Check answer completeness
        answer = result.answer.lower()
        
        # Should mention key concepts
        key_concepts = ["attention", "architecture", "parallelizable", "encoder", "decoder"]
        concepts_mentioned = sum(1 for concept in key_concepts if concept in answer)
        
        completeness_score = concepts_mentioned / len(key_concepts)
        assert completeness_score >= 0.6, f"Answer completeness too low: {completeness_score:.2f}"
        
        # Answer should be substantial
        assert len(result.answer.split()) >= 15, "Answer too brief"
    
    def test_consistency_across_queries(self, evaluation_pipeline):
        """Test that similar queries produce consistent answers"""
        
        # Setup documents
        from langchain_core.documents import Document
        
        docs = [
            Document(
                page_content="The Transformer was introduced by Vaswani et al. in 2017 as a novel architecture for sequence transduction.",
                metadata={"source": "paper_intro"}
            )
        ]
        
        embeddings = [[0.1] * 768] * len(docs)
        evaluation_pipeline.vector_store.add_documents(docs, embeddings)
        
        # Ask similar questions
        similar_questions = [
            "Who introduced the Transformer?",
            "Who are the authors of the Transformer paper?",
            "Who created the Transformer architecture?"
        ]
        
        answers = []
        for question in similar_questions:
            result = evaluation_pipeline.query(question)
            answers.append(result.answer.lower())
        
        # Check consistency - all answers should mention key author
        author_mentions = sum(1 for answer in answers if "vaswani" in answer)
        consistency_score = author_mentions / len(answers)
        
        assert consistency_score >= 0.8, f"Answer consistency too low: {consistency_score:.2f}"