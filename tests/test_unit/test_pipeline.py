"""
Unit tests for RAG pipeline orchestration.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path

from src.core.pipeline import RAGPipeline
from src.core.ingest import DocumentMetadata
from src.utils.config import RAGConfig
from langchain_core.documents import Document


class TestRAGPipeline:
    """Test suite for RAGPipeline class"""
    
    def test_init_with_config(self, test_config, mock_logger, mock_metrics):
        """Test pipeline initialization with configuration"""
        # Mock the component creation functions
        with patch('src.core.pipeline.create_llm_client') as mock_create_llm, \
             patch('src.core.pipeline.create_embedding_client') as mock_create_embed, \
             patch('src.core.pipeline.create_vector_store') as mock_create_store, \
             patch('src.core.pipeline.get_logger') as mock_get_logger, \
             patch('src.core.pipeline.get_metrics') as mock_get_metrics:
            
            mock_get_logger.return_value = mock_logger
            mock_get_metrics.return_value = mock_metrics
            mock_create_llm.return_value = Mock()
            mock_create_embed.return_value = Mock()
            mock_create_store.return_value = Mock()
            
            pipeline = RAGPipeline(test_config)
            
            assert pipeline.config == test_config
            assert pipeline.logger == mock_logger
            assert pipeline.metrics == mock_metrics
            
            # Verify component creation was called
            mock_create_llm.assert_called_once()
            mock_create_embed.assert_called_once()
            mock_create_store.assert_called_once()
    
    def test_process_document_success(self, test_config, mock_llm_client, mock_embedding_client, 
                                    mock_vector_store, mock_logger, mock_metrics, sample_documents, sample_metadata):
        """Test successful document processing"""
        
        with patch('src.core.pipeline.create_llm_client', return_value=mock_llm_client), \
             patch('src.core.pipeline.create_embedding_client', return_value=mock_embedding_client), \
             patch('src.core.pipeline.create_vector_store', return_value=mock_vector_store), \
             patch('src.core.pipeline.get_logger', return_value=mock_logger), \
             patch('src.core.pipeline.get_metrics', return_value=mock_metrics):
            
            pipeline = RAGPipeline(test_config)
            
            # Mock the ingestor
            mock_ingestor = Mock()
            mock_ingestor.ingest_document.return_value = (sample_documents, sample_metadata)
            pipeline.ingestor = mock_ingestor
            
            # Mock the splitter
            mock_splitter = Mock()
            mock_splitter.split_documents.return_value = sample_documents
            pipeline.splitter = mock_splitter
            
            # Mock embedding generation
            mock_embeddings = [[0.1] * 768 for _ in sample_documents]
            mock_embedding_client.embed_documents.return_value = mock_embeddings
            
            result = pipeline.process_document("test_source.pdf")
            
            assert result.success is True
            assert result.document_id is not None
            assert result.chunk_count == len(sample_documents)
            assert result.metadata == sample_metadata
            
            # Verify components were called
            mock_ingestor.ingest_document.assert_called_once_with("test_source.pdf", None)
            mock_splitter.split_documents.assert_called_once_with(sample_documents)
            mock_embedding_client.embed_documents.assert_called_once()
            mock_vector_store.add_documents.assert_called_once()
    
    def test_process_document_with_custom_id(self, test_config, mock_llm_client, mock_embedding_client,
                                           mock_vector_store, mock_logger, mock_metrics, sample_documents, sample_metadata):
        """Test document processing with custom document ID"""
        
        with patch('src.core.pipeline.create_llm_client', return_value=mock_llm_client), \
             patch('src.core.pipeline.create_embedding_client', return_value=mock_embedding_client), \
             patch('src.core.pipeline.create_vector_store', return_value=mock_vector_store), \
             patch('src.core.pipeline.get_logger', return_value=mock_logger), \
             patch('src.core.pipeline.get_metrics', return_value=mock_metrics):
            
            pipeline = RAGPipeline(test_config)
            
            # Mock components
            mock_ingestor = Mock()
            mock_ingestor.ingest_document.return_value = (sample_documents, sample_metadata)
            pipeline.ingestor = mock_ingestor
            
            mock_splitter = Mock()
            mock_splitter.split_documents.return_value = sample_documents
            pipeline.splitter = mock_splitter
            
            mock_embeddings = [[0.1] * 768 for _ in sample_documents]
            mock_embedding_client.embed_documents.return_value = mock_embeddings
            
            custom_id = "custom_doc_123"
            result = pipeline.process_document("test_source.pdf", document_id=custom_id)
            
            assert result.success is True
            assert result.document_id == custom_id
            
            # Verify custom ID was passed to ingestor
            mock_ingestor.ingest_document.assert_called_once_with("test_source.pdf", custom_id)
    
    def test_process_document_ingestion_failure(self, test_config, mock_llm_client, mock_embedding_client,
                                              mock_vector_store, mock_logger, mock_metrics):
        """Test document processing with ingestion failure"""
        
        with patch('src.core.pipeline.create_llm_client', return_value=mock_llm_client), \
             patch('src.core.pipeline.create_embedding_client', return_value=mock_embedding_client), \
             patch('src.core.pipeline.create_vector_store', return_value=mock_vector_store), \
             patch('src.core.pipeline.get_logger', return_value=mock_logger), \
             patch('src.core.pipeline.get_metrics', return_value=mock_metrics):
            
            pipeline = RAGPipeline(test_config)
            
            # Mock ingestor to raise exception
            mock_ingestor = Mock()
            mock_ingestor.ingest_document.side_effect = Exception("Ingestion failed")
            pipeline.ingestor = mock_ingestor
            
            result = pipeline.process_document("test_source.pdf")
            
            assert result.success is False
            assert "Ingestion failed" in result.error_message
            assert result.document_id is None
            assert result.chunk_count == 0
            
            # Verify error was logged
            error_logs = [log for log in mock_logger.logs if log["level"] == "error"]
            assert len(error_logs) > 0
    
    def test_process_document_embedding_failure(self, test_config, mock_llm_client, mock_embedding_client,
                                              mock_vector_store, mock_logger, mock_metrics, sample_documents, sample_metadata):
        """Test document processing with embedding failure"""
        
        with patch('src.core.pipeline.create_llm_client', return_value=mock_llm_client), \
             patch('src.core.pipeline.create_embedding_client', return_value=mock_embedding_client), \
             patch('src.core.pipeline.create_vector_store', return_value=mock_vector_store), \
             patch('src.core.pipeline.get_logger', return_value=mock_logger), \
             patch('src.core.pipeline.get_metrics', return_value=mock_metrics):
            
            pipeline = RAGPipeline(test_config)
            
            # Mock successful ingestion and splitting
            mock_ingestor = Mock()
            mock_ingestor.ingest_document.return_value = (sample_documents, sample_metadata)
            pipeline.ingestor = mock_ingestor
            
            mock_splitter = Mock()
            mock_splitter.split_documents.return_value = sample_documents
            pipeline.splitter = mock_splitter
            
            # Mock embedding failure
            mock_embedding_client.embed_documents.side_effect = Exception("Embedding failed")
            
            result = pipeline.process_document("test_source.pdf")
            
            assert result.success is False
            assert "Embedding failed" in result.error_message
    
    def test_query_success(self, test_config, mock_llm_client, mock_embedding_client,
                          mock_vector_store, mock_logger, mock_metrics, sample_documents):
        """Test successful query processing"""
        
        with patch('src.core.pipeline.create_llm_client', return_value=mock_llm_client), \
             patch('src.core.pipeline.create_embedding_client', return_value=mock_embedding_client), \
             patch('src.core.pipeline.create_vector_store', return_value=mock_vector_store), \
             patch('src.core.pipeline.get_logger', return_value=mock_logger), \
             patch('src.core.pipeline.get_metrics', return_value=mock_metrics):
            
            pipeline = RAGPipeline(test_config)
            
            # Mock retriever
            mock_retriever = Mock()
            mock_retriever.retrieve_and_generate.return_value = Mock(
                answer="Test answer",
                sources=sample_documents[:2],
                processing_time_ms=100,
                tokens_used=50,
                cost_usd=0.01,
                metadata={"strategy": "basic"}
            )
            pipeline.retriever = mock_retriever
            
            result = pipeline.query("Who are the authors?")
            
            assert result.answer == "Test answer"
            assert len(result.sources) == 2
            assert result.processing_time_ms == 100
            assert result.tokens_used == 50
            assert result.cost_usd == 0.01
            
            # Verify retriever was called correctly
            mock_retriever.retrieve_and_generate.assert_called_once_with(
                question="Who are the authors?",
                k=4,  # from test config
                strategy="basic"
            )
    
    def test_query_with_custom_parameters(self, test_config, mock_llm_client, mock_embedding_client,
                                        mock_vector_store, mock_logger, mock_metrics):
        """Test query with custom parameters"""
        
        with patch('src.core.pipeline.create_llm_client', return_value=mock_llm_client), \
             patch('src.core.pipeline.create_embedding_client', return_value=mock_embedding_client), \
             patch('src.core.pipeline.create_vector_store', return_value=mock_vector_store), \
             patch('src.core.pipeline.get_logger', return_value=mock_logger), \
             patch('src.core.pipeline.get_metrics', return_value=mock_metrics):
            
            pipeline = RAGPipeline(test_config)
            
            mock_retriever = Mock()
            mock_retriever.retrieve_and_generate.return_value = Mock(
                answer="Enhanced answer",
                sources=[],
                processing_time_ms=150,
                tokens_used=75,
                cost_usd=0.02,
                metadata={"strategy": "enhanced"}
            )
            pipeline.retriever = mock_retriever
            
            result = pipeline.query(
                question="What is the main contribution?",
                k=8,
                strategy="enhanced"
            )
            
            assert result.answer == "Enhanced answer"
            
            # Verify custom parameters were passed
            mock_retriever.retrieve_and_generate.assert_called_once_with(
                question="What is the main contribution?",
                k=8,
                strategy="enhanced"
            )
    
    def test_query_failure(self, test_config, mock_llm_client, mock_embedding_client,
                          mock_vector_store, mock_logger, mock_metrics):
        """Test query processing failure"""
        
        with patch('src.core.pipeline.create_llm_client', return_value=mock_llm_client), \
             patch('src.core.pipeline.create_embedding_client', return_value=mock_embedding_client), \
             patch('src.core.pipeline.create_vector_store', return_value=mock_vector_store), \
             patch('src.core.pipeline.get_logger', return_value=mock_logger), \
             patch('src.core.pipeline.get_metrics', return_value=mock_metrics):
            
            pipeline = RAGPipeline(test_config)
            
            # Mock retriever failure
            mock_retriever = Mock()
            mock_retriever.retrieve_and_generate.side_effect = Exception("Query failed")
            pipeline.retriever = mock_retriever
            
            with pytest.raises(Exception, match="Query failed"):
                pipeline.query("What is this about?")
            
            # Verify error was logged
            error_logs = [log for log in mock_logger.logs if log["level"] == "error"]
            assert len(error_logs) > 0
    
    def test_save_index_success(self, test_config, mock_llm_client, mock_embedding_client,
                              mock_vector_store, mock_logger, mock_metrics):
        """Test successful index saving"""
        
        with patch('src.core.pipeline.create_llm_client', return_value=mock_llm_client), \
             patch('src.core.pipeline.create_embedding_client', return_value=mock_embedding_client), \
             patch('src.core.pipeline.create_vector_store', return_value=mock_vector_store), \
             patch('src.core.pipeline.get_logger', return_value=mock_logger), \
             patch('src.core.pipeline.get_metrics', return_value=mock_metrics):
            
            pipeline = RAGPipeline(test_config)
            
            mock_vector_store.save_index.return_value = True
            
            result = pipeline.save_index("./data/test_index")
            
            assert result is True
            mock_vector_store.save_index.assert_called_once_with("./data/test_index")
    
    def test_save_index_failure(self, test_config, mock_llm_client, mock_embedding_client,
                              mock_vector_store, mock_logger, mock_metrics):
        """Test index saving failure"""
        
        with patch('src.core.pipeline.create_llm_client', return_value=mock_llm_client), \
             patch('src.core.pipeline.create_embedding_client', return_value=mock_embedding_client), \
             patch('src.core.pipeline.create_vector_store', return_value=mock_vector_store), \
             patch('src.core.pipeline.get_logger', return_value=mock_logger), \
             patch('src.core.pipeline.get_metrics', return_value=mock_metrics):
            
            pipeline = RAGPipeline(test_config)
            
            mock_vector_store.save_index.side_effect = Exception("Save failed")
            
            result = pipeline.save_index("./data/test_index")
            
            assert result is False
            
            # Verify error was logged
            error_logs = [log for log in mock_logger.logs if log["level"] == "error"]
            assert len(error_logs) > 0
    
    def test_load_index_success(self, test_config, mock_llm_client, mock_embedding_client,
                              mock_vector_store, mock_logger, mock_metrics):
        """Test successful index loading"""
        
        with patch('src.core.pipeline.create_llm_client', return_value=mock_llm_client), \
             patch('src.core.pipeline.create_embedding_client', return_value=mock_embedding_client), \
             patch('src.core.pipeline.create_vector_store', return_value=mock_vector_store), \
             patch('src.core.pipeline.get_logger', return_value=mock_logger), \
             patch('src.core.pipeline.get_metrics', return_value=mock_metrics):
            
            pipeline = RAGPipeline(test_config)
            
            mock_vector_store.load_index.return_value = True
            
            result = pipeline.load_index("./data/test_index")
            
            assert result is True
            mock_vector_store.load_index.assert_called_once_with("./data/test_index")
    
    def test_load_index_failure(self, test_config, mock_llm_client, mock_embedding_client,
                              mock_vector_store, mock_logger, mock_metrics):
        """Test index loading failure"""
        
        with patch('src.core.pipeline.create_llm_client', return_value=mock_llm_client), \
             patch('src.core.pipeline.create_embedding_client', return_value=mock_embedding_client), \
             patch('src.core.pipeline.create_vector_store', return_value=mock_vector_store), \
             patch('src.core.pipeline.get_logger', return_value=mock_logger), \
             patch('src.core.pipeline.get_metrics', return_value=mock_metrics):
            
            pipeline = RAGPipeline(test_config)
            
            mock_vector_store.load_index.side_effect = Exception("Load failed")
            
            result = pipeline.load_index("./data/test_index")
            
            assert result is False
            
            # Verify error was logged
            error_logs = [log for log in mock_logger.logs if log["level"] == "error"]
            assert len(error_logs) > 0
    
    def test_clear_cache_success(self, test_config, mock_llm_client, mock_embedding_client,
                               mock_vector_store, mock_logger, mock_metrics):
        """Test successful cache clearing"""
        
        with patch('src.core.pipeline.create_llm_client', return_value=mock_llm_client), \
             patch('src.core.pipeline.create_embedding_client', return_value=mock_embedding_client), \
             patch('src.core.pipeline.create_vector_store', return_value=mock_vector_store), \
             patch('src.core.pipeline.get_logger', return_value=mock_logger), \
             patch('src.core.pipeline.get_metrics', return_value=mock_metrics):
            
            pipeline = RAGPipeline(test_config)
            
            # Mock cache
            mock_cache = Mock()
            mock_cache.clear.return_value = True
            pipeline.cache = mock_cache
            
            result = pipeline.clear_cache()
            
            assert result is True
            mock_cache.clear.assert_called_once()
    
    def test_health_check(self, test_config, mock_llm_client, mock_embedding_client,
                         mock_vector_store, mock_logger, mock_metrics):
        """Test system health check"""
        
        with patch('src.core.pipeline.create_llm_client', return_value=mock_llm_client), \
             patch('src.core.pipeline.create_embedding_client', return_value=mock_embedding_client), \
             patch('src.core.pipeline.create_vector_store', return_value=mock_vector_store), \
             patch('src.core.pipeline.get_logger', return_value=mock_logger), \
             patch('src.core.pipeline.get_metrics', return_value=mock_metrics):
            
            pipeline = RAGPipeline(test_config)
            
            # Mock component health checks
            mock_llm_client.health_check = Mock(return_value={"status": "healthy"})
            mock_embedding_client.health_check = Mock(return_value={"status": "healthy"})
            mock_vector_store.health_check = Mock(return_value={"status": "healthy", "doc_count": 100})
            
            result = pipeline.health_check()
            
            assert result["overall"] == "healthy"
            assert "llm_client" in result["components"]
            assert "embedding_client" in result["components"]
            assert "vector_store" in result["components"]
            assert "timestamp" in result
    
    def test_get_stats(self, test_config, mock_llm_client, mock_embedding_client,
                      mock_vector_store, mock_logger, mock_metrics):
        """Test getting pipeline statistics"""
        
        with patch('src.core.pipeline.create_llm_client', return_value=mock_llm_client), \
             patch('src.core.pipeline.create_embedding_client', return_value=mock_embedding_client), \
             patch('src.core.pipeline.create_vector_store', return_value=mock_vector_store), \
             patch('src.core.pipeline.get_logger', return_value=mock_logger), \
             patch('src.core.pipeline.get_metrics', return_value=mock_metrics):
            
            pipeline = RAGPipeline(test_config)
            
            # Mock component stats
            mock_vector_store.get_document_count.return_value = 150
            
            result = pipeline.get_stats()
            
            assert "documents_processed" in result
            assert "total_queries" in result
            assert "cache_stats" in result
            assert result["documents_processed"] == 150
    
    def test_cost_tracking(self, test_config, mock_llm_client, mock_embedding_client,
                          mock_vector_store, mock_logger, mock_metrics, sample_documents, sample_metadata):
        """Test cost tracking during operations"""
        
        with patch('src.core.pipeline.create_llm_client', return_value=mock_llm_client), \
             patch('src.core.pipeline.create_embedding_client', return_value=mock_embedding_client), \
             patch('src.core.pipeline.create_vector_store', return_value=mock_vector_store), \
             patch('src.core.pipeline.get_logger', return_value=mock_logger), \
             patch('src.core.pipeline.get_metrics', return_value=mock_metrics):
            
            pipeline = RAGPipeline(test_config)
            
            # Mock cost tracker
            mock_cost_tracker = Mock()
            mock_cost_tracker.track_request_cost.return_value = None
            mock_cost_tracker.get_daily_spend.return_value = 5.50
            pipeline.cost_tracker = mock_cost_tracker
            
            # Mock successful processing
            mock_ingestor = Mock()
            mock_ingestor.ingest_document.return_value = (sample_documents, sample_metadata)
            pipeline.ingestor = mock_ingestor
            
            mock_splitter = Mock()
            mock_splitter.split_documents.return_value = sample_documents
            pipeline.splitter = mock_splitter
            
            mock_embeddings = [[0.1] * 768 for _ in sample_documents]
            mock_embedding_client.embed_documents.return_value = mock_embeddings
            
            result = pipeline.process_document("test_source.pdf")
            
            assert result.success is True
            
            # Verify cost tracking was called
            assert mock_cost_tracker.track_request_cost.call_count > 0