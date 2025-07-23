"""
Comprehensive test suite for the RAG system.
"""
import pytest
import asyncio
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
import tempfile
import json
from typing import List, Dict, Any

from langchain_core.documents import Document
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

# Import your modules (adjust imports based on actual structure)
# from enhanced_document_processor import EnhancedPDFProcessor, PaperMetadata
# from vector_store_manager import VectorStoreManager
# from hybrid_retrieval import HybridRetriever, BM25Retriever
# from config_example import RAGConfig


class TestPaperMetadata:
    """Test PaperMetadata class."""
    
    def test_metadata_creation(self):
        """Test metadata object creation."""
        metadata = PaperMetadata(
            title="Test Paper",
            authors=["Author 1", "Author 2"],
            institutions=["University A"],
            publication_date="2024"
        )
        
        assert metadata.title == "Test Paper"
        assert len(metadata.authors) == 2
        assert metadata.keywords == []  # Should be initialized to empty list
    
    def test_metadata_validation(self):
        """Test metadata validation and cleaning."""
        metadata = PaperMetadata(
            title="  Test Paper  ",
            authors=["  Author 1  ", "", "Author 2"],
            institutions=[],
            publication_date="2024"
        )
        
        assert metadata.title == "Test Paper"  # Should be stripped
        assert len(metadata.authors) == 2  # Empty authors should be filtered
        assert "Author 1" in metadata.authors
        assert "" not in metadata.authors


class TestEnhancedPDFProcessor:
    """Test PDF processing functionality."""
    
    @pytest.fixture
    def mock_llm(self):
        """Mock LLM for testing."""
        llm = Mock(spec=ChatOpenAI)
        mock_response = Mock()
        mock_response.content = """
        PAPER METADATA:
        Title: Sample Research Paper
        Authors: John Doe, Jane Smith
        Institutions: University of Test
        Publication Date: 2024
        ArXiv ID: arXiv:2024.12345
        Keywords: machine learning, neural networks
        Abstract: This is a test abstract.
        """
        llm.invoke.return_value = mock_response
        return llm
    
    @pytest.fixture
    def pdf_processor(self, mock_llm):
        """Create PDF processor with mocked dependencies."""
        return EnhancedPDFProcessor(
            llm=mock_llm,
            chunk_size=100,
            chunk_overlap=20
        )
    
    def test_metadata_extraction_success(self, pdf_processor):
        """Test successful metadata extraction."""
        content = "Sample paper content with title and authors."
        
        metadata = pdf_processor.extract_metadata(content)
        
        assert isinstance(metadata, PaperMetadata)
        assert metadata.title != "Unknown Paper"  # Should extract actual title
        assert len(metadata.authors) > 0
    
    def test_metadata_extraction_failure(self, pdf_processor):
        """Test metadata extraction with failure fallback."""
        # Mock LLM to raise an exception
        pdf_processor.llm.invoke.side_effect = Exception("API Error")
        
        metadata = pdf_processor.extract_metadata("content")
        
        assert metadata.title == "Unknown Paper"
        assert metadata.confidence_score == 0.1
    
    def test_empty_content_handling(self, pdf_processor):
        """Test handling of empty content."""
        with pytest.raises(ValueError, match="Content cannot be empty"):
            pdf_processor.extract_metadata("")
    
    @patch('enhanced_document_processor.PdfReader')
    def test_document_processing(self, mock_pdf_reader, pdf_processor):
        """Test complete document processing."""
        # Mock PDF reader
        mock_reader = Mock()
        mock_page = Mock()
        mock_page.extract_text.return_value = "Sample page content"
        mock_reader.pages = [mock_page, mock_page]
        mock_pdf_reader.return_value = mock_reader
        
        # Mock file path handling
        with patch.object(pdf_processor, '_get_pdf_path') as mock_get_path:
            mock_get_path.return_value = Path("test.pdf")
            
            with patch.object(pdf_processor, '_create_document_chunks') as mock_create_chunks:
                mock_chunks = [
                    Document(page_content="Chunk 1", metadata={"page": 1}),
                    Document(page_content="Chunk 2", metadata={"page": 2})
                ]
                mock_create_chunks.return_value = mock_chunks
                
                chunks, metadata = pdf_processor.process_document("test_url")
                
                assert len(chunks) == 2
                assert isinstance(metadata, PaperMetadata)
                mock_pdf_reader.assert_called_once()


class TestVectorStoreManager:
    """Test vector store management."""
    
    @pytest.fixture
    def mock_embeddings(self):
        """Mock embeddings for testing."""
        embeddings = Mock(spec=OpenAIEmbeddings)
        embeddings.embed_documents.return_value = [
            [0.1, 0.2, 0.3],  # Mock embedding vectors
            [0.4, 0.5, 0.6]
        ]
        embeddings.embed_query.return_value = [0.1, 0.2, 0.3]
        return embeddings
    
    @pytest.fixture
    def temp_index_path(self):
        """Create temporary directory for index."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            yield Path(tmp_dir)
    
    def test_vector_store_initialization(self, mock_embeddings, temp_index_path):
        """Test vector store initialization."""
        manager = VectorStoreManager(
            embeddings=mock_embeddings,
            index_path=temp_index_path,
            index_type="flat"
        )
        
        assert manager.embeddings == mock_embeddings
        assert manager.index_path == temp_index_path
        assert manager.index_type == "flat"
        assert manager.documents == []
    
    def test_add_documents(self, mock_embeddings, temp_index_path):
        """Test adding documents to vector store."""
        manager = VectorStoreManager(
            embeddings=mock_embeddings,
            index_path=temp_index_path
        )
        
        docs = [
            Document(page_content="Test document 1", metadata={"id": 1}),
            Document(page_content="Test document 2", metadata={"id": 2})
        ]
        
        manager.add_documents(docs)
        
        assert len(manager.documents) == 2
        assert manager.index is not None
        mock_embeddings.embed_documents.assert_called_once()
    
    @patch('vector_store_manager.faiss')
    def test_search_functionality(self, mock_faiss, mock_embeddings, temp_index_path):
        """Test search functionality."""
        # Setup mocks
        mock_index = Mock()
        mock_index.search.return_value = ([0.9, 0.8], [0, 1])
        mock_faiss.IndexFlatIP.return_value = mock_index
        
        manager = VectorStoreManager(
            embeddings=mock_embeddings,
            index_path=temp_index_path
        )
        
        # Add test documents
        docs = [
            Document(page_content="Test document 1"),
            Document(page_content="Test document 2")
        ]
        manager.add_documents(docs)
        
        # Perform search
        results = manager.search("test query", k=2)
        
        assert len(results) == 2
        assert all(isinstance(doc, Document) for doc, score in results)
        assert all(isinstance(score, float) for doc, score in results)


class TestHybridRetrieval:
    """Test hybrid retrieval functionality."""
    
    @pytest.fixture
    def sample_documents(self):
        """Create sample documents for testing."""
        return [
            Document(page_content="Machine learning algorithms for classification tasks"),
            Document(page_content="Deep neural networks with backpropagation training"),
            Document(page_content="Natural language processing using transformer models"),
            Document(page_content="Computer vision applications with convolutional networks")
        ]
    
    def test_bm25_retriever_initialization(self, sample_documents):
        """Test BM25 retriever initialization."""
        retriever = BM25Retriever(sample_documents)
        
        assert len(retriever.documents) == 4
        assert len(retriever.tokenized_docs) == 4
        assert retriever.bm25 is not None
    
    def test_bm25_search(self, sample_documents):
        """Test BM25 search functionality."""
        retriever = BM25Retriever(sample_documents)
        
        results = retriever.search("neural networks", k=2)
        
        assert len(results) <= 2
        assert all(isinstance(doc, Document) for doc, score in results)
        assert all(score > 0 for doc, score in results)
    
    def test_hybrid_retrieval_combination(self, sample_documents):
        """Test hybrid retrieval combining dense and sparse."""
        # Mock dense retriever
        mock_dense = Mock()
        mock_dense.search.return_value = [
            (sample_documents[0], 0.9),
            (sample_documents[1], 0.8)
        ]
        
        # Create sparse retriever
        sparse_retriever = BM25Retriever(sample_documents)
        
        # Create hybrid retriever
        hybrid = HybridRetriever(
            dense_retriever=mock_dense,
            sparse_retriever=sparse_retriever,
            alpha=0.7,
            beta=0.3
        )
        
        results = hybrid.search("neural networks", k=2)
        
        assert len(results) <= 2
        assert all(isinstance(result.document, Document) for result in results)
        assert all(result.combined_score > 0 for result in results)


class TestRAGConfig:
    """Test configuration management."""
    
    def test_config_validation(self):
        """Test configuration validation."""
        # Test valid config
        config = RAGConfig(
            openai_api_key="test_key",
            chunk_size=800,
            chunk_overlap=150
        )
        
        assert config.openai_api_key == "test_key"
        assert config.chunk_size == 800
        assert config.chunk_overlap == 150
    
    def test_invalid_chunk_size(self):
        """Test invalid chunk size validation."""
        with pytest.raises(ValueError, match="chunk_size must be between 100 and 2000"):
            RAGConfig(
                openai_api_key="test_key",
                chunk_size=50  # Too small
            )
    
    def test_invalid_chunk_overlap(self):
        """Test invalid chunk overlap validation."""
        with pytest.raises(ValueError, match="chunk_overlap must be less than chunk_size"):
            RAGConfig(
                openai_api_key="test_key",
                chunk_size=800,
                chunk_overlap=900  # Larger than chunk_size
            )


class TestAsyncProcessing:
    """Test async processing functionality."""
    
    @pytest.mark.asyncio
    async def test_async_embedding_processor(self):
        """Test async embedding processing."""
        # Mock embeddings
        mock_embeddings = Mock()
        mock_embeddings.embed_documents.return_value = [[0.1, 0.2], [0.3, 0.4]]
        
        processor = AsyncEmbeddingProcessor(
            embeddings=mock_embeddings,
            batch_size=2,
            max_concurrent=2
        )
        
        docs = [
            Document(page_content="Test doc 1"),
            Document(page_content="Test doc 2")
        ]
        
        results = await processor.embed_documents_batch(docs)
        
        assert len(results) == 2
        assert all(result.success for result in results)
    
    @pytest.mark.asyncio
    async def test_async_pdf_downloader(self):
        """Test async PDF downloading."""
        downloader = AsyncPDFDownloader(max_concurrent=2)
        
        # Mock aiohttp session
        with patch('async_processor.aiohttp.ClientSession') as mock_session:
            mock_response = Mock()
            mock_response.read.return_value = b"PDF content"
            mock_response.raise_for_status.return_value = None
            
            mock_session.return_value.__aenter__.return_value.get.return_value.__aenter__.return_value = mock_response
            
            urls = ["http://example.com/paper1.pdf"]
            results = await downloader.download_pdfs(urls)
            
            assert len(results) == 1
            assert results[urls[0]].success


# Integration tests
class TestRAGIntegration:
    """Integration tests for the complete RAG system."""
    
    @pytest.fixture
    def mock_rag_system(self):
        """Create mock RAG system for integration testing."""
        # This would use your actual RAG system class
        pass
    
    def test_end_to_end_processing(self):
        """Test complete pipeline from PDF to Q&A."""
        # This would test the complete workflow:
        # 1. Process PDF
        # 2. Extract metadata
        # 3. Generate summary
        # 4. Create embeddings
        # 5. Build vector store
        # 6. Answer questions
        pass


# Fixtures for test data
@pytest.fixture
def sample_pdf_content():
    """Sample PDF content for testing."""
    return """
    Title: Test Research Paper
    Authors: John Doe, Jane Smith
    Abstract: This is a test paper about machine learning.
    
    1. Introduction
    Machine learning has become increasingly important...
    
    2. Methodology
    We propose a novel approach...
    """


@pytest.fixture
def mock_api_responses():
    """Mock API responses for testing."""
    return {
        'embedding_response': [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]],
        'chat_response': "This is a mock response from the language model.",
        'metadata_response': {
            'title': 'Test Paper',
            'authors': ['John Doe'],
            'year': '2024'
        }
    }


# Performance tests
@pytest.mark.performance
class TestPerformance:
    """Performance and load testing."""
    
    def test_embedding_performance(self):
        """Test embedding generation performance."""
        # Test with various document sizes
        pass
    
    def test_search_performance(self):
        """Test search performance with large indices."""
        # Test search time with increasing index sizes
        pass


# CLI tests
class TestCLI:
    """Test CLI functionality."""
    
    def test_cli_paper_processing(self):
        """Test CLI paper processing command."""
        # Use click.testing.CliRunner to test CLI commands
        pass
    
    def test_cli_query_command(self):
        """Test CLI query command."""
        pass


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "--tb=short"])