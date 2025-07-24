"""
Unit tests for document ingestion module.
"""

import pytest
from pathlib import Path
from unittest.mock import Mock, patch, mock_open
import requests

from src.core.ingest import DocumentIngestor, DocumentMetadata
from src.utils.errors import DocumentError, ValidationError
from langchain_core.documents import Document


class TestDocumentIngestor:
    """Test suite for DocumentIngestor class"""
    
    def test_init(self, mock_llm_client, mock_logger, mock_metrics):
        """Test DocumentIngestor initialization"""
        config = {
            'max_pages_for_metadata': 3,
            'metadata_char_limit': 8000,
            'supported_formats': ['.pdf'],
            'download_timeout': 60,
            'max_file_size_mb': 100
        }
        
        ingestor = DocumentIngestor(mock_llm_client, mock_logger, mock_metrics, config)
        
        assert ingestor.llm_client == mock_llm_client
        assert ingestor.logger == mock_logger
        assert ingestor.metrics == mock_metrics
        assert ingestor.max_pages_for_metadata == 3
        assert ingestor.metadata_char_limit == 8000
        assert ingestor.supported_formats == ['.pdf']
    
    def test_validate_source_valid_url(self, mock_llm_client, mock_logger, mock_metrics):
        """Test source validation with valid URL"""
        ingestor = DocumentIngestor(mock_llm_client, mock_logger, mock_metrics, {})
        
        # Should not raise exception
        ingestor._validate_source("https://arxiv.org/pdf/2101.00001.pdf")
        ingestor._validate_source("http://example.com/paper.pdf")
    
    def test_validate_source_invalid_url(self, mock_llm_client, mock_logger, mock_metrics):
        """Test source validation with invalid URL"""
        ingestor = DocumentIngestor(mock_llm_client, mock_logger, mock_metrics, {})
        
        with pytest.raises(ValidationError):
            ingestor._validate_source("http://")
        
        with pytest.raises(ValidationError):
            ingestor._validate_source("not-a-url")
    
    def test_validate_source_valid_file(self, mock_llm_client, mock_logger, mock_metrics, create_test_pdf):
        """Test source validation with valid local file"""
        test_file = create_test_pdf()
        ingestor = DocumentIngestor(mock_llm_client, mock_logger, mock_metrics, {})
        
        # Should not raise exception
        ingestor._validate_source(str(test_file))
    
    def test_validate_source_missing_file(self, mock_llm_client, mock_logger, mock_metrics):
        """Test source validation with missing local file"""
        ingestor = DocumentIngestor(mock_llm_client, mock_logger, mock_metrics, {})
        
        with pytest.raises(ValidationError):
            ingestor._validate_source("/path/to/nonexistent/file.pdf")
    
    def test_validate_source_empty(self, mock_llm_client, mock_logger, mock_metrics):
        """Test source validation with empty source"""
        ingestor = DocumentIngestor(mock_llm_client, mock_logger, mock_metrics, {})
        
        with pytest.raises(ValidationError):
            ingestor._validate_source("")
        
        with pytest.raises(ValidationError):
            ingestor._validate_source(None)
    
    @patch('requests.get')
    def test_download_file_success(self, mock_get, mock_llm_client, mock_logger, mock_metrics, temp_dir):
        """Test successful file download"""
        # Mock successful HTTP response
        mock_response = Mock()
        mock_response.raise_for_status.return_value = None
        mock_response.headers = {'content-length': '1024'}
        mock_response.iter_content.return_value = [b'test content chunk 1', b'test content chunk 2']
        mock_get.return_value = mock_response
        
        ingestor = DocumentIngestor(mock_llm_client, mock_logger, mock_metrics, {
            'download_timeout': 60,
            'max_file_size_mb': 100
        })
        
        # Patch the download directory to use temp_dir
        with patch.object(Path, 'mkdir'):
            with patch('src.core.ingest.Path') as mock_path_class:
                mock_path_class.return_value = temp_dir / "downloads"
                mock_path_class.return_value.mkdir.return_value = None
                
                # Mock the file path construction
                expected_file = temp_dir / "downloads" / "2101.00001.pdf"
                
                with patch('builtins.open', mock_open()) as mock_file:
                    result = ingestor._download_file("https://arxiv.org/pdf/2101.00001.pdf")
                    
                    # Verify the request was made correctly
                    mock_get.assert_called_once_with(
                        "https://arxiv.org/pdf/2101.00001.pdf",
                        timeout=60,
                        stream=True
                    )
                    
                    # Verify file was written
                    mock_file.assert_called()
    
    @patch('requests.get')
    def test_download_file_too_large(self, mock_get, mock_llm_client, mock_logger, mock_metrics):
        """Test download failure due to file size limit"""
        # Mock response with large content-length
        mock_response = Mock()
        mock_response.headers = {'content-length': str(200 * 1024 * 1024)}  # 200MB
        mock_get.return_value = mock_response
        
        ingestor = DocumentIngestor(mock_llm_client, mock_logger, mock_metrics, {
            'max_file_size_mb': 100  # 100MB limit
        })
        
        with pytest.raises(ValidationError, match="File too large"):
            ingestor._download_file("https://example.com/large_file.pdf")
    
    @patch('requests.get')
    def test_download_file_request_error(self, mock_get, mock_llm_client, mock_logger, mock_metrics):
        """Test download failure due to request error"""
        mock_get.side_effect = requests.RequestException("Connection failed")
        
        ingestor = DocumentIngestor(mock_llm_client, mock_logger, mock_metrics, {})
        
        with pytest.raises(DocumentError, match="Failed to download document"):
            ingestor._download_file("https://example.com/file.pdf")
    
    def test_validate_file_success(self, mock_llm_client, mock_logger, mock_metrics, create_test_pdf):
        """Test successful file validation"""
        test_file = create_test_pdf()
        
        config = {
            'supported_formats': ['.pdf'],
            'max_file_size_mb': 100
        }
        ingestor = DocumentIngestor(mock_llm_client, mock_logger, mock_metrics, config)
        
        # Mock PdfReader to succeed
        with patch('src.core.ingest.PdfReader') as mock_reader:
            mock_reader.return_value.pages = [Mock(), Mock()]  # 2 pages
            
            # Should not raise exception
            ingestor._validate_file(test_file)
    
    def test_validate_file_missing(self, mock_llm_client, mock_logger, mock_metrics):
        """Test file validation with missing file"""
        ingestor = DocumentIngestor(mock_llm_client, mock_logger, mock_metrics, {})
        
        with pytest.raises(ValidationError, match="File does not exist"):
            ingestor._validate_file(Path("/nonexistent/file.pdf"))
    
    def test_validate_file_unsupported_format(self, mock_llm_client, mock_logger, mock_metrics, temp_dir):
        """Test file validation with unsupported format"""
        # Create .txt file instead of .pdf
        txt_file = temp_dir / "test.txt"
        txt_file.write_text("test content")
        
        config = {'supported_formats': ['.pdf'], 'max_file_size_mb': 100}
        ingestor = DocumentIngestor(mock_llm_client, mock_logger, mock_metrics, config)
        
        with pytest.raises(ValidationError, match="Unsupported file format"):
            ingestor._validate_file(txt_file)
    
    def test_validate_file_empty_pdf(self, mock_llm_client, mock_logger, mock_metrics, create_test_pdf):
        """Test file validation with empty PDF"""
        test_file = create_test_pdf()
        
        config = {'supported_formats': ['.pdf'], 'max_file_size_mb': 100}
        ingestor = DocumentIngestor(mock_llm_client, mock_logger, mock_metrics, config)
        
        # Mock PdfReader to return empty pages
        with patch('src.core.ingest.PdfReader') as mock_reader:
            mock_reader.return_value.pages = []  # No pages
            
            with pytest.raises(ValidationError, match="PDF has no pages"):
                ingestor._validate_file(test_file)
    
    def test_load_pdf_pages_success(self, mock_llm_client, mock_logger, mock_metrics, create_test_pdf):
        """Test successful PDF page loading"""
        test_file = create_test_pdf()
        ingestor = DocumentIngestor(mock_llm_client, mock_logger, mock_metrics, {})
        
        # Mock PyPDFLoader
        mock_pages = [
            Document(page_content="Page 1 content", metadata={"page": 0}),
            Document(page_content="Page 2 content", metadata={"page": 1})
        ]
        
        with patch('src.core.ingest.PyPDFLoader') as mock_loader:
            mock_loader.return_value.load.return_value = mock_pages
            
            result = ingestor._load_pdf_pages(test_file)
            
            assert len(result) == 2
            assert result[0].page_content == "Page 1 content"
            assert result[1].page_content == "Page 2 content"
            
            # Verify loader was called with correct path
            mock_loader.assert_called_once_with(str(test_file))
    
    def test_load_pdf_pages_empty(self, mock_llm_client, mock_logger, mock_metrics, create_test_pdf):
        """Test PDF page loading with empty result"""
        test_file = create_test_pdf()
        ingestor = DocumentIngestor(mock_llm_client, mock_logger, mock_metrics, {})
        
        with patch('src.core.ingest.PyPDFLoader') as mock_loader:
            mock_loader.return_value.load.return_value = []
            
            with pytest.raises(DocumentError, match="No content could be extracted"):
                ingestor._load_pdf_pages(test_file)
    
    def test_extract_metadata_success(self, mock_llm_client, mock_logger, mock_metrics, create_test_pdf):
        """Test successful metadata extraction"""
        test_file = create_test_pdf()
        
        # Mock pages
        pages = [
            Document(page_content="Test Paper Title\nBy John Doe and Jane Smith\nAbstract: This is a test paper."),
            Document(page_content="Introduction content...")
        ]
        
        config = {'max_pages_for_metadata': 2, 'metadata_char_limit': 8000}
        ingestor = DocumentIngestor(mock_llm_client, mock_logger, mock_metrics, config)
        
        result = ingestor._extract_metadata(pages, "test_source", test_file)
        
        # Verify result structure
        assert isinstance(result, DocumentMetadata)
        assert result.title == "Test Research Paper"
        assert result.authors == ["John Doe", "Jane Smith"]  
        assert result.file_path == str(test_file)
        assert result.page_count == 2
        
        # Verify LLM was called for metadata extraction
        assert mock_llm_client.call_count > 0
    
    def test_extract_metadata_llm_failure(self, mock_logger, mock_metrics, create_test_pdf):
        """Test metadata extraction with LLM failure"""
        test_file = create_test_pdf()
        
        # Mock LLM client that raises exception
        mock_llm_client = Mock()
        mock_llm_client.generate_with_retry.side_effect = Exception("LLM failed")
        
        pages = [Document(page_content="Test content")]
        
        ingestor = DocumentIngestor(mock_llm_client, mock_logger, mock_metrics, {})
        
        # Should return default metadata when LLM fails
        result = ingestor._extract_metadata(pages, "test_source", test_file)
        
        assert result.title == "Unknown Document"
        assert result.authors == ["Unknown"]
        assert mock_logger.logs[-1]["level"] == "warning"
    
    def test_parse_metadata_text_success(self, mock_llm_client, mock_logger, mock_metrics):
        """Test parsing of LLM-generated metadata text"""
        ingestor = DocumentIngestor(mock_llm_client, mock_logger, mock_metrics, {})
        
        metadata_text = """Title: Advanced Machine Learning Techniques
Authors: Alice Johnson, Bob Wilson, Carol Davis
Institutions: MIT, Stanford University, UC Berkeley  
Publication Date: 2024-02-15
ArXiv ID: 2402.12345
Keywords: machine learning, deep learning, neural networks
Abstract: This paper presents novel approaches to machine learning that achieve state-of-the-art results on multiple benchmarks.
--- END OF METADATA ---"""
        
        result = ingestor._parse_metadata_text(metadata_text)
        
        assert result.title == "Advanced Machine Learning Techniques"
        assert result.authors == ["Alice Johnson", "Bob Wilson", "Carol Davis"]
        assert result.institutions == ["MIT", "Stanford University", "UC Berkeley"]
        assert result.publication_date == "2024-02-15"
        assert result.arxiv_id == "2402.12345"
        assert result.keywords == ["machine learning", "deep learning", "neural networks"]
        assert "novel approaches" in result.abstract
    
    def test_parse_list_field(self, mock_llm_client, mock_logger, mock_metrics):
        """Test parsing of comma-separated list fields"""
        ingestor = DocumentIngestor(mock_llm_client, mock_logger, mock_metrics, {})
        
        # Test normal list
        result = ingestor._parse_list_field("Alice Johnson, Bob Wilson, Carol Davis")
        assert result == ["Alice Johnson", "Bob Wilson", "Carol Davis"]
        
        # Test with extra whitespace
        result = ingestor._parse_list_field("  Alice Johnson  ,  Bob Wilson  ,  Carol Davis  ")
        assert result == ["Alice Johnson", "Bob Wilson", "Carol Davis"]
        
        # Test unknown value
        result = ingestor._parse_list_field("Unknown")
        assert result == ["Unknown"]
        
        # Test empty/None
        result = ingestor._parse_list_field("")
        assert result == ["Unknown"]
        
        result = ingestor._parse_list_field(None)
        assert result == ["Unknown"]
    
    def test_create_document_chunks(self, mock_llm_client, mock_logger, mock_metrics, sample_metadata):
        """Test document chunk creation"""
        ingestor = DocumentIngestor(mock_llm_client, mock_logger, mock_metrics, {})
        
        pages = [
            Document(page_content="Page 1 content", metadata={"page": 0}),
            Document(page_content="Page 2 content", metadata={"page": 1})
        ]
        
        result = ingestor._create_document_chunks(pages, sample_metadata)
        
        # Should have metadata doc + 2 page docs = 3 total
        assert len(result) == 3
        
        # First doc should be metadata
        assert result[0].metadata["type"] == "paper_metadata"
        assert result[0].metadata["chunk_id"] == "metadata_0"
        assert "PAPER METADATA:" in result[0].page_content
        
        # Other docs should be pages
        assert result[1].metadata["type"] == "content"
        assert result[1].metadata["chunk_id"] == "page_0"
        assert result[1].metadata["page_number"] == 1
        
        assert result[2].metadata["type"] == "content"
        assert result[2].metadata["chunk_id"] == "page_1"
        assert result[2].metadata["page_number"] == 2
    
    def test_format_metadata_for_storage(self, mock_llm_client, mock_logger, mock_metrics, sample_metadata):
        """Test metadata formatting for storage"""
        ingestor = DocumentIngestor(mock_llm_client, mock_logger, mock_metrics, {})
        
        result = ingestor._format_metadata_for_storage(sample_metadata)
        
        assert "PAPER METADATA:" in result
        assert "Title: Test Machine Learning Paper" in result
        assert "Authors: John Doe, Jane Smith" in result
        assert "Keywords: machine learning, neural networks" in result
        assert "--- END OF METADATA ---" in result
    
    def test_calculate_content_hash(self, mock_llm_client, mock_logger, mock_metrics):
        """Test content hash calculation"""
        ingestor = DocumentIngestor(mock_llm_client, mock_logger, mock_metrics, {})
        
        # Same content should produce same hash
        hash1 = ingestor._calculate_content_hash("test content")
        hash2 = ingestor._calculate_content_hash("test content")
        assert hash1 == hash2
        
        # Different content should produce different hash
        hash3 = ingestor._calculate_content_hash("different content")
        assert hash1 != hash3
        
        # Hash should be 16 characters (truncated SHA256)
        assert len(hash1) == 16
    
    def test_generate_document_id(self, mock_llm_client, mock_logger, mock_metrics, sample_metadata):
        """Test document ID generation"""
        ingestor = DocumentIngestor(mock_llm_client, mock_logger, mock_metrics, {})
        
        doc_id = ingestor._generate_document_id(sample_metadata)
        
        # Should be 12 characters (truncated MD5)
        assert len(doc_id) == 12
        
        # Same metadata should produce same ID
        doc_id2 = ingestor._generate_document_id(sample_metadata)
        assert doc_id == doc_id2
    
    def test_validate_source_api_method(self, mock_llm_client, mock_logger, mock_metrics, create_test_pdf):
        """Test public validate_source method"""
        ingestor = DocumentIngestor(mock_llm_client, mock_logger, mock_metrics, {})
        
        # Test valid local file
        test_file = create_test_pdf()
        result = ingestor.validate_source(str(test_file))
        
        assert result["valid"] is True
        assert result["source_type"] == "local_file"
        assert result["format"] == ".pdf"
        assert result["estimated_size_mb"] is not None
        
        # Test valid URL
        result = ingestor.validate_source("https://arxiv.org/pdf/2101.00001.pdf")
        
        assert result["valid"] is True
        assert result["source_type"] == "url"
        
        # Test invalid source
        result = ingestor.validate_source("/nonexistent/file.pdf")
        
        assert result["valid"] is False
        assert "error" in result
    
    def test_get_supported_formats(self, mock_llm_client, mock_logger, mock_metrics):
        """Test getting supported formats"""
        config = {'supported_formats': ['.pdf', '.txt']}
        ingestor = DocumentIngestor(mock_llm_client, mock_logger, mock_metrics, config)
        
        formats = ingestor.get_supported_formats()
        assert formats == ['.pdf', '.txt']
        
        # Should return copy, not original list
        formats.append('.doc')
        assert ingestor.supported_formats == ['.pdf', '.txt']