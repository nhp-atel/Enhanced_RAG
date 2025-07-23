"""
Basic tests for CI/CD pipeline.
"""

def test_imports():
    """Test that all critical imports work."""
    try:
        from langchain_openai import ChatOpenAI, OpenAIEmbeddings
        from langchain_community.vectorstores import FAISS
        from langchain_core.documents import Document
        from pypdf import PdfReader
        import requests
        import numpy as np
        from domain_analyzer import DomainAnalyzer
        assert True
    except ImportError as e:
        assert False, f"Import failed: {e}"

def test_document_creation():
    """Test basic document creation."""
    from langchain_core.documents import Document
    
    doc = Document(
        page_content="Test content", 
        metadata={"source": "test"}
    )
    
    assert doc.page_content == "Test content"
    assert doc.metadata["source"] == "test"

def test_domain_analyzer_init():
    """Test domain analyzer initialization."""
    from unittest.mock import Mock, patch
    
    # Mock both LLM and OpenAI embeddings
    mock_llm = Mock()
    
    with patch('domain_analyzer.OpenAIEmbeddings') as mock_embeddings:
        mock_embeddings.return_value = Mock()
        
        from domain_analyzer import DomainAnalyzer
        analyzer = DomainAnalyzer(mock_llm)
        
        assert analyzer.llm == mock_llm
        assert hasattr(analyzer, 'summarization_prompt')
        assert hasattr(analyzer, 'react_prompt')

def test_requirements_format():
    """Test that requirements.txt is properly formatted."""
    import os
    
    req_file = os.path.join(os.path.dirname(__file__), '..', 'requirements.txt')
    assert os.path.exists(req_file), "requirements.txt not found"
    
    with open(req_file, 'r') as f:
        content = f.read()
        
    # Check that core packages are present
    assert 'langchain' in content
    assert 'langchain-openai' in content
    assert 'faiss-cpu' in content
    assert 'pypdf' in content