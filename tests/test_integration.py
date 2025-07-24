"""
Integration tests for the RAG system.
"""

import pytest
import sys
from pathlib import Path
import tempfile
import shutil

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from src.core.pipeline import RAGPipeline, create_pipeline
from src.utils.config import RAGConfig, ConfigManager
from src.utils.errors import RAGError


class TestRAGIntegration:
    """Integration tests for the complete RAG pipeline"""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for test files"""
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def test_config(self, temp_dir):
        """Create test configuration"""
        config = RAGConfig()
        
        # Use smaller values for faster tests
        config.document_processing.chunk_size = 200
        config.document_processing.chunk_overlap = 50
        config.retrieval.default_k = 3
        
        # Use temp directory for persistence
        config.vector_store.persistence['directory'] = str(temp_dir / 'indices')
        config.caching.directory = str(temp_dir / 'cache')
        
        # Disable some features for testing
        config.caching.enabled = False
        config.monitoring.enabled = False
        
        return config
    
    @pytest.fixture
    def pipeline(self, test_config):
        """Create test pipeline"""
        return RAGPipeline(test_config)
    
    def test_pipeline_initialization(self, pipeline):
        """Test pipeline initializes correctly"""
        assert pipeline.is_initialized
        assert not pipeline.vector_store_loaded
        
        stats = pipeline.get_stats()
        assert stats['initialized']
        assert not stats['vector_store_loaded']
    
    def test_document_processing_success(self, pipeline, temp_dir):
        """Test successful document processing"""
        # Create a simple test document
        test_doc = temp_dir / 'test.txt'
        test_content = """
        Test Document Title
        
        This is a test document for the RAG system.
        It contains multiple paragraphs to test document splitting.
        
        The document discusses various topics including:
        - Natural language processing
        - Information retrieval
        - Question answering systems
        
        This content should be processed successfully by the system.
        """
        
        with open(test_doc, 'w') as f:
            f.write(test_content)
        
        # Process document
        result = pipeline.process_document(str(test_doc))
        
        assert result.success
        assert result.document_id is not None
        assert result.chunk_count > 0
        assert result.processing_time_ms > 0
        assert pipeline.vector_store_loaded
        
        # Check stats after processing
        stats = pipeline.get_stats()
        assert stats['vector_store_loaded']
        assert stats['vector_store']['document_count'] > 0
    
    def test_document_processing_invalid_source(self, pipeline):
        """Test document processing with invalid source"""
        result = pipeline.process_document('nonexistent_file.pdf')
        
        assert not result.success
        assert result.error_message is not None
        assert 'not exist' in result.error_message.lower()
    
    def test_query_without_documents(self, pipeline):
        """Test querying without processed documents"""
        with pytest.raises(RAGError) as exc_info:
            pipeline.query("What is this document about?")
        
        assert "No documents have been processed" in str(exc_info.value)
    
    def test_complete_workflow(self, pipeline, temp_dir):
        """Test complete document processing and querying workflow"""
        # Create test document
        test_doc = temp_dir / 'test_paper.txt'
        test_content = """
        Machine Learning in Natural Language Processing
        Authors: John Doe, Jane Smith
        
        Abstract:
        This paper presents a comprehensive study of machine learning applications
        in natural language processing. We explore various algorithms and their
        effectiveness in text analysis tasks.
        
        Introduction:
        Natural language processing (NLP) is a subfield of artificial intelligence
        that focuses on the interaction between computers and human language.
        Machine learning has revolutionized this field by enabling systems to
        learn patterns from data.
        
        Methodology:
        We implemented several machine learning algorithms including:
        - Support Vector Machines (SVM)
        - Random Forest classifiers
        - Neural networks
        
        Results:
        Our experiments show that neural networks achieve the highest accuracy
        of 95% on text classification tasks, followed by SVM at 87%.
        
        Conclusion:
        Machine learning techniques significantly improve NLP system performance.
        Future work will explore deep learning architectures.
        """
        
        with open(test_doc, 'w') as f:
            f.write(test_content)
        
        # Process document
        process_result = pipeline.process_document(str(test_doc))
        assert process_result.success
        
        # Test various query types
        queries = [
            "Who are the authors of this paper?",
            "What is the accuracy of neural networks?",
            "What algorithms were implemented?",
            "What is the conclusion of this study?"
        ]
        
        for question in queries:
            query_result = pipeline.query(question, k=2, strategy="enhanced")
            
            assert query_result.answer is not None
            assert len(query_result.answer) > 0
            assert query_result.sources is not None
            assert len(query_result.sources) > 0
            assert query_result.processing_time_ms > 0
            assert query_result.tokens_used > 0
            
            # Check that answer is relevant (basic sanity check)
            if "authors" in question.lower():
                assert any(name in query_result.answer.lower() 
                          for name in ["john", "jane", "doe", "smith"])
            elif "accuracy" in question.lower():
                assert any(num in query_result.answer 
                          for num in ["95", "87", "accuracy"])
    
    def test_index_persistence(self, pipeline, temp_dir):
        """Test index saving and loading"""
        # Create and process test document
        test_doc = temp_dir / 'test.txt'
        with open(test_doc, 'w') as f:
            f.write("Test document for persistence testing.")
        
        process_result = pipeline.process_document(str(test_doc))
        assert process_result.success
        
        # Save index
        index_path = temp_dir / 'saved_index'
        save_success = pipeline.save_index(str(index_path))
        assert save_success
        assert index_path.exists()
        
        # Create new pipeline and load index
        new_pipeline = RAGPipeline(pipeline.config)
        assert not new_pipeline.vector_store_loaded
        
        load_success = new_pipeline.load_index(str(index_path))
        assert load_success
        assert new_pipeline.vector_store_loaded
        
        # Test that loaded pipeline works
        query_result = new_pipeline.query("What is this document about?")
        assert query_result.answer is not None
    
    def test_health_check(self, pipeline):
        """Test system health check"""
        health = pipeline.health_check()
        
        assert 'overall' in health
        assert 'components' in health
        assert 'timestamp' in health
        
        # Check component health
        assert 'llm' in health['components']
        assert 'embeddings' in health['components']
        assert 'vector_store' in health['components']


class TestConfigurationSystem:
    """Test configuration management"""
    
    def test_default_config_creation(self, tmp_path):
        """Test default configuration creation"""
        config_file = tmp_path / 'test_config.yaml'
        
        config_manager = ConfigManager(str(config_file))
        config = config_manager.load_config()
        
        assert config is not None
        assert config.system.name == "Enhanced RAG System"
        assert config.llm.provider == "openai"
        assert config.embeddings.provider == "openai"
        assert config_file.exists()
    
    def test_config_validation(self):
        """Test configuration validation"""
        config = RAGConfig()
        
        # Test invalid chunk size
        config.document_processing.chunk_size = -1
        
        config_manager = ConfigManager()
        config_manager.config = config
        
        with pytest.raises(Exception):  # Should raise validation error
            config_manager._validate_config(config)


class TestErrorHandling:
    """Test error handling throughout the system"""
    
    def test_missing_api_key(self):
        """Test handling of missing API keys"""
        import os
        
        # Backup original key
        original_key = os.environ.get('OPENAI_API_KEY')
        
        try:
            # Remove API key
            if 'OPENAI_API_KEY' in os.environ:
                del os.environ['OPENAI_API_KEY']
            
            config = RAGConfig()
            config.llm.api_key = None
            
            # Should raise error during initialization
            with pytest.raises(Exception):
                RAGPipeline(config)
        
        finally:
            # Restore original key
            if original_key:
                os.environ['OPENAI_API_KEY'] = original_key
    
    def test_invalid_document_source(self):
        """Test handling of invalid document sources"""
        config = RAGConfig()
        pipeline = RAGPipeline(config)
        
        # Test empty source
        result = pipeline.process_document("")
        assert not result.success
        assert "cannot be empty" in result.error_message.lower()
        
        # Test invalid URL
        result = pipeline.process_document("not_a_valid_url")
        assert not result.success


# Utility functions for test setup

def create_test_document(content: str, file_path: Path) -> None:
    """Create a test document with given content"""
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)


def mock_llm_response(prompt: str) -> str:
    """Mock LLM response for testing"""
    if "authors" in prompt.lower():
        return "The authors are John Doe and Jane Smith."
    elif "title" in prompt.lower():
        return "The title is 'Test Document for RAG System'."
    else:
        return "This is a test response from the mock LLM."


# Test fixtures for common test data

@pytest.fixture
def sample_research_paper():
    """Sample research paper content for testing"""
    return """
    Deep Learning for Natural Language Processing: A Comprehensive Survey
    
    Authors: Alice Johnson, Bob Wilson, Carol Davis
    Institution: University of Technology
    
    Abstract:
    This survey provides a comprehensive overview of deep learning applications
    in natural language processing. We review recent advances in neural
    architectures and their applications to various NLP tasks.
    
    1. Introduction
    Natural language processing has been transformed by deep learning.
    Neural networks have achieved state-of-the-art results across many tasks.
    
    2. Neural Network Architectures
    2.1 Recurrent Neural Networks
    RNNs process sequential data effectively.
    
    2.2 Transformer Architecture
    Transformers use attention mechanisms for better performance.
    
    3. Applications
    3.1 Machine Translation
    Neural machine translation systems achieve high BLEU scores.
    
    3.2 Question Answering
    BERT and GPT models excel at question answering tasks.
    
    4. Conclusion
    Deep learning continues to advance NLP capabilities.
    Future research will focus on multimodal understanding.
    """


if __name__ == '__main__':
    pytest.main([__file__, '-v'])