"""
Pytest configuration and shared fixtures for RAG system testing.
"""

import pytest
import tempfile
import shutil
from pathlib import Path
from typing import List, Dict, Any, Optional
from unittest.mock import Mock, MagicMock
import json
import numpy as np

from src.interfaces import LLMClient, EmbeddingClient, VectorStore, LoggerInterface, MetricsInterface
from src.core.ingest import DocumentMetadata
from src.utils.config import RAGConfig
from langchain_core.documents import Document


class MockLLMClient(LLMClient):
    """Mock LLM client for testing"""
    
    def __init__(self, responses: Optional[Dict[str, str]] = None):
        self.responses = responses or {}
        self.call_count = 0
        self.last_messages = []
        
    def generate(self, messages: List[Dict[str, str]], **kwargs) -> Any:
        self.call_count += 1
        self.last_messages = messages
        
        # Return predefined responses based on message content
        message_text = " ".join([msg.get("content", "") for msg in messages])
        
        # Metadata extraction responses
        if "extract metadata" in message_text.lower():
            return Mock(content="""Title: Test Research Paper
Authors: John Doe, Jane Smith
Institutions: MIT, Stanford
Publication Date: 2024-01-15
ArXiv ID: 2401.12345
Keywords: machine learning, artificial intelligence
Abstract: This is a test paper about advanced AI techniques.
--- END OF METADATA ---""")
        
        # Summary generation responses  
        if "summarize" in message_text.lower():
            return Mock(content="This paper presents novel approaches to machine learning with significant improvements in accuracy and efficiency.")
            
        # Question answering responses
        if "who are the authors" in message_text.lower():
            return Mock(content="The authors are John Doe and Jane Smith.")
            
        if "main contribution" in message_text.lower():
            return Mock(content="The main contribution is a novel neural network architecture that improves classification accuracy by 15%.")
            
        # Default response
        return Mock(content="This is a mock LLM response for testing purposes.")
    
    def generate_with_retry(self, messages: List[Dict[str, str]], **kwargs) -> Any:
        return self.generate(messages, **kwargs)


class MockEmbeddingClient(EmbeddingClient):
    """Mock embedding client for testing"""
    
    def __init__(self, dimension: int = 768):
        self.dimension = dimension
        self.call_count = 0
        self.last_texts = []
        
    def embed_documents(self, texts: List[str], **kwargs) -> List[List[float]]:
        self.call_count += 1
        self.last_texts = texts
        
        # Generate deterministic embeddings based on text hash
        embeddings = []
        for text in texts:
            # Create deterministic embedding from text hash
            text_hash = hash(text) % (2**31)  # Ensure positive
            np.random.seed(text_hash)
            embedding = np.random.normal(0, 1, self.dimension).tolist()
            embeddings.append(embedding)
            
        return embeddings
    
    def embed_query(self, text: str, **kwargs) -> List[float]:
        return self.embed_documents([text])[0]


class MockVectorStore(VectorStore):
    """Mock vector store for testing"""
    
    def __init__(self):
        self.documents = []
        self.embeddings = []
        self.doc_count = 0
        
    def add_documents(self, documents: List[Document], embeddings: List[List[float]]) -> None:
        self.documents.extend(documents)
        self.embeddings.extend(embeddings)
        self.doc_count += len(documents)
        
    def similarity_search(self, query_embedding: List[float], k: int = 6) -> List[Document]:
        # Return first k documents for simplicity
        return self.documents[:min(k, len(self.documents))]
    
    def save_index(self, path: str) -> bool:
        return True
        
    def load_index(self, path: str) -> bool:
        return True
        
    def get_document_count(self) -> int:
        return self.doc_count


class MockLogger(LoggerInterface):
    """Mock logger for testing"""
    
    def __init__(self):
        self.logs = []
        
    def info(self, message: str, **kwargs):
        self.logs.append({"level": "info", "message": message, "kwargs": kwargs})
        
    def warning(self, message: str, **kwargs):
        self.logs.append({"level": "warning", "message": message, "kwargs": kwargs})
        
    def error(self, message: str, **kwargs):
        self.logs.append({"level": "error", "message": message, "kwargs": kwargs})
        
    def debug(self, message: str, **kwargs):
        self.logs.append({"level": "debug", "message": message, "kwargs": kwargs})


class MockMetrics(MetricsInterface):
    """Mock metrics collector for testing"""
    
    def __init__(self):
        self.counters = {}
        self.histograms = {}
        self.gauges = {}
        
    def increment_counter(self, name: str, value: int = 1, **tags):
        self.counters[name] = self.counters.get(name, 0) + value
        
    def record_histogram(self, name: str, value: float, **tags):
        if name not in self.histograms:
            self.histograms[name] = []
        self.histograms[name].append(value)
        
    def record_gauge(self, name: str, value: float, **tags):
        self.gauges[name] = value


@pytest.fixture
def temp_dir():
    """Create temporary directory for test files"""
    temp_path = Path(tempfile.mkdtemp())
    yield temp_path
    shutil.rmtree(temp_path)


@pytest.fixture
def mock_llm_client():
    """Mock LLM client fixture"""
    return MockLLMClient()


@pytest.fixture
def mock_embedding_client():
    """Mock embedding client fixture"""
    return MockEmbeddingClient()


@pytest.fixture
def mock_vector_store():
    """Mock vector store fixture"""
    return MockVectorStore()


@pytest.fixture
def mock_logger():
    """Mock logger fixture"""
    return MockLogger()


@pytest.fixture
def mock_metrics():
    """Mock metrics fixture"""
    return MockMetrics()


@pytest.fixture
def test_config():
    """Test configuration fixture"""
    return RAGConfig(
        system={"environment": "test"},
        document_processing={
            "chunk_size": 500,
            "overlap": 50,
            "max_pages_for_metadata": 2
        },
        llm={
            "provider": "mock",
            "model": "mock-model",
            "temperature": 0.1
        },
        embeddings={
            "provider": "mock", 
            "model": "mock-embeddings",
            "dimension": 768
        },
        vector_store={
            "provider": "mock",
            "persistence": {"enabled": False}
        },
        retrieval={"default_k": 4},
        cost_control={"daily_budget_usd": 100.0}
    )


@pytest.fixture
def sample_documents():
    """Sample documents for testing"""
    return [
        Document(
            page_content="This is a research paper about machine learning and neural networks.",
            metadata={"source": "test_paper.pdf", "page": 1, "type": "content"}
        ),
        Document(
            page_content="The methodology involves training deep neural networks on large datasets.",
            metadata={"source": "test_paper.pdf", "page": 2, "type": "content"}
        ),
        Document(
            page_content="Results show 95% accuracy on the benchmark dataset.",
            metadata={"source": "test_paper.pdf", "page": 3, "type": "content"}
        )
    ]


@pytest.fixture
def sample_metadata():
    """Sample document metadata for testing"""
    return DocumentMetadata(
        title="Test Machine Learning Paper",
        authors=["John Doe", "Jane Smith"],
        institutions=["MIT", "Stanford"],
        publication_date="2024-01-15",
        arxiv_id="2401.12345",
        keywords=["machine learning", "neural networks"],
        abstract="This paper presents novel approaches to machine learning.",
        source_url="https://arxiv.org/pdf/2401.12345.pdf",
        content_hash="abc123"
    )


@pytest.fixture
def create_test_pdf(temp_dir):
    """Create a test PDF file"""
    def _create_pdf(content: str = "Test PDF content", filename: str = "test.pdf"):
        # Create a simple text file that can be used as a mock PDF
        pdf_path = temp_dir / filename
        pdf_path.write_text(content)
        return pdf_path
    return _create_pdf


# Ground truth data for evaluation
GROUND_TRUTH_QA_PAIRS = [
    {
        "question": "Who are the authors of this paper?",
        "expected_answer": "John Doe and Jane Smith",
        "answer_type": "factual",
        "relevant_chunks": ["metadata"]
    },
    {
        "question": "What is the main contribution of this research?",
        "expected_answer": "novel neural network architecture",
        "answer_type": "conceptual", 
        "relevant_chunks": ["abstract", "conclusion"]
    },
    {
        "question": "What accuracy was achieved?",
        "expected_answer": "95% accuracy",
        "answer_type": "factual",
        "relevant_chunks": ["results"]
    },
    {
        "question": "What methodology was used?",
        "expected_answer": "training deep neural networks on large datasets",
        "answer_type": "technical",
        "relevant_chunks": ["methodology"]
    }
]


@pytest.fixture
def ground_truth_qa():
    """Ground truth Q&A pairs for evaluation"""
    return GROUND_TRUTH_QA_PAIRS


# Test paper corpus for integration testing
TEST_CORPUS = {
    "paper_1": {
        "title": "Attention Is All You Need",
        "authors": ["Ashish Vaswani", "Noam Shazeer"],
        "content": "We propose the Transformer, a model architecture based solely on attention mechanisms...",
        "questions": [
            ("What is the Transformer?", "A model architecture based solely on attention mechanisms"),
            ("Who are the authors?", "Ashish Vaswani and Noam Shazeer")
        ]
    },
    "paper_2": {
        "title": "BERT: Pre-training of Deep Bidirectional Transformers",
        "authors": ["Jacob Devlin", "Ming-Wei Chang"],
        "content": "We introduce BERT, which stands for Bidirectional Encoder Representations from Transformers...",
        "questions": [
            ("What does BERT stand for?", "Bidirectional Encoder Representations from Transformers"),
            ("Who created BERT?", "Jacob Devlin and Ming-Wei Chang")
        ]
    }
}


@pytest.fixture
def test_corpus():
    """Test corpus for integration testing"""
    return TEST_CORPUS