"""
Abstract interfaces for LLM and Embedding providers.
Enables easy swapping between different providers and models.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass
from enum import Enum


class ProviderType(Enum):
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    HUGGINGFACE = "huggingface"
    LOCAL = "local"


@dataclass
class LLMResponse:
    """Standardized LLM response structure"""
    content: str
    model: str
    tokens_used: int
    cost_usd: float
    latency_ms: float
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class EmbeddingResponse:
    """Standardized embedding response structure"""
    embeddings: List[List[float]]
    model: str
    tokens_used: int
    cost_usd: float
    latency_ms: float
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class LLMClient(ABC):
    """Abstract interface for LLM providers"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.provider_type = ProviderType(config.get('provider', 'openai'))
        self.model = config.get('model', 'gpt-4o-mini')
        self.temperature = config.get('temperature', 0.1)
        self.max_tokens = config.get('max_tokens', 2048)
        self.timeout = config.get('timeout', 30)
    
    @abstractmethod
    def generate(self, messages: List[Dict[str, str]], **kwargs) -> LLMResponse:
        """Generate response from messages"""
        pass
    
    @abstractmethod
    def generate_with_retry(self, messages: List[Dict[str, str]], **kwargs) -> LLMResponse:
        """Generate with automatic retry logic"""
        pass
    
    @abstractmethod
    def estimate_cost(self, text: str) -> float:
        """Estimate cost for processing text"""
        pass
    
    @abstractmethod
    def count_tokens(self, text: str) -> int:
        """Count tokens in text"""
        pass
    
    @property
    @abstractmethod
    def max_context_length(self) -> int:
        """Maximum context length for the model"""
        pass


class EmbeddingClient(ABC):
    """Abstract interface for embedding providers"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.provider_type = ProviderType(config.get('provider', 'openai'))
        self.model = config.get('model', 'text-embedding-3-small')
        self.dimensions = config.get('dimensions', 1536)
        self.batch_size = config.get('batch_size', 100)
        self.timeout = config.get('timeout', 30)
    
    @abstractmethod
    def embed_query(self, text: str, **kwargs) -> List[float]:
        """Embed a single query"""
        pass
    
    @abstractmethod
    def embed_documents(self, texts: List[str], **kwargs) -> EmbeddingResponse:
        """Embed multiple documents"""
        pass
    
    @abstractmethod
    def embed_with_retry(self, texts: Union[str, List[str]], **kwargs) -> EmbeddingResponse:
        """Embed with automatic retry logic"""
        pass
    
    @abstractmethod
    def estimate_cost(self, texts: Union[str, List[str]]) -> float:
        """Estimate cost for embedding texts"""
        pass
    
    @abstractmethod
    def count_tokens(self, text: str) -> int:
        """Count tokens in text for embedding"""
        pass


class PromptTemplate(ABC):
    """Abstract interface for prompt templates"""
    
    def __init__(self, template_name: str, template_data: Dict[str, Any]):
        self.name = template_name
        self.system_prompt = template_data.get('system', '')
        self.human_prompt = template_data.get('human', '')
        self.metadata = template_data.get('metadata', {})
    
    @abstractmethod
    def format(self, **kwargs) -> List[Dict[str, str]]:
        """Format the prompt with given variables"""
        pass
    
    def validate_variables(self, **kwargs) -> List[str]:
        """Validate that all required variables are provided"""
        import re
        
        # Extract variables from template
        system_vars = set(re.findall(r'\{(\w+)\}', self.system_prompt))
        human_vars = set(re.findall(r'\{(\w+)\}', self.human_prompt))
        required_vars = system_vars.union(human_vars)
        
        # Check for missing variables
        provided_vars = set(kwargs.keys())
        missing_vars = required_vars - provided_vars
        
        return list(missing_vars)


class VectorStore(ABC):
    """Abstract interface for vector stores"""
    
    @abstractmethod
    def add_documents(self, documents: List[Any], **kwargs) -> List[str]:
        """Add documents to vector store"""
        pass
    
    @abstractmethod
    def similarity_search(self, query: str, k: int = 6, **kwargs) -> List[Any]:
        """Search for similar documents"""
        pass
    
    @abstractmethod
    def similarity_search_with_score(self, query: str, k: int = 6, **kwargs) -> List[tuple]:
        """Search with similarity scores"""
        pass
    
    @abstractmethod
    def delete_documents(self, document_ids: List[str]) -> bool:
        """Delete documents from vector store"""
        pass
    
    @abstractmethod
    def save_local(self, path: str) -> bool:
        """Save index to local storage"""
        pass
    
    @abstractmethod
    def load_local(self, path: str) -> bool:
        """Load index from local storage"""
        pass
    
    @abstractmethod
    def get_stats(self) -> Dict[str, Any]:
        """Get vector store statistics"""
        pass


class CacheInterface(ABC):
    """Abstract interface for caching systems"""
    
    @abstractmethod
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        pass
    
    @abstractmethod
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in cache"""
        pass
    
    @abstractmethod
    def delete(self, key: str) -> bool:
        """Delete value from cache"""
        pass
    
    @abstractmethod
    def clear(self) -> bool:
        """Clear all cache entries"""
        pass
    
    @abstractmethod
    def exists(self, key: str) -> bool:
        """Check if key exists in cache"""
        pass
    
    @abstractmethod
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        pass


class MetricsInterface(ABC):
    """Abstract interface for metrics collection"""
    
    @abstractmethod
    def increment_counter(self, name: str, value: int = 1, tags: Dict[str, str] = None):
        """Increment counter metric"""
        pass
    
    @abstractmethod
    def record_histogram(self, name: str, value: float, tags: Dict[str, str] = None):
        """Record histogram metric"""
        pass
    
    @abstractmethod
    def record_gauge(self, name: str, value: float, tags: Dict[str, str] = None):
        """Record gauge metric"""
        pass
    
    @abstractmethod
    def get_metrics(self) -> Dict[str, Any]:
        """Get all recorded metrics"""
        pass


class LoggerInterface(ABC):
    """Abstract interface for structured logging"""
    
    @abstractmethod
    def info(self, message: str, **kwargs):
        """Log info message"""
        pass
    
    @abstractmethod
    def warning(self, message: str, **kwargs):
        """Log warning message"""
        pass
    
    @abstractmethod
    def error(self, message: str, **kwargs):
        """Log error message"""
        pass
    
    @abstractmethod
    def debug(self, message: str, **kwargs):
        """Log debug message"""
        pass
    
    @abstractmethod
    def set_context(self, **kwargs):
        """Set logging context"""
        pass


# Factory functions for creating implementations
def create_llm_client(config: Dict[str, Any]) -> LLMClient:
    """Factory function to create LLM client based on config"""
    provider = config.get('provider', 'openai').lower()
    
    if provider == 'openai':
        from .providers.openai_provider import OpenAILLMClient
        return OpenAILLMClient(config)
    elif provider == 'anthropic':
        from .providers.anthropic_provider import AnthropicLLMClient
        return AnthropicLLMClient(config)
    else:
        raise ValueError(f"Unsupported LLM provider: {provider}")


def create_embedding_client(config: Dict[str, Any]) -> EmbeddingClient:
    """Factory function to create embedding client based on config"""
    provider = config.get('provider', 'openai').lower()
    
    if provider == 'openai':
        from .providers.openai_provider import OpenAIEmbeddingClient
        return OpenAIEmbeddingClient(config)
    elif provider == 'huggingface':
        from .providers.huggingface_provider import HuggingFaceEmbeddingClient
        return HuggingFaceEmbeddingClient(config)
    else:
        raise ValueError(f"Unsupported embedding provider: {provider}")


def create_vector_store(config: Dict[str, Any], embedding_client: EmbeddingClient) -> VectorStore:
    """Factory function to create vector store based on config"""
    provider = config.get('provider', 'faiss').lower()
    
    if provider == 'faiss':
        from .stores.faiss_store import FAISVectorStore
        return FAISVectorStore(config, embedding_client)
    elif provider == 'pinecone':
        from .stores.pinecone_store import PineconeVectorStore
        return PineconeVectorStore(config, embedding_client)
    else:
        raise ValueError(f"Unsupported vector store provider: {provider}")