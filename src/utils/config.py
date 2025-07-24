"""
Configuration management system with YAML/CLI support and validation.
"""

import os
import yaml
import argparse
from pathlib import Path
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass, field
from copy import deepcopy

from .errors import ConfigurationError


@dataclass
class SystemConfig:
    """System-level configuration"""
    name: str = "Enhanced RAG System"
    version: str = "1.0.0"
    environment: str = "development"


@dataclass
class LLMConfig:
    """LLM provider configuration"""
    provider: str = "openai"
    model: str = "gpt-4o-mini"
    temperature: float = 0.1
    max_tokens: int = 2048
    timeout: int = 30
    api_key: Optional[str] = None
    rate_limit: Dict[str, int] = field(default_factory=lambda: {
        "requests_per_minute": 60,
        "tokens_per_minute": 150000
    })


@dataclass
class EmbeddingConfig:
    """Embedding provider configuration"""
    provider: str = "openai"
    model: str = "text-embedding-3-small"
    dimensions: int = 1536
    batch_size: int = 100
    timeout: int = 30
    api_key: Optional[str] = None


@dataclass
class DocumentProcessingConfig:
    """Document processing configuration"""
    chunk_size: int = 800
    chunk_overlap: int = 150
    separators: List[str] = field(default_factory=lambda: ["\n\n", "\n", ". ", " ", ""])
    max_pages_for_metadata: int = 3
    max_pages_for_summary: int = 5
    metadata_char_limit: int = 8000
    summary_char_limit: int = 12000
    split_strategy: str = "adaptive"
    max_chunk_size: int = 2000
    min_chunk_size: int = 100


@dataclass
class VectorStoreConfig:
    """Vector store configuration"""
    provider: str = "faiss"
    index_type: str = "IndexFlatIP"
    similarity_metric: str = "cosine"
    persistence: Dict[str, Any] = field(default_factory=lambda: {
        "enabled": True,
        "directory": "./data/indices",
        "save_every_n_docs": 100
    })


@dataclass
class RetrievalConfig:
    """Retrieval configuration"""
    default_k: int = 6
    max_k: int = 20
    similarity_threshold: float = 0.7
    rerank_enabled: bool = True
    strategies: Dict[str, List[str]] = field(default_factory=lambda: {
        "metadata_keywords": ["author", "title", "year", "published", "wrote", "when", "date", "institution"],
        "concept_keywords": ["what is", "define", "explain", "concept", "term", "meaning"],
        "method_keywords": ["how", "method", "approach", "technique", "algorithm"],
        "finding_keywords": ["result", "finding", "conclusion", "discovered", "showed"],
        "summary_keywords": ["summary", "overview", "about", "main", "key points"]
    })


@dataclass
class CachingConfig:
    """Caching configuration"""
    enabled: bool = True
    backend: str = "file"  # file, redis, memory
    directory: str = "./data/cache"
    ttl_seconds: int = 3600
    max_size_mb: int = 1024
    eviction_policy: str = "lru"


@dataclass
class ErrorHandlingConfig:
    """Error handling and retry configuration"""
    max_retries: int = 3
    backoff_factor: float = 2.0
    retry_on_status: List[int] = field(default_factory=lambda: [429, 500, 502, 503, 504])
    circuit_breaker: Dict[str, Any] = field(default_factory=lambda: {
        "failure_threshold": 5,
        "recovery_timeout": 30
    })


@dataclass
class LoggingConfig:
    """Logging configuration"""
    level: str = "INFO"
    format: str = "structured"  # structured, simple
    output: str = "console"  # console, file, both
    file_path: str = "./logs/rag_system.log"
    max_file_size_mb: int = 100
    backup_count: int = 5
    include_request_ids: bool = True


@dataclass
class MonitoringConfig:
    """Monitoring and metrics configuration"""
    enabled: bool = True
    metrics_port: int = 8080
    health_check_enabled: bool = True
    performance_tracking: Dict[str, bool] = field(default_factory=lambda: {
        "track_latency": True,
        "track_token_usage": True,
        "track_error_rates": True,
        "export_to_prometheus": False
    })


@dataclass
class CostControlConfig:
    """Cost control configuration"""
    daily_budget_usd: float = 50.0
    per_request_budget_usd: float = 1.0
    token_budget_per_day: int = 1000000
    alert_threshold_percent: int = 80
    auto_shutdown_on_budget_exceeded: bool = False


@dataclass
class DomainConfig:
    """Domain-specific configuration"""
    definitions_file: str = "./config/domains.yaml"
    auto_detection: bool = True
    default_domain: str = "general_research"
    concept_extraction: Dict[str, Any] = field(default_factory=lambda: {
        "max_concepts_per_type": 5,
        "confidence_threshold": 0.8
    })


@dataclass
class APIConfig:
    """API service configuration"""
    host: str = "0.0.0.0"
    port: int = 8000
    cors_enabled: bool = True
    rate_limiting: Dict[str, Any] = field(default_factory=lambda: {
        "enabled": True,
        "requests_per_minute": 100
    })
    authentication: Dict[str, Any] = field(default_factory=lambda: {
        "enabled": False,
        "type": "api_key"
    })


@dataclass
class TestingConfig:
    """Testing configuration"""
    ground_truth_file: str = "./tests/data/ground_truth.json"
    evaluation_metrics: List[str] = field(default_factory=lambda: ["recall@k", "precision@k", "mrr", "faithfulness"])
    benchmark_papers: List[str] = field(default_factory=lambda: ["./tests/data/sample_papers/"])
    automated_evaluation: bool = True


@dataclass
class RAGConfig:
    """Complete RAG system configuration"""
    system: SystemConfig = field(default_factory=SystemConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    embeddings: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    document_processing: DocumentProcessingConfig = field(default_factory=DocumentProcessingConfig)
    vector_store: VectorStoreConfig = field(default_factory=VectorStoreConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    caching: CachingConfig = field(default_factory=CachingConfig)
    error_handling: ErrorHandlingConfig = field(default_factory=ErrorHandlingConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    monitoring: MonitoringConfig = field(default_factory=MonitoringConfig)
    cost_control: CostControlConfig = field(default_factory=CostControlConfig)
    domains: DomainConfig = field(default_factory=DomainConfig)
    api: APIConfig = field(default_factory=APIConfig)
    testing: TestingConfig = field(default_factory=TestingConfig)


class ConfigManager:
    """Manages configuration loading, validation, and CLI argument parsing"""
    
    def __init__(self, config_file: Optional[str] = None):
        self.config_file = config_file or "./config/config.yaml"
        self.config: Optional[RAGConfig] = None
        self._cli_overrides: Dict[str, Any] = {}
    
    def load_config(self, config_file: Optional[str] = None) -> RAGConfig:
        """Load configuration from YAML file"""
        config_path = Path(config_file or self.config_file)
        
        if not config_path.exists():
            # Create default config file
            self._create_default_config(config_path)
        
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config_data = yaml.safe_load(f)
            
            # Apply environment variable overrides
            config_data = self._apply_env_overrides(config_data)
            
            # Apply CLI overrides
            config_data = self._apply_cli_overrides(config_data)
            
            # Convert to structured config
            self.config = self._dict_to_config(config_data)
            
            # Validate configuration
            self._validate_config(self.config)
            
            return self.config
            
        except Exception as e:
            raise ConfigurationError(f"Failed to load configuration from {config_path}: {e}")
    
    def save_config(self, config: RAGConfig, config_file: Optional[str] = None) -> None:
        """Save configuration to YAML file"""
        config_path = Path(config_file or self.config_file)
        config_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            config_dict = self._config_to_dict(config)
            
            with open(config_path, 'w', encoding='utf-8') as f:
                yaml.dump(config_dict, f, default_flow_style=False, indent=2)
                
        except Exception as e:
            raise ConfigurationError(f"Failed to save configuration to {config_path}: {e}")
    
    def parse_cli_args(self, args: Optional[List[str]] = None) -> argparse.Namespace:
        """Parse CLI arguments and store overrides"""
        parser = argparse.ArgumentParser(description="Enhanced RAG System")
        
        # General options
        parser.add_argument('--config', type=str, help='Configuration file path')
        parser.add_argument('--environment', type=str, choices=['development', 'staging', 'production'],
                          help='Environment override')
        
        # LLM options
        parser.add_argument('--llm-provider', type=str, help='LLM provider (openai, anthropic)')
        parser.add_argument('--llm-model', type=str, help='LLM model name')
        parser.add_argument('--temperature', type=float, help='LLM temperature')
        parser.add_argument('--max-tokens', type=int, help='Maximum tokens per request')
        
        # Embedding options  
        parser.add_argument('--embedding-provider', type=str, help='Embedding provider')
        parser.add_argument('--embedding-model', type=str, help='Embedding model name')
        
        # Processing options
        parser.add_argument('--chunk-size', type=int, help='Document chunk size')
        parser.add_argument('--chunk-overlap', type=int, help='Document chunk overlap')
        parser.add_argument('--split-strategy', type=str, 
                          choices=['fixed_size', 'semantic', 'section_aware', 'adaptive'],
                          help='Document splitting strategy')
        
        # Retrieval options
        parser.add_argument('--retrieval-k', type=int, help='Number of documents to retrieve')
        parser.add_argument('--similarity-threshold', type=float, help='Similarity threshold')
        
        # Cost control
        parser.add_argument('--daily-budget', type=float, help='Daily budget in USD')
        parser.add_argument('--per-request-budget', type=float, help='Per request budget in USD')
        
        # API options
        parser.add_argument('--api-host', type=str, help='API host')
        parser.add_argument('--api-port', type=int, help='API port')
        
        # Modes
        parser.add_argument('--lite-mode', action='store_true', help='Enable lite mode (faster, less comprehensive)')
        parser.add_argument('--full-mode', action='store_true', help='Enable full mode (comprehensive analysis)')
        
        # Parse arguments
        parsed_args = parser.parse_args(args)
        
        # Store CLI overrides
        self._cli_overrides = self._extract_cli_overrides(parsed_args)
        
        return parsed_args
    
    def get_config(self) -> RAGConfig:
        """Get current configuration"""
        if self.config is None:
            self.config = self.load_config()
        return self.config
    
    def update_config(self, updates: Dict[str, Any]) -> None:
        """Update configuration with new values"""
        if self.config is None:
            self.config = self.load_config()
        
        # Apply updates
        config_dict = self._config_to_dict(self.config)
        config_dict = self._deep_update(config_dict, updates)
        self.config = self._dict_to_config(config_dict)
        
        # Validate updated configuration
        self._validate_config(self.config)
    
    def _create_default_config(self, config_path: Path) -> None:
        """Create default configuration file"""
        config_path.parent.mkdir(parents=True, exist_ok=True)
        
        default_config = RAGConfig()
        self.save_config(default_config, str(config_path))
    
    def _apply_env_overrides(self, config_data: Dict[str, Any]) -> Dict[str, Any]:
        """Apply environment variable overrides"""
        env_mappings = {
            'OPENAI_API_KEY': ['llm', 'api_key'],
            'OPENAI_EMBEDDING_API_KEY': ['embeddings', 'api_key'],
            'RAG_ENVIRONMENT': ['system', 'environment'],
            'RAG_LOG_LEVEL': ['logging', 'level'],
            'RAG_CHUNK_SIZE': ['document_processing', 'chunk_size'],
            'RAG_DAILY_BUDGET': ['cost_control', 'daily_budget_usd'],
            'RAG_API_PORT': ['api', 'port']
        }
        
        for env_var, config_path in env_mappings.items():
            value = os.environ.get(env_var)
            if value is not None:
                # Convert value to appropriate type
                if config_path[-1] in ['chunk_size', 'port', 'daily_budget_usd']:
                    try:
                        value = int(value) if config_path[-1] != 'daily_budget_usd' else float(value)
                    except ValueError:
                        continue
                
                # Apply to config
                current = config_data
                for key in config_path[:-1]:
                    if key not in current:
                        current[key] = {}
                    current = current[key]
                current[config_path[-1]] = value
        
        return config_data
    
    def _apply_cli_overrides(self, config_data: Dict[str, Any]) -> Dict[str, Any]:
        """Apply CLI argument overrides"""
        return self._deep_update(config_data, self._cli_overrides)
    
    def _extract_cli_overrides(self, args: argparse.Namespace) -> Dict[str, Any]:
        """Extract CLI overrides from parsed arguments"""
        overrides = {}
        
        # Map CLI arguments to config paths
        arg_mappings = {
            'environment': ['system', 'environment'],
            'llm_provider': ['llm', 'provider'],
            'llm_model': ['llm', 'model'],
            'temperature': ['llm', 'temperature'],
            'max_tokens': ['llm', 'max_tokens'],
            'embedding_provider': ['embeddings', 'provider'],
            'embedding_model': ['embeddings', 'model'],
            'chunk_size': ['document_processing', 'chunk_size'],
            'chunk_overlap': ['document_processing', 'chunk_overlap'],
            'split_strategy': ['document_processing', 'split_strategy'],
            'retrieval_k': ['retrieval', 'default_k'],
            'similarity_threshold': ['retrieval', 'similarity_threshold'],
            'daily_budget': ['cost_control', 'daily_budget_usd'],
            'per_request_budget': ['cost_control', 'per_request_budget_usd'],
            'api_host': ['api', 'host'],
            'api_port': ['api', 'port']
        }
        
        for arg_name, config_path in arg_mappings.items():
            value = getattr(args, arg_name, None)
            if value is not None:
                current = overrides
                for key in config_path[:-1]:
                    if key not in current:
                        current[key] = {}
                    current = current[key]
                current[config_path[-1]] = value
        
        # Handle mode flags
        if getattr(args, 'lite_mode', False):
            overrides.setdefault('system', {})['mode'] = 'lite'
        elif getattr(args, 'full_mode', False):
            overrides.setdefault('system', {})['mode'] = 'full'
        
        return overrides
    
    def _dict_to_config(self, config_dict: Dict[str, Any]) -> RAGConfig:
        """Convert dictionary to structured configuration"""
        try:
            return RAGConfig(
                system=SystemConfig(**config_dict.get('system', {})),
                llm=LLMConfig(**config_dict.get('llm', {})),
                embeddings=EmbeddingConfig(**config_dict.get('embeddings', {})),
                document_processing=DocumentProcessingConfig(**config_dict.get('document_processing', {})),
                vector_store=VectorStoreConfig(**config_dict.get('vector_store', {})),
                retrieval=RetrievalConfig(**config_dict.get('retrieval', {})),
                caching=CachingConfig(**config_dict.get('caching', {})),
                error_handling=ErrorHandlingConfig(**config_dict.get('error_handling', {})),
                logging=LoggingConfig(**config_dict.get('logging', {})),
                monitoring=MonitoringConfig(**config_dict.get('monitoring', {})),
                cost_control=CostControlConfig(**config_dict.get('cost_control', {})),
                domains=DomainConfig(**config_dict.get('domains', {})),
                api=APIConfig(**config_dict.get('api', {})),
                testing=TestingConfig(**config_dict.get('testing', {}))
            )
        except Exception as e:
            raise ConfigurationError(f"Failed to parse configuration: {e}")
    
    def _config_to_dict(self, config: RAGConfig) -> Dict[str, Any]:
        """Convert structured configuration to dictionary"""
        import dataclasses
        
        result = {}
        for field in dataclasses.fields(config):
            value = getattr(config, field.name)
            if dataclasses.is_dataclass(value):
                result[field.name] = dataclasses.asdict(value)
            else:
                result[field.name] = value
        
        return result
    
    def _deep_update(self, base_dict: Dict[str, Any], update_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Deep update of nested dictionaries"""
        result = deepcopy(base_dict)
        
        for key, value in update_dict.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_update(result[key], value)
            else:
                result[key] = value
        
        return result
    
    def _validate_config(self, config: RAGConfig) -> None:
        """Validate configuration values"""
        # Validate API keys
        if not config.llm.api_key and not os.environ.get('OPENAI_API_KEY'):
            raise ConfigurationError("LLM API key not provided")
        
        # Validate chunk sizes
        if config.document_processing.chunk_size <= 0:
            raise ConfigurationError("Chunk size must be positive")
        
        if config.document_processing.chunk_overlap >= config.document_processing.chunk_size:
            raise ConfigurationError("Chunk overlap must be less than chunk size")
        
        # Validate retrieval parameters
        if config.retrieval.default_k <= 0:
            raise ConfigurationError("Retrieval k must be positive")
        
        if not 0 <= config.retrieval.similarity_threshold <= 1:
            raise ConfigurationError("Similarity threshold must be between 0 and 1")
        
        # Validate budgets
        if config.cost_control.daily_budget_usd <= 0:
            raise ConfigurationError("Daily budget must be positive")
        
        # Validate paths
        for path_field in ['domains.definitions_file', 'testing.ground_truth_file']:
            path_value = self._get_nested_value(config, path_field)
            if path_value and not Path(path_value).parent.exists():
                Path(path_value).parent.mkdir(parents=True, exist_ok=True)
    
    def _get_nested_value(self, obj: Any, path: str) -> Any:
        """Get nested value using dot notation"""
        parts = path.split('.')
        current = obj
        
        for part in parts:
            if hasattr(current, part):
                current = getattr(current, part)
            else:
                return None
        
        return current


# Global configuration manager
_global_config_manager = None


def get_config_manager(config_file: Optional[str] = None) -> ConfigManager:
    """Get global configuration manager"""
    global _global_config_manager
    
    if _global_config_manager is None:
        _global_config_manager = ConfigManager(config_file)
    
    return _global_config_manager


def get_config() -> RAGConfig:
    """Get current configuration"""
    return get_config_manager().get_config()