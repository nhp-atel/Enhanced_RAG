"""
Centralized configuration management with validation.
"""
from typing import Dict, List, Optional
from pydantic import BaseSettings, validator
from pathlib import Path
import logging


class RAGConfig(BaseSettings):
    """Main configuration for RAG system."""
    
    # API Keys
    openai_api_key: str
    langsmith_api_key: Optional[str] = None
    
    # Model Configuration
    embedding_model: str = "text-embedding-3-small"
    chat_model: str = "gpt-4o-mini"
    temperature: float = 0.0
    max_tokens: Optional[int] = None
    
    # Document Processing
    chunk_size: int = 800
    chunk_overlap: int = 150
    max_pages_for_metadata: int = 3
    
    # Vector Store
    faiss_index_type: str = "flat"  # flat, ivf, hnsw
    similarity_search_k: int = 6
    
    # Performance
    batch_size: int = 10
    enable_caching: bool = True
    cache_ttl_seconds: int = 3600
    max_concurrent_requests: int = 5
    
    # Logging
    log_level: str = "INFO"
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # Storage
    vector_store_path: Path = Path("./data/vector_store")
    cache_dir: Path = Path("./data/cache")
    
    @validator('chunk_size')
    def validate_chunk_size(cls, v):
        if v < 100 or v > 2000:
            raise ValueError('chunk_size must be between 100 and 2000')
        return v
    
    @validator('chunk_overlap')
    def validate_overlap(cls, v, values):
        if 'chunk_size' in values and v >= values['chunk_size']:
            raise ValueError('chunk_overlap must be less than chunk_size')
        return v
    
    @validator('log_level')
    def validate_log_level(cls, v):
        valid_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
        if v.upper() not in valid_levels:
            raise ValueError(f'log_level must be one of {valid_levels}')
        return v.upper()
    
    class Config:
        env_file = ".env"
        env_prefix = "RAG_"


class PromptConfig:
    """Centralized prompt templates with versioning."""
    
    METADATA_EXTRACTION_V1 = """
    You are an expert at extracting metadata from academic papers.
    
    Extract the following information from the paper text:
    - Title (full title of the paper)
    - Authors (list all authors)
    - Institutions/Affiliations
    - Publication Date/Year
    - ArXiv ID or DOI (if present)
    - Keywords
    - Abstract
    
    Return valid JSON with confidence score.
    
    Paper text: {text}
    """
    
    CONCEPT_EXTRACTION_V1 = """
    Extract key concepts from this research paper summary.
    
    Categories to extract:
    1. technical_terms: Important algorithms, models, methods
    2. key_concepts: Core theoretical frameworks
    3. methodologies: Experimental approaches
    4. findings: Key results and conclusions
    5. entities: Important names, datasets, systems
    
    Summary: {summary}
    """
    
    RAG_SYSTEM_V1 = """
    You are an expert research assistant with access to multiple knowledge sources.
    
    Context includes:
    - Paper metadata (titles, authors, dates)
    - Paper summary (comprehensive overview)
    - Relevant concepts (definitions and explanations)
    - Document content (specific passages)
    
    Guidelines:
    1. For factual questions, prioritize PAPER METADATA
    2. For definitions, use RELEVANT CONCEPTS and PAPER SUMMARY
    3. For detailed information, integrate DOCUMENT CONTENT
    4. Provide comprehensive yet concise answers
    5. If information spans multiple sources, synthesize coherently
    
    Context: {context}
    Question: {question}
    """


def setup_logging(config: RAGConfig) -> None:
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, config.log_level),
        format=config.log_format,
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('rag_system.log')
        ]
    )


# Usage
def get_config() -> RAGConfig:
    """Get validated configuration."""
    try:
        config = RAGConfig()
        setup_logging(config)
        return config
    except Exception as e:
        print(f"Configuration error: {e}")
        raise