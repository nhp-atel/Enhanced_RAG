"""
RAG Pipeline orchestrator - coordinates all components with dependency injection.
"""

import time
from typing import Dict, Any, List, Optional, Union
from pathlib import Path
from dataclasses import dataclass

from ..interfaces import (
    LLMClient, EmbeddingClient, VectorStore, CacheInterface, 
    MetricsInterface, LoggerInterface, create_llm_client, 
    create_embedding_client, create_vector_store
)
from ..core.ingest import DocumentIngestor, DocumentMetadata
from ..core.splitter import DocumentSplitter, SplitStrategy
from ..utils.config import RAGConfig, get_config
from ..utils.logging import get_logger, RequestTracker, timed_operation
from ..utils.errors import ProcessingError, ValidationError
from ..utils.prompts import get_prompt_manager

from langchain_core.documents import Document


@dataclass
class PipelineResult:
    """Result from pipeline processing"""
    success: bool
    document_id: str
    metadata: Optional[DocumentMetadata] = None
    chunk_count: int = 0
    processing_time_ms: int = 0
    error_message: Optional[str] = None
    vector_store_stats: Optional[Dict[str, Any]] = None


@dataclass
class QueryResult:
    """Result from query processing"""
    answer: str
    sources: List[Document]
    metadata: Dict[str, Any]
    processing_time_ms: int
    tokens_used: int
    cost_usd: float


class RAGPipeline:
    """Main RAG pipeline orchestrator with dependency injection"""
    
    def __init__(self, config: Optional[RAGConfig] = None):
        # Load configuration
        self.config = config or get_config()
        
        # Initialize logger
        self.logger = get_logger("rag_pipeline")
        
        # Initialize components with dependency injection
        self._initialize_components()
        
        # Pipeline state
        self.is_initialized = False
        self.vector_store_loaded = False
        
        self.logger.info("RAG Pipeline initialized", 
                        llm_provider=self.config.llm.provider,
                        embedding_provider=self.config.embeddings.provider,
                        vector_store_provider=self.config.vector_store.provider)
    
    def _initialize_components(self) -> None:
        """Initialize all pipeline components"""
        try:
            # Create LLM client
            self.llm_client = create_llm_client({
                **self.config.llm.__dict__,
                'api_key': self.config.llm.api_key or self._get_api_key('llm')
            })
            
            # Create embedding client
            self.embedding_client = create_embedding_client({
                **self.config.embeddings.__dict__,
                'api_key': self.config.embeddings.api_key or self._get_api_key('embeddings')
            })
            
            # Create vector store
            self.vector_store = create_vector_store(
                self.config.vector_store.__dict__,
                self.embedding_client
            )
            
            # Create document ingestor
            self.document_ingestor = DocumentIngestor(
                llm_client=self.llm_client,
                logger=self.logger.with_context(component="ingestor"),
                metrics=self._create_metrics_client(),
                config=self.config.document_processing.__dict__
            )
            
            # Create document splitter
            self.document_splitter = DocumentSplitter(
                llm_client=self.llm_client,
                logger=self.logger.with_context(component="splitter"),
                metrics=self._create_metrics_client(),
                config=self.config.document_processing.__dict__
            )
            
            # Create cache client
            self.cache_client = self._create_cache_client()
            
            # Get prompt manager
            self.prompt_manager = get_prompt_manager()
            
            self.is_initialized = True
            
        except Exception as e:
            self.logger.error("Failed to initialize pipeline components", error=str(e))
            raise ProcessingError(f"Pipeline initialization failed: {e}")
    
    def process_document(
        self, 
        source: Union[str, Path], 
        document_id: Optional[str] = None,
        split_strategy: Optional[SplitStrategy] = None
    ) -> PipelineResult:
        """
        Process a document through the complete RAG pipeline
        
        Args:
            source: Document source (URL or file path)
            document_id: Optional custom document ID
            split_strategy: Optional splitting strategy override
            
        Returns:
            Pipeline processing result
        """
        start_time = time.time()
        
        with RequestTracker(self.logger) as tracker:
            try:
                self.logger.info("Starting document processing", source=str(source))
                
                # Validate input
                if not source:
                    raise ValidationError("Document source cannot be empty")
                
                # Step 1: Ingest document
                tracker.log_milestone("Starting document ingestion")
                with timed_operation(self.logger, "document_ingestion"):
                    documents, metadata = self.document_ingestor.ingest_document(
                        source, document_id
                    )
                
                # Step 2: Split documents
                tracker.log_milestone("Starting document splitting", chunk_count=len(documents))
                with timed_operation(self.logger, "document_splitting"):
                    document_chunks = self.document_splitter.split_documents(
                        documents, strategy=split_strategy
                    )
                
                # Step 3: Generate embeddings and store
                tracker.log_milestone("Starting vector storage", chunk_count=len(document_chunks))
                with timed_operation(self.logger, "vector_storage"):
                    vector_ids = self.vector_store.add_documents(document_chunks)
                
                # Step 4: Cache results if enabled
                if self.config.caching.enabled and self.cache_client:
                    tracker.log_milestone("Caching results")
                    self._cache_document_metadata(metadata)
                
                # Calculate processing time
                processing_time_ms = int((time.time() - start_time) * 1000)
                
                # Get vector store stats
                vector_stats = self.vector_store.get_stats()
                
                result = PipelineResult(
                    success=True,
                    document_id=metadata.content_hash or document_id or "unknown",
                    metadata=metadata,
                    chunk_count=len(document_chunks),
                    processing_time_ms=processing_time_ms,
                    vector_store_stats=vector_stats
                )
                
                self.logger.info("Document processing completed successfully",
                               document_id=result.document_id,
                               chunk_count=result.chunk_count,
                               processing_time_ms=processing_time_ms)
                
                self.vector_store_loaded = True
                return result
                
            except Exception as e:
                processing_time_ms = int((time.time() - start_time) * 1000)
                
                result = PipelineResult(
                    success=False,
                    document_id=document_id or "unknown",
                    processing_time_ms=processing_time_ms,
                    error_message=str(e)
                )
                
                self.logger.error("Document processing failed",
                                error=str(e),
                                processing_time_ms=processing_time_ms)
                
                return result
    
    def query(
        self, 
        question: str, 
        k: Optional[int] = None,
        strategy: str = "enhanced",
        include_metadata: bool = True
    ) -> QueryResult:
        """
        Query the RAG system
        
        Args:
            question: User question
            k: Number of documents to retrieve
            strategy: Query strategy ("basic" or "enhanced")
            include_metadata: Whether to include metadata in response
            
        Returns:
            Query result with answer and metadata
        """
        start_time = time.time()
        
        if not self.vector_store_loaded:
            raise ProcessingError("No documents have been processed. Please process a document first.")
        
        with RequestTracker(self.logger) as tracker:
            try:
                self.logger.info("Starting query processing", 
                               question=question[:100] + "..." if len(question) > 100 else question,
                               strategy=strategy)
                
                # Step 1: Retrieve relevant documents
                tracker.log_milestone("Starting document retrieval")
                with timed_operation(self.logger, "document_retrieval"):
                    retrieved_docs = self._retrieve_documents(question, k or self.config.retrieval.default_k)
                
                # Step 2: Generate response
                tracker.log_milestone("Starting response generation", retrieved_docs=len(retrieved_docs))
                with timed_operation(self.logger, "response_generation"):
                    answer, response_metadata = self._generate_response(
                        question, retrieved_docs, strategy
                    )
                
                # Calculate metrics
                processing_time_ms = int((time.time() - start_time) * 1000)
                
                result = QueryResult(
                    answer=answer,
                    sources=retrieved_docs,
                    metadata={
                        **response_metadata,
                        "retrieval_count": len(retrieved_docs),
                        "strategy": strategy,
                        "include_metadata": include_metadata
                    },
                    processing_time_ms=processing_time_ms,
                    tokens_used=response_metadata.get("tokens_used", 0),
                    cost_usd=response_metadata.get("cost_usd", 0.0)
                )
                
                self.logger.info("Query processing completed",
                               tokens_used=result.tokens_used,
                               cost_usd=result.cost_usd,
                               processing_time_ms=processing_time_ms)
                
                return result
                
            except Exception as e:
                processing_time_ms = int((time.time() - start_time) * 1000)
                
                self.logger.error("Query processing failed",
                                error=str(e),
                                processing_time_ms=processing_time_ms)
                
                raise ProcessingError(f"Query processing failed: {e}")
    
    def _retrieve_documents(self, query: str, k: int) -> List[Document]:
        """Retrieve relevant documents using vector search"""
        try:
            # Check cache first
            if self.config.caching.enabled and self.cache_client:
                cache_key = f"retrieval:{hash(query)}:{k}"
                cached_docs = self.cache_client.get(cache_key)
                if cached_docs:
                    self.logger.debug("Retrieved documents from cache", query_hash=hash(query))
                    return cached_docs
            
            # Perform vector search
            docs = self.vector_store.similarity_search(query, k=k)
            
            # Cache results
            if self.config.caching.enabled and self.cache_client:
                self.cache_client.set(cache_key, docs, ttl=self.config.caching.ttl_seconds)
            
            return docs
            
        except Exception as e:
            raise ProcessingError(f"Document retrieval failed: {e}")
    
    def _generate_response(
        self, 
        question: str, 
        documents: List[Document], 
        strategy: str
    ) -> tuple[str, Dict[str, Any]]:
        """Generate response using LLM"""
        try:
            # Prepare context
            context_text = "\n\n".join([doc.page_content for doc in documents])
            
            # Select prompt based on strategy
            prompt_name = "enhanced_rag" if strategy == "enhanced" else "basic_rag"
            
            # Format prompt
            messages = self.prompt_manager.format_prompt(
                prompt_name,
                question=question,
                context=context_text
            )
            
            # Generate response
            response = self.llm_client.generate_with_retry(messages)
            
            # Prepare metadata
            metadata = {
                "tokens_used": response.tokens_used,
                "cost_usd": response.cost_usd,
                "model": response.model,
                "latency_ms": response.latency_ms,
                "context_length": len(context_text),
                "document_count": len(documents)
            }
            
            return response.content, metadata
            
        except Exception as e:
            raise ProcessingError(f"Response generation failed: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get pipeline statistics"""
        stats = {
            "initialized": self.is_initialized,
            "vector_store_loaded": self.vector_store_loaded,
            "config": {
                "llm_provider": self.config.llm.provider,
                "llm_model": self.config.llm.model,
                "embedding_provider": self.config.embeddings.provider,
                "embedding_model": self.config.embeddings.model,
                "chunk_size": self.config.document_processing.chunk_size,
                "retrieval_k": self.config.retrieval.default_k
            }
        }
        
        if self.vector_store_loaded:
            stats["vector_store"] = self.vector_store.get_stats()
        
        return stats
    
    def save_index(self, path: str) -> bool:
        """Save vector index to disk"""
        try:
            if not self.vector_store_loaded:
                raise ProcessingError("No vector store to save")
            
            success = self.vector_store.save_local(path)
            if success:
                self.logger.info("Vector index saved", path=path)
            else:
                self.logger.warning("Failed to save vector index", path=path)
            
            return success
            
        except Exception as e:
            self.logger.error("Error saving vector index", path=path, error=str(e))
            return False
    
    def load_index(self, path: str) -> bool:
        """Load vector index from disk"""
        try:
            success = self.vector_store.load_local(path)
            if success:
                self.vector_store_loaded = True
                self.logger.info("Vector index loaded", path=path)
            else:
                self.logger.warning("Failed to load vector index", path=path)
            
            return success
            
        except Exception as e:
            self.logger.error("Error loading vector index", path=path, error=str(e))
            return False
    
    def clear_cache(self) -> bool:
        """Clear all cached data"""
        try:
            if self.cache_client:
                success = self.cache_client.clear()
                self.logger.info("Cache cleared", success=success)
                return success
            return True
            
        except Exception as e:
            self.logger.error("Error clearing cache", error=str(e))
            return False
    
    def health_check(self) -> Dict[str, Any]:
        """Perform health check on all components"""
        health = {
            "overall": "healthy",
            "components": {},
            "timestamp": time.time()
        }
        
        # Check LLM client
        try:
            test_response = self.llm_client.generate([{"role": "user", "content": "test"}])
            health["components"]["llm"] = {
                "status": "healthy",
                "model": self.config.llm.model,
                "provider": self.config.llm.provider
            }
        except Exception as e:
            health["components"]["llm"] = {
                "status": "unhealthy",
                "error": str(e)
            }
            health["overall"] = "degraded"
        
        # Check embedding client
        try:
            test_embedding = self.embedding_client.embed_query("test")
            health["components"]["embeddings"] = {
                "status": "healthy",
                "model": self.config.embeddings.model,
                "dimensions": len(test_embedding)
            }
        except Exception as e:
            health["components"]["embeddings"] = {
                "status": "unhealthy",
                "error": str(e)
            }
            health["overall"] = "degraded"
        
        # Check vector store
        try:
            vector_stats = self.vector_store.get_stats()
            health["components"]["vector_store"] = {
                "status": "healthy",
                "loaded": self.vector_store_loaded,
                **vector_stats
            }
        except Exception as e:
            health["components"]["vector_store"] = {
                "status": "unhealthy",
                "error": str(e)
            }
            health["overall"] = "degraded"
        
        return health
    
    def _get_api_key(self, service: str) -> Optional[str]:
        """Get API key from environment"""
        import os
        
        if service == 'llm':
            return os.environ.get('OPENAI_API_KEY')
        elif service == 'embeddings':
            return os.environ.get('OPENAI_API_KEY')
        
        return None
    
    def _create_metrics_client(self) -> Optional[MetricsInterface]:
        """Create metrics client (placeholder)"""
        # TODO: Implement metrics client based on config
        return None
    
    def _create_cache_client(self) -> Optional[CacheInterface]:
        """Create cache client (placeholder)"""
        # TODO: Implement cache client based on config
        return None
    
    def _cache_document_metadata(self, metadata: DocumentMetadata) -> None:
        """Cache document metadata"""
        if self.cache_client and metadata.content_hash:
            cache_key = f"metadata:{metadata.content_hash}"
            self.cache_client.set(cache_key, metadata, ttl=self.config.caching.ttl_seconds * 24)  # Longer TTL for metadata


# Pipeline factory and convenience functions

def create_pipeline(config_file: Optional[str] = None) -> RAGPipeline:
    """Create RAG pipeline with configuration"""
    if config_file:
        from ..utils.config import ConfigManager
        config_manager = ConfigManager(config_file)
        config = config_manager.load_config()
        return RAGPipeline(config)
    else:
        return RAGPipeline()


def quick_setup(source: Union[str, Path], config_file: Optional[str] = None) -> RAGPipeline:
    """Quick setup: create pipeline and process document"""
    pipeline = create_pipeline(config_file)
    
    result = pipeline.process_document(source)
    if not result.success:
        raise ProcessingError(f"Failed to process document: {result.error_message}")
    
    return pipeline