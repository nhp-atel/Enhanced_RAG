"""
FastAPI server for RAG system deployment.
"""

import sys
import time
from typing import List, Optional, Dict, Any
from pathlib import Path

try:
    from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import JSONResponse
    from pydantic import BaseModel, Field
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False
    # Create dummy classes for type hints
    class BaseModel: pass
    class FastAPI: pass

from ..core.pipeline import RAGPipeline, create_pipeline
from ..utils.config import RAGConfig
from ..utils.logging import get_logger, RequestTracker
from ..utils.errors import RAGError, format_error_response


# Request/Response Models
class ProcessDocumentRequest(BaseModel):
    source: str = Field(..., description="Document source (URL or file path)")
    document_id: Optional[str] = Field(None, description="Custom document ID")
    split_strategy: Optional[str] = Field(None, description="Document splitting strategy")
    save_index: Optional[bool] = Field(False, description="Save index after processing")


class ProcessDocumentResponse(BaseModel):
    status: str
    document_id: Optional[str] = None
    chunks: Optional[int] = None
    processing_time_ms: Optional[int] = None
    title: Optional[str] = None
    authors: Optional[List[str]] = None
    error: Optional[str] = None


class QueryRequest(BaseModel):
    question: str = Field(..., description="Question to ask")
    k: Optional[int] = Field(6, description="Number of documents to retrieve")
    strategy: Optional[str] = Field("enhanced", description="Query strategy")
    include_sources: Optional[bool] = Field(True, description="Include source documents")


class QueryResponse(BaseModel):
    question: str
    answer: str
    sources: Optional[List[Dict[str, Any]]] = None
    processing_time_ms: int
    tokens_used: int
    cost_usd: float
    metadata: Dict[str, Any]


class HealthResponse(BaseModel):
    status: str
    timestamp: float
    components: Dict[str, Any]
    version: str = "1.0.0"


class StatsResponse(BaseModel):
    pipeline_stats: Dict[str, Any]
    system_info: Dict[str, Any]
    timestamp: float


# Global pipeline instance
_pipeline_instance: Optional[RAGPipeline] = None


def get_pipeline() -> RAGPipeline:
    """Get or create pipeline instance"""
    global _pipeline_instance
    
    if _pipeline_instance is None:
        _pipeline_instance = create_pipeline()
    
    return _pipeline_instance


def create_app(config: Optional[RAGConfig] = None) -> FastAPI:
    """Create FastAPI application"""
    if not FASTAPI_AVAILABLE:
        raise ImportError("FastAPI and uvicorn required for API server")
    
    # Initialize app
    app = FastAPI(
        title="Enhanced RAG System API",
        description="Production-ready RAG system with modular architecture",
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc"
    )
    
    # Configure CORS
    if config and config.api.cors_enabled:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],  # Configure appropriately for production
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
    
    # Initialize logger
    logger = get_logger("api_server")
    
    # Initialize pipeline
    global _pipeline_instance
    if config:
        _pipeline_instance = RAGPipeline(config)
    
    @app.middleware("http")
    async def request_logging_middleware(request, call_next):
        """Log all requests"""
        start_time = time.time()
        
        with RequestTracker(logger) as tracker:
            response = await call_next(request)
            
            processing_time = int((time.time() - start_time) * 1000)
            logger.info("API request completed",
                       method=request.method,
                       url=str(request.url),
                       status_code=response.status_code,
                       processing_time_ms=processing_time)
        
        return response
    
    @app.exception_handler(RAGError)
    async def rag_error_handler(request, exc: RAGError):
        """Handle RAG-specific errors"""
        logger.error("RAG error occurred",
                    error_code=exc.error_code,
                    error_message=exc.message,
                    url=str(request.url))
        
        return JSONResponse(
            status_code=400,
            content=format_error_response(exc)
        )
    
    @app.exception_handler(Exception)
    async def general_error_handler(request, exc: Exception):
        """Handle general errors"""
        logger.error("Unhandled error occurred",
                    error_type=type(exc).__name__,
                    error_message=str(exc),
                    url=str(request.url))
        
        return JSONResponse(
            status_code=500,
            content={
                "error": {
                    "message": "Internal server error",
                    "code": "INTERNAL_ERROR"
                }
            }
        )
    
    # API Endpoints
    
    @app.get("/health", response_model=HealthResponse)
    async def health_check():
        """Health check endpoint"""
        try:
            pipeline = get_pipeline()
            health_data = pipeline.health_check()
            
            return HealthResponse(
                status=health_data["overall"],
                timestamp=health_data["timestamp"],
                components=health_data["components"]
            )
        except Exception as e:
            logger.error("Health check failed", error=str(e))
            return HealthResponse(
                status="unhealthy",
                timestamp=time.time(),
                components={"error": str(e)}
            )
    
    @app.get("/stats", response_model=StatsResponse)
    async def get_stats():
        """Get system statistics"""
        try:
            pipeline = get_pipeline()
            stats = pipeline.get_stats()
            
            return StatsResponse(
                pipeline_stats=stats,
                system_info={
                    "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
                    "platform": sys.platform
                },
                timestamp=time.time()
            )
        except Exception as e:
            logger.error("Failed to get stats", error=str(e))
            raise HTTPException(status_code=500, detail="Failed to get statistics")
    
    @app.post("/process", response_model=ProcessDocumentResponse)
    async def process_document(request: ProcessDocumentRequest, background_tasks: BackgroundTasks):
        """Process a document"""
        try:
            pipeline = get_pipeline()
            
            # Convert split strategy
            split_strategy = None
            if request.split_strategy:
                from ..core.splitter import SplitStrategy
                split_strategy = SplitStrategy(request.split_strategy)
            
            # Process document
            result = pipeline.process_document(
                source=request.source,
                document_id=request.document_id,
                split_strategy=split_strategy
            )
            
            # Save index in background if requested
            if request.save_index and result.success:
                background_tasks.add_task(
                    pipeline.save_index, 
                    "./data/indices/api_auto_save"
                )
            
            if result.success:
                return ProcessDocumentResponse(
                    status="success",
                    document_id=result.document_id,
                    chunks=result.chunk_count,
                    processing_time_ms=result.processing_time_ms,
                    title=result.metadata.title if result.metadata else None,
                    authors=result.metadata.authors if result.metadata else None
                )
            else:
                return ProcessDocumentResponse(
                    status="error",
                    error=result.error_message,
                    processing_time_ms=result.processing_time_ms
                )
                
        except Exception as e:
            logger.error("Document processing failed", error=str(e))
            raise HTTPException(status_code=400, detail=str(e))
    
    @app.post("/query", response_model=QueryResponse)
    async def query_documents(request: QueryRequest):
        """Query the RAG system"""
        try:
            pipeline = get_pipeline()
            
            result = pipeline.query(
                question=request.question,
                k=request.k,
                strategy=request.strategy
            )
            
            # Format sources
            sources = None
            if request.include_sources:
                sources = [
                    {
                        "content": doc.page_content[:500] + "..." if len(doc.page_content) > 500 else doc.page_content,
                        "metadata": doc.metadata,
                        "similarity_score": doc.metadata.get("similarity_score")
                    }
                    for doc in result.sources
                ]
            
            return QueryResponse(
                question=request.question,
                answer=result.answer,
                sources=sources,
                processing_time_ms=result.processing_time_ms,
                tokens_used=result.tokens_used,
                cost_usd=result.cost_usd,
                metadata=result.metadata
            )
            
        except Exception as e:
            logger.error("Query processing failed", error=str(e))
            raise HTTPException(status_code=400, detail=str(e))
    
    @app.post("/load-index")
    async def load_index(index_path: str):
        """Load vector index from specified path"""
        try:
            pipeline = get_pipeline()
            success = pipeline.load_index(index_path)
            
            if success:
                return {"status": "success", "message": f"Index loaded from {index_path}"}
            else:
                raise HTTPException(status_code=400, detail="Failed to load index")
                
        except Exception as e:
            logger.error("Index loading failed", error=str(e))
            raise HTTPException(status_code=400, detail=str(e))
    
    @app.post("/save-index")
    async def save_index(index_path: str):
        """Save vector index to specified path"""
        try:
            pipeline = get_pipeline()
            success = pipeline.save_index(index_path)
            
            if success:
                return {"status": "success", "message": f"Index saved to {index_path}"}
            else:
                raise HTTPException(status_code=400, detail="Failed to save index")
                
        except Exception as e:
            logger.error("Index saving failed", error=str(e))
            raise HTTPException(status_code=400, detail=str(e))
    
    @app.delete("/cache")
    async def clear_cache():
        """Clear system cache"""
        try:
            pipeline = get_pipeline()
            success = pipeline.clear_cache()
            
            return {"status": "success", "cache_cleared": success}
            
        except Exception as e:
            logger.error("Cache clearing failed", error=str(e))
            raise HTTPException(status_code=500, detail=str(e))
    
    # Development endpoints
    if config and config.system.environment == "development":
        
        @app.get("/config")
        async def get_config():
            """Get current configuration (development only)"""
            try:
                from ..utils.config import get_config_manager
                config_manager = get_config_manager()
                config_dict = config_manager._config_to_dict(config_manager.get_config())
                return config_dict
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
    
    logger.info("FastAPI application created successfully")
    return app


# For direct uvicorn usage
app = create_app()