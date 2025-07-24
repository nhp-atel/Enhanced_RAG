"""
Custom exception classes for the RAG system.
"""

from typing import Optional, Dict, Any


class RAGError(Exception):
    """Base exception class for RAG system"""
    
    def __init__(self, message: str, error_code: Optional[str] = None, details: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.details = details or {}


class ConfigurationError(RAGError):
    """Raised when there's a configuration issue"""
    
    def __init__(self, message: str, config_key: Optional[str] = None):
        super().__init__(message, "CONFIG_ERROR", {"config_key": config_key})


class ValidationError(RAGError):
    """Raised when input validation fails"""
    
    def __init__(self, message: str, field: Optional[str] = None, value: Optional[Any] = None):
        super().__init__(message, "VALIDATION_ERROR", {"field": field, "value": value})


class DocumentError(RAGError):
    """Raised when document processing fails"""
    
    def __init__(self, message: str, document_path: Optional[str] = None):
        super().__init__(message, "DOCUMENT_ERROR", {"document_path": document_path})


class ProcessingError(RAGError):
    """Raised when document processing fails"""
    
    def __init__(self, message: str, stage: Optional[str] = None):
        super().__init__(message, "PROCESSING_ERROR", {"stage": stage})


class APIError(RAGError):
    """Raised when external API calls fail"""
    
    def __init__(self, message: str, provider: Optional[str] = None, status_code: Optional[int] = None):
        super().__init__(message, "API_ERROR", {"provider": provider, "status_code": status_code})


class RateLimitError(APIError):
    """Raised when API rate limits are exceeded"""
    
    def __init__(self, message: str, provider: Optional[str] = None, retry_after: Optional[int] = None):
        super().__init__(message, provider, 429)
        self.error_code = "RATE_LIMIT_ERROR"
        self.details["retry_after"] = retry_after


class BudgetExceededError(RAGError):
    """Raised when cost budgets are exceeded"""
    
    def __init__(self, message: str, current_cost: Optional[float] = None, budget: Optional[float] = None):
        super().__init__(message, "BUDGET_EXCEEDED", {"current_cost": current_cost, "budget": budget})


class VectorStoreError(RAGError):
    """Raised when vector store operations fail"""
    
    def __init__(self, message: str, operation: Optional[str] = None):
        super().__init__(message, "VECTOR_STORE_ERROR", {"operation": operation})


class RetrievalError(RAGError):
    """Raised when document retrieval fails"""
    
    def __init__(self, message: str, query: Optional[str] = None):
        super().__init__(message, "RETRIEVAL_ERROR", {"query": query})


class GenerationError(RAGError):
    """Raised when response generation fails"""
    
    def __init__(self, message: str, model: Optional[str] = None):
        super().__init__(message, "GENERATION_ERROR", {"model": model})


class CacheError(RAGError):
    """Raised when cache operations fail"""
    
    def __init__(self, message: str, operation: Optional[str] = None, key: Optional[str] = None):
        super().__init__(message, "CACHE_ERROR", {"operation": operation, "key": key})


class AuthenticationError(RAGError):
    """Raised when authentication fails"""
    
    def __init__(self, message: str, provider: Optional[str] = None):
        super().__init__(message, "AUTH_ERROR", {"provider": provider})


class CircuitBreakerError(RAGError):
    """Raised when circuit breaker is open"""
    
    def __init__(self, message: str, service: Optional[str] = None):
        super().__init__(message, "CIRCUIT_BREAKER_OPEN", {"service": service})


# Error handling utilities

def format_error_response(error: RAGError) -> Dict[str, Any]:
    """Format error for API responses"""
    return {
        "error": {
            "message": error.message,
            "code": error.error_code,
            "details": error.details
        }
    }


def is_retryable_error(error: Exception) -> bool:
    """Check if an error is retryable"""
    retryable_errors = (
        RateLimitError,
        APIError,
        VectorStoreError,
        CacheError
    )
    
    # Check error type
    if isinstance(error, retryable_errors):
        return True
    
    # Check specific API errors
    if isinstance(error, APIError):
        retryable_codes = [429, 500, 502, 503, 504]
        return error.details.get("status_code") in retryable_codes
    
    return False


def get_retry_delay(error: Exception, attempt: int) -> float:
    """Get retry delay based on error and attempt number"""
    base_delay = 1.0
    
    if isinstance(error, RateLimitError):
        # Use retry-after if provided, otherwise exponential backoff
        retry_after = error.details.get("retry_after")
        if retry_after:
            return float(retry_after)
    
    # Exponential backoff with jitter
    import random
    delay = base_delay * (2 ** attempt)
    jitter = random.uniform(0, 0.1) * delay
    return delay + jitter