"""
Retry utilities with exponential backoff and circuit breaker patterns.
"""

import time
import random
import functools
from typing import Callable, List, Type, Any, Optional
from dataclasses import dataclass
from enum import Enum

from .errors import CircuitBreakerError, is_retryable_error, get_retry_delay


class CircuitBreakerState(Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


@dataclass
class RetryConfig:
    """Configuration for retry behavior"""
    max_retries: int = 3
    backoff_factor: float = 2.0
    max_backoff: float = 60.0
    jitter: bool = True
    retryable_exceptions: Optional[List[Type[Exception]]] = None
    retry_on_status: Optional[List[int]] = None


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker"""
    failure_threshold: int = 5
    recovery_timeout: float = 30.0
    expected_exception: Type[Exception] = Exception


class CircuitBreaker:
    """Circuit breaker implementation for fault tolerance"""
    
    def __init__(self, config: CircuitBreakerConfig):
        self.config = config
        self.state = CircuitBreakerState.CLOSED
        self.failure_count = 0
        self.last_failure_time = 0
        self.success_count = 0
    
    def call(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with circuit breaker protection"""
        if self.state == CircuitBreakerState.OPEN:
            if time.time() - self.last_failure_time < self.config.recovery_timeout:
                raise CircuitBreakerError(
                    f"Circuit breaker is open for {func.__name__}",
                    service=func.__name__
                )
            else:
                self.state = CircuitBreakerState.HALF_OPEN
                self.success_count = 0
        
        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
        except self.config.expected_exception as e:
            self._on_failure()
            raise
    
    def _on_success(self):
        """Handle successful call"""
        if self.state == CircuitBreakerState.HALF_OPEN:
            self.success_count += 1
            if self.success_count >= 3:  # Require multiple successes to close
                self.state = CircuitBreakerState.CLOSED
                self.failure_count = 0
        elif self.state == CircuitBreakerState.CLOSED:
            self.failure_count = 0
    
    def _on_failure(self):
        """Handle failed call"""
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.failure_count >= self.config.failure_threshold:
            self.state = CircuitBreakerState.OPEN


def retry_with_backoff(
    max_retries: int = 3,
    backoff_factor: float = 2.0,
    max_backoff: float = 60.0,
    jitter: bool = True,
    retryable_exceptions: Optional[List[Type[Exception]]] = None,
    retry_on_status: Optional[List[int]] = None
):
    """
    Decorator for retrying functions with exponential backoff
    
    Args:
        max_retries: Maximum number of retry attempts
        backoff_factor: Factor by which delay increases each retry
        max_backoff: Maximum delay between retries
        jitter: Add random jitter to prevent thundering herd
        retryable_exceptions: List of exceptions that should trigger retry
        retry_on_status: List of HTTP status codes that should trigger retry
    """
    
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            config = RetryConfig(
                max_retries=max_retries,
                backoff_factor=backoff_factor,
                max_backoff=max_backoff,
                jitter=jitter,
                retryable_exceptions=retryable_exceptions,
                retry_on_status=retry_on_status
            )
            
            return _execute_with_retry(func, config, *args, **kwargs)
        
        return wrapper
    return decorator


def _execute_with_retry(func: Callable, config: RetryConfig, *args, **kwargs) -> Any:
    """Execute function with retry logic"""
    last_exception = None
    
    for attempt in range(config.max_retries + 1):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            last_exception = e
            
            # Check if we should retry
            if attempt == config.max_retries:
                break
            
            if not _should_retry(e, config):
                break
            
            # Calculate delay
            delay = _calculate_delay(e, attempt, config)
            
            # Log retry attempt (would be better with actual logger)
            print(f"Retry attempt {attempt + 1}/{config.max_retries} for {func.__name__} after {delay:.2f}s: {e}")
            
            time.sleep(delay)
    
    # All retries exhausted
    raise last_exception


def _should_retry(exception: Exception, config: RetryConfig) -> bool:
    """Determine if exception should trigger a retry"""
    # Check against configured retryable exceptions
    if config.retryable_exceptions:
        if not isinstance(exception, tuple(config.retryable_exceptions)):
            return False
    
    # Check against general retryable error logic
    if not is_retryable_error(exception):
        return False
    
    # Check HTTP status codes if applicable
    if config.retry_on_status:
        if hasattr(exception, 'status_code'):
            return exception.status_code in config.retry_on_status
        elif hasattr(exception, 'details') and 'status_code' in exception.details:
            return exception.details['status_code'] in config.retry_on_status
    
    return True


def _calculate_delay(exception: Exception, attempt: int, config: RetryConfig) -> float:
    """Calculate delay before next retry"""
    # Use error-specific delay if available
    error_delay = get_retry_delay(exception, attempt)
    if error_delay > 0 and not isinstance(exception, type(exception).__bases__[0]):
        return min(error_delay, config.max_backoff)
    
    # Exponential backoff
    delay = min(config.backoff_factor ** attempt, config.max_backoff)
    
    # Add jitter to prevent thundering herd
    if config.jitter:
        jitter_amount = delay * 0.1
        delay += random.uniform(-jitter_amount, jitter_amount)
    
    return max(0, delay)


class RetryableOperation:
    """Class-based retry handler for more complex scenarios"""
    
    def __init__(self, config: RetryConfig, circuit_breaker: Optional[CircuitBreaker] = None):
        self.config = config
        self.circuit_breaker = circuit_breaker
        self.attempt_count = 0
        self.total_delay = 0
    
    def execute(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with retry and circuit breaker protection"""
        self.attempt_count = 0
        self.total_delay = 0
        
        if self.circuit_breaker:
            return self.circuit_breaker.call(self._execute_with_metrics, func, *args, **kwargs)
        else:
            return self._execute_with_metrics(func, *args, **kwargs)
    
    def _execute_with_metrics(self, func: Callable, *args, **kwargs) -> Any:
        """Execute with retry metrics tracking"""
        start_time = time.time()
        
        try:
            result = _execute_with_retry(func, self.config, *args, **kwargs)
            self.total_delay = time.time() - start_time
            return result
        except Exception as e:
            self.total_delay = time.time() - start_time
            raise
    
    def get_metrics(self) -> dict:
        """Get retry metrics"""
        return {
            "attempt_count": self.attempt_count,
            "total_delay": self.total_delay,
            "circuit_breaker_state": self.circuit_breaker.state.value if self.circuit_breaker else None
        }


# Predefined retry configurations

DEFAULT_RETRY_CONFIG = RetryConfig(
    max_retries=3,
    backoff_factor=2.0,
    max_backoff=60.0,
    jitter=True
)

API_RETRY_CONFIG = RetryConfig(
    max_retries=5,
    backoff_factor=1.5,
    max_backoff=30.0,
    jitter=True,
    retry_on_status=[429, 500, 502, 503, 504]
)

DB_RETRY_CONFIG = RetryConfig(
    max_retries=3,
    backoff_factor=2.0,
    max_backoff=10.0,
    jitter=True
)

# Predefined circuit breaker configurations

DEFAULT_CIRCUIT_BREAKER = CircuitBreakerConfig(
    failure_threshold=5,
    recovery_timeout=30.0,
    expected_exception=Exception
)

API_CIRCUIT_BREAKER = CircuitBreakerConfig(
    failure_threshold=3,
    recovery_timeout=60.0,
    expected_exception=Exception
)


# Convenience functions

def with_retry(func: Callable, config: Optional[RetryConfig] = None) -> Any:
    """Execute function with retry using provided or default config"""
    config = config or DEFAULT_RETRY_CONFIG
    operation = RetryableOperation(config)
    return operation.execute(func)


def with_circuit_breaker(func: Callable, cb_config: Optional[CircuitBreakerConfig] = None) -> Any:
    """Execute function with circuit breaker protection"""
    cb_config = cb_config or DEFAULT_CIRCUIT_BREAKER
    circuit_breaker = CircuitBreaker(cb_config)
    return circuit_breaker.call(func)