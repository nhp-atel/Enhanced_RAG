"""
Structured logging system with request tracking and metrics integration.
"""

import os
import json
import logging
import logging.handlers
import uuid
import time
from datetime import datetime
from typing import Dict, Any, Optional, Union
from pathlib import Path
from contextvars import ContextVar
import threading

from ..interfaces import LoggerInterface
from ..utils.errors import ConfigurationError


# Context variables for request tracking
request_id_context: ContextVar[str] = ContextVar('request_id', default='')
user_id_context: ContextVar[str] = ContextVar('user_id', default='')
session_id_context: ContextVar[str] = ContextVar('session_id', default='')


class StructuredFormatter(logging.Formatter):
    """Custom formatter for structured JSON logging"""
    
    def format(self, record: logging.LogRecord) -> str:
        # Base log entry
        log_entry = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno
        }
        
        # Add context information
        request_id = request_id_context.get('')
        if request_id:
            log_entry["request_id"] = request_id
        
        user_id = user_id_context.get('')
        if user_id:
            log_entry["user_id"] = user_id
        
        session_id = session_id_context.get('')
        if session_id:
            log_entry["session_id"] = session_id
        
        # Add thread information
        log_entry["thread_id"] = threading.current_thread().ident
        log_entry["thread_name"] = threading.current_thread().name
        
        # Add process information
        log_entry["process_id"] = os.getpid()
        
        # Add extra fields from record
        extra_fields = {}
        for key, value in record.__dict__.items():
            if key not in ['name', 'msg', 'args', 'levelname', 'levelno', 'pathname', 
                          'filename', 'module', 'exc_info', 'exc_text', 'stack_info',
                          'lineno', 'funcName', 'created', 'msecs', 'relativeCreated',
                          'thread', 'threadName', 'processName', 'process', 'message']:
                extra_fields[key] = value
        
        if extra_fields:
            log_entry["extra"] = extra_fields
        
        # Add exception information
        if record.exc_info:
            log_entry["exception"] = {
                "type": record.exc_info[0].__name__ if record.exc_info[0] else None,
                "message": str(record.exc_info[1]) if record.exc_info[1] else None,
                "traceback": self.formatException(record.exc_info) if record.exc_info else None
            }
        
        return json.dumps(log_entry, default=str, ensure_ascii=False)


class SimpleFormatter(logging.Formatter):
    """Simple human-readable formatter"""
    
    def __init__(self):
        super().__init__(
            fmt='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )


class RAGLogger(LoggerInterface):
    """Enhanced logger implementation with structured logging and context tracking"""
    
    def __init__(self, name: str, config: Dict[str, Any]):
        self.name = name
        self.config = config
        self.logger = logging.getLogger(name)
        self.context: Dict[str, Any] = {}
        
        # Configure logger
        self._configure_logger()
    
    def _configure_logger(self) -> None:
        """Configure the underlying logger"""
        # Set level
        level = getattr(logging, self.config.get('level', 'INFO').upper())
        self.logger.setLevel(level)
        
        # Clear existing handlers
        self.logger.handlers.clear()
        
        # Create formatter
        log_format = self.config.get('format', 'structured')
        if log_format == 'structured':
            formatter = StructuredFormatter()
        else:
            formatter = SimpleFormatter()
        
        # Configure output
        output = self.config.get('output', 'console')
        
        if output in ['console', 'both']:
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)
        
        if output in ['file', 'both']:
            self._setup_file_handler(formatter)
        
        # Prevent propagation to root logger
        self.logger.propagate = False
    
    def _setup_file_handler(self, formatter: logging.Formatter) -> None:
        """Setup rotating file handler"""
        file_path = self.config.get('file_path', './logs/rag_system.log')
        max_size_mb = self.config.get('max_file_size_mb', 100)
        backup_count = self.config.get('backup_count', 5)
        
        # Create log directory
        log_path = Path(file_path)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Create rotating file handler
        max_bytes = max_size_mb * 1024 * 1024
        file_handler = logging.handlers.RotatingFileHandler(
            filename=file_path,
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding='utf-8'
        )
        
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)
    
    def info(self, message: str, **kwargs) -> None:
        """Log info message"""
        self._log(logging.INFO, message, **kwargs)
    
    def warning(self, message: str, **kwargs) -> None:
        """Log warning message"""
        self._log(logging.WARNING, message, **kwargs)
    
    def error(self, message: str, **kwargs) -> None:
        """Log error message"""
        self._log(logging.ERROR, message, **kwargs)
    
    def debug(self, message: str, **kwargs) -> None:
        """Log debug message"""
        self._log(logging.DEBUG, message, **kwargs)
    
    def _log(self, level: int, message: str, **kwargs) -> None:
        """Internal logging method with context"""
        # Merge context and kwargs
        extra = {**self.context, **kwargs}
        
        # Filter out None values
        extra = {k: v for k, v in extra.items() if v is not None}
        
        self.logger.log(level, message, extra=extra)
    
    def set_context(self, **kwargs) -> None:
        """Set logging context"""
        self.context.update(kwargs)
    
    def clear_context(self) -> None:
        """Clear logging context"""
        self.context.clear()
    
    def with_context(self, **kwargs) -> 'RAGLogger':
        """Create logger with additional context"""
        new_logger = RAGLogger(self.name, self.config)
        new_logger.context = {**self.context, **kwargs}
        return new_logger


class RequestTracker:
    """Tracks requests with unique IDs and timing information"""
    
    def __init__(self, logger: RAGLogger):
        self.logger = logger
        self.start_time: Optional[float] = None
        self.request_id: str = str(uuid.uuid4())
    
    def __enter__(self) -> 'RequestTracker':
        """Start request tracking"""
        self.start_time = time.time()
        request_id_context.set(self.request_id)
        
        self.logger.info("Request started", request_id=self.request_id)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """End request tracking"""
        duration_ms = int((time.time() - self.start_time) * 1000) if self.start_time else 0
        
        if exc_type is None:
            self.logger.info("Request completed", 
                           request_id=self.request_id,
                           duration_ms=duration_ms,
                           status="success")
        else:
            self.logger.error("Request failed",
                            request_id=self.request_id,
                            duration_ms=duration_ms,
                            status="error",
                            error_type=exc_type.__name__ if exc_type else None,
                            error_message=str(exc_val) if exc_val else None)
        
        # Clear context
        request_id_context.set('')
    
    def log_milestone(self, milestone: str, **kwargs) -> None:
        """Log a milestone within the request"""
        current_time = time.time()
        elapsed_ms = int((current_time - self.start_time) * 1000) if self.start_time else 0
        
        self.logger.info(f"Milestone: {milestone}",
                        request_id=self.request_id,
                        elapsed_ms=elapsed_ms,
                        **kwargs)


class PerformanceLogger:
    """Specialized logger for performance metrics"""
    
    def __init__(self, logger: RAGLogger):
        self.logger = logger
    
    def log_operation(self, operation: str, duration_ms: float, **kwargs) -> None:
        """Log operation performance"""
        self.logger.info("Operation completed",
                        operation=operation,
                        duration_ms=duration_ms,
                        performance=True,
                        **kwargs)
    
    def log_api_call(self, provider: str, model: str, tokens: int, cost: float, 
                    duration_ms: float, **kwargs) -> None:
        """Log API call metrics"""
        self.logger.info("API call completed",
                        provider=provider,
                        model=model,
                        tokens_used=tokens,
                        cost_usd=cost,
                        duration_ms=duration_ms,
                        api_call=True,
                        **kwargs)
    
    def log_vector_operation(self, operation: str, document_count: int, 
                           embedding_dimensions: int, duration_ms: float, **kwargs) -> None:
        """Log vector store operations"""
        self.logger.info("Vector operation completed",
                        operation=operation,
                        document_count=document_count,
                        embedding_dimensions=embedding_dimensions,
                        duration_ms=duration_ms,
                        vector_operation=True,
                        **kwargs)


class SecurityLogger:
    """Specialized logger for security events"""
    
    def __init__(self, logger: RAGLogger):
        self.logger = logger
    
    def log_authentication(self, user_id: str, success: bool, method: str, **kwargs) -> None:
        """Log authentication attempts"""
        self.logger.info("Authentication attempt",
                        user_id=user_id,
                        success=success,
                        auth_method=method,
                        security_event=True,
                        **kwargs)
    
    def log_authorization(self, user_id: str, resource: str, action: str, 
                         allowed: bool, **kwargs) -> None:
        """Log authorization checks"""
        self.logger.info("Authorization check",
                        user_id=user_id,
                        resource=resource,
                        action=action,
                        allowed=allowed,
                        security_event=True,
                        **kwargs)
    
    def log_suspicious_activity(self, event_type: str, user_id: str = None, 
                              ip_address: str = None, **kwargs) -> None:
        """Log suspicious activities"""
        self.logger.warning("Suspicious activity detected",
                          event_type=event_type,
                          user_id=user_id,
                          ip_address=ip_address,
                          security_event=True,
                          **kwargs)


class LoggerFactory:
    """Factory for creating configured loggers"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self._loggers: Dict[str, RAGLogger] = {}
    
    def get_logger(self, name: str) -> RAGLogger:
        """Get or create logger for given name"""
        if name not in self._loggers:
            self._loggers[name] = RAGLogger(name, self.config)
        
        return self._loggers[name]
    
    def get_performance_logger(self, name: str = "performance") -> PerformanceLogger:
        """Get performance logger"""
        base_logger = self.get_logger(name)
        return PerformanceLogger(base_logger)
    
    def get_security_logger(self, name: str = "security") -> SecurityLogger:
        """Get security logger"""
        base_logger = self.get_logger(name)
        return SecurityLogger(base_logger)
    
    def create_request_tracker(self, logger_name: str = "requests") -> RequestTracker:
        """Create request tracker"""
        base_logger = self.get_logger(logger_name)
        return RequestTracker(base_logger)


# Context managers for logging

def with_request_context(request_id: str = None, user_id: str = None, 
                        session_id: str = None):
    """Context manager for setting request context"""
    class RequestContext:
        def __init__(self):
            self.tokens = []
        
        def __enter__(self):
            if request_id:
                self.tokens.append(request_id_context.set(request_id))
            if user_id:
                self.tokens.append(user_id_context.set(user_id))
            if session_id:
                self.tokens.append(session_id_context.set(session_id))
            return self
        
        def __exit__(self, exc_type, exc_val, exc_tb):
            for token in reversed(self.tokens):
                token.var.set(token.old_value)
    
    return RequestContext()


def timed_operation(logger: RAGLogger, operation: str, **kwargs):
    """Context manager for timing operations"""
    class TimedOperation:
        def __init__(self):
            self.start_time = None
        
        def __enter__(self):
            self.start_time = time.time()
            logger.debug(f"Starting operation: {operation}", **kwargs)
            return self
        
        def __exit__(self, exc_type, exc_val, exc_tb):
            duration_ms = int((time.time() - self.start_time) * 1000)
            
            if exc_type is None:
                logger.info(f"Operation completed: {operation}",
                          duration_ms=duration_ms,
                          **kwargs)
            else:
                logger.error(f"Operation failed: {operation}",
                           duration_ms=duration_ms,
                           error_type=exc_type.__name__ if exc_type else None,
                           error_message=str(exc_val) if exc_val else None,
                           **kwargs)
    
    return TimedOperation()


# Global logger factory
_global_logger_factory: Optional[LoggerFactory] = None


def get_logger_factory(config: Optional[Dict[str, Any]] = None) -> LoggerFactory:
    """Get global logger factory"""
    global _global_logger_factory
    
    if _global_logger_factory is None:
        if config is None:
            # Default configuration
            config = {
                'level': 'INFO',
                'format': 'structured',
                'output': 'console',
                'include_request_ids': True
            }
        
        _global_logger_factory = LoggerFactory(config)
    
    return _global_logger_factory


def get_logger(name: str) -> RAGLogger:
    """Get logger instance"""
    return get_logger_factory().get_logger(name)


def setup_logging(config: Dict[str, Any]) -> None:
    """Setup global logging configuration"""
    global _global_logger_factory
    _global_logger_factory = LoggerFactory(config)