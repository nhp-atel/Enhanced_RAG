#!/usr/bin/env python3
"""
Demonstration of error handling and observability features
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from src.core.pipeline import create_pipeline
from src.utils.logging import get_logger, RequestTracker, with_request_context
from src.utils.errors import RAGError


def demonstrate_error_handling():
    """Show how errors are handled with full observability"""
    
    # Create pipeline and logger
    pipeline = create_pipeline()
    logger = get_logger("error_demo")
    
    print("🔍 Error Handling & Observability Demo")
    print("=" * 50)
    
    # Scenario 1: Network timeout with retry
    print("\n1. Network Timeout Scenario:")
    with RequestTracker(logger) as tracker:
        try:
            # This will fail but retry automatically
            result = pipeline.process_document("https://invalid-url-that-will-timeout.com/paper.pdf")
            
        except Exception as e:
            # Even failures are fully logged with context
            logger.error("Processing failed after retries",
                        error_type=type(e).__name__,
                        original_error=str(e),
                        retries_attempted=3)
            print(f"❌ Failed as expected: {e}")
    
    # Scenario 2: Rate limit with exponential backoff
    print("\n2. Rate Limit Scenario:")
    with with_request_context(user_id="demo_user", session_id="demo_session"):
        try:
            # Simulate many rapid requests (would trigger rate limiting)
            for i in range(3):
                tracker.log_milestone(f"Request {i+1}")
                # In real scenario, this would hit rate limits and retry
                
        except RAGError as e:
            logger.error("Rate limit exceeded",
                        error_code=e.error_code,
                        retry_after=e.details.get('retry_after'),
                        user_action="slow_down_requests")
    
    # Scenario 3: Malformed document with graceful degradation
    print("\n3. Malformed Document Scenario:")
    with RequestTracker(logger) as tracker:
        try:
            # Create a malformed PDF file
            malformed_file = Path("/tmp/malformed.pdf")
            with open(malformed_file, 'w') as f:
                f.write("This is not a valid PDF content")
            
            result = pipeline.process_document(str(malformed_file))
            
            if not result.success:
                # Structured error with full context
                logger.warning("Document processing degraded",
                             source=str(malformed_file),
                             error_message=result.error_message,
                             fallback_action="manual_review_required",
                             processing_time_ms=result.processing_time_ms)
                print(f"⚠️  Graceful degradation: {result.error_message}")
            
        finally:
            # Cleanup
            if malformed_file.exists():
                malformed_file.unlink()


def demonstrate_performance_tracking():
    """Show performance monitoring capabilities"""
    
    print("\n📊 Performance Monitoring Demo")
    print("=" * 40)
    
    from src.utils.logging import PerformanceLogger
    
    logger = get_logger("perf_demo")
    perf_logger = PerformanceLogger(logger)
    
    # Track different operation types
    operations = [
        ("document_download", 1200, {"size_mb": 5.2}),
        ("pdf_parsing", 800, {"pages": 42}),
        ("embedding_generation", 2300, {"chunks": 156, "tokens": 12400}),
        ("vector_indexing", 450, {"dimensions": 1536, "documents": 156}),
        ("similarity_search", 120, {"query_length": 45, "k": 6})
    ]
    
    for operation, duration_ms, extra_metrics in operations:
        perf_logger.log_operation(operation, duration_ms, **extra_metrics)
        print(f"📈 Logged {operation}: {duration_ms}ms")
    
    # Track API costs
    perf_logger.log_api_call(
        provider="openai",
        model="gpt-4o-mini", 
        tokens=8540,
        cost=0.034,
        duration_ms=1800,
        request_type="completion",
        success=True
    )
    print("💰 Logged API cost: $0.034")


def demonstrate_structured_logs():
    """Show structured logging output"""
    
    print("\n📝 Structured Logging Demo")
    print("=" * 35)
    
    logger = get_logger("structured_demo")
    
    # Different log levels with rich context
    logger.debug("Processing pipeline step", 
                step="document_ingestion",
                progress=0.25,
                documents_processed=1,
                total_documents=4)
    
    logger.info("User query processed",
               user_id="user_12345",
               query="What is machine learning?",
               query_length=26,
               response_time_ms=1240,
               tokens_used=890,
               cost_cents=3.2,
               satisfaction_score=None)
    
    logger.warning("Cache miss for expensive operation",
                  operation="concept_extraction", 
                  cache_key="hash_abc123",
                  fallback_duration_ms=3400,
                  cache_hit_rate=0.73)
    
    logger.error("Configuration validation failed",
               config_file="./config/production.yaml",
               validation_errors=["chunk_size must be positive", "missing API key"],
               suggested_action="check_config_documentation")
    
    print("✅ All logs include request_id, timestamp, and structured data")


def demonstrate_health_monitoring():
    """Show health check and monitoring"""
    
    print("\n🏥 Health Monitoring Demo")  
    print("=" * 32)
    
    pipeline = create_pipeline()
    
    # Comprehensive health check
    health = pipeline.health_check()
    
    print("System Health Status:")
    print(f"Overall: {health['overall']}")
    
    for component, status in health['components'].items():
        emoji = "✅" if status['status'] == 'healthy' else "❌" 
        print(f"{emoji} {component}: {status['status']}")
        
        if status['status'] != 'healthy':
            print(f"   Error: {status.get('error', 'N/A')}")


def demonstrate_correlation_tracking():
    """Show request correlation across components"""
    
    print("\n🔗 Request Correlation Demo")
    print("=" * 33)
    
    # Simulate a request that flows through multiple components
    logger = get_logger("correlation_demo")
    
    request_id = "req_demo_12345"
    user_id = "user_alice"
    
    with with_request_context(request_id=request_id, user_id=user_id):
        
        # Component 1: API Gateway
        logger.info("Request received", 
                   component="api_gateway",
                   endpoint="/query",
                   method="POST")
        
        # Component 2: Document Retrieval  
        logger.info("Retrieving documents",
                   component="retrieval",
                   query_embedding_dims=1536,
                   search_k=6)
        
        # Component 3: LLM Generation
        logger.info("Generating response",
                   component="llm_client", 
                   model="gpt-4o-mini",
                   prompt_tokens=1240)
        
        # Component 4: Response Assembly
        logger.info("Assembling response",
                   component="response_builder",
                   sources_count=6,
                   total_chars=2840)
        
        print(f"🔍 All logs linked by request_id: {request_id}")
        print("📊 Full trace available for debugging")


if __name__ == "__main__":
    try:
        demonstrate_error_handling()
        demonstrate_performance_tracking() 
        demonstrate_structured_logs()
        demonstrate_health_monitoring()
        demonstrate_correlation_tracking()
        
        print("\n🎉 Demo completed! Check logs for structured output.")
        
    except Exception as e:
        print(f"Demo failed: {e}")
        sys.exit(1)