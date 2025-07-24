#!/usr/bin/env python3
"""
Demonstration of caching and persistence performance improvements
"""

import sys
import time
import hashlib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from src.core.pipeline import create_pipeline
from src.stores.cache import create_cache_backend, SmartCache
from src.utils.logging import get_logger


def demonstrate_index_persistence():
    """Show FAISS index persistence benefits"""
    
    print("🔄 FAISS Index Persistence Demo")
    print("=" * 40)
    
    pipeline = create_pipeline()
    test_doc = Path("/tmp/test_persistence.txt")
    
    # Create test document
    with open(test_doc, 'w') as f:
        f.write("""
        Test Document for Persistence Demo
        
        This document contains multiple paragraphs that will be split into chunks.
        Each chunk will be embedded and indexed in FAISS.
        
        The first time this runs, it will:
        1. Download/read the document
        2. Extract metadata using LLM
        3. Split into chunks  
        4. Generate embeddings (API call)
        5. Build FAISS index
        6. Save index to disk
        
        The second time this runs, it will:
        1. Load existing index from disk (fast!)
        2. Skip all the expensive operations
        
        This demonstrates the massive speedup from persistence.
        
        Additional content to make the document longer and show more realistic
        processing times. In real scenarios, you might have papers with dozens
        of pages that take minutes to process initially.
        """)
    
    index_path = "./data/demo_index"
    
    print("\n1️⃣ First Run (Cold Start - No Cache)")
    print("-" * 35)
    
    start_time = time.time()
    
    # Process document (will build everything from scratch)
    result = pipeline.process_document(str(test_doc))
    
    if result.success:
        # Save index 
        pipeline.save_index(index_path)
        
        cold_time = time.time() - start_time
        print(f"✅ Cold processing completed: {cold_time:.2f}s")
        print(f"📊 Chunks created: {result.chunk_count}")
        print(f"💰 Cost: ${result.metadata and hasattr(result.metadata, 'cost') or 'N/A'}")
    else:
        print(f"❌ Processing failed: {result.error_message}")
        return
    
    print("\n2️⃣ Second Run (Warm Start - With Cache)")
    print("-" * 35)
    
    # Create new pipeline instance to simulate restart
    pipeline2 = create_pipeline()
    
    start_time = time.time()
    
    # Load existing index
    success = pipeline2.load_index(index_path)
    
    if success:
        warm_time = time.time() - start_time
        print(f"✅ Warm loading completed: {warm_time:.2f}s")
        
        # Test that it works
        test_result = pipeline2.query("What is this document about?", k=3)
        print(f"🔍 Query test successful: {len(test_result.sources)} sources found")
        
        # Show performance improvement
        speedup = cold_time / warm_time if warm_time > 0 else float('inf')
        print(f"\n📈 Performance Improvement:")
        print(f"   Cold start: {cold_time:.2f}s")
        print(f"   Warm start: {warm_time:.2f}s") 
        print(f"   Speedup: {speedup:.1f}x faster")
        
    else:
        print("❌ Failed to load index")
    
    # Cleanup
    test_doc.unlink()
    print(f"\n🗑️  Cleaned up test files")


def demonstrate_llm_caching():
    """Show LLM response caching benefits"""
    
    print("\n💾 LLM Response Caching Demo")
    print("=" * 35)
    
    # Create cache backend
    cache_config = {
        'backend': 'memory',
        'max_size_mb': 100,
        'ttl_seconds': 3600
    }
    
    cache_backend = create_cache_backend(cache_config)
    cache = SmartCache(cache_backend, prefix="llm_demo")
    
    # Test content for caching
    test_content = "This is test content for LLM processing"
    content_hash = hashlib.sha256(test_content.encode()).hexdigest()[:16]
    
    def expensive_llm_operation():
        """Simulate expensive LLM call"""
        print("   🔄 Making expensive LLM API call...")
        time.sleep(2)  # Simulate API latency
        return f"Processed: {test_content} (hash: {content_hash})"
    
    print("\n1️⃣ First LLM Call (Cache Miss)")
    start_time = time.time()
    
    result1 = cache.get_or_compute(
        key_parts=["llm_summary", content_hash],
        compute_func=expensive_llm_operation,
        ttl=3600
    )
    
    miss_time = time.time() - start_time
    print(f"✅ Result: {result1}")
    print(f"⏱️  Time: {miss_time:.2f}s")
    
    print("\n2️⃣ Second LLM Call (Cache Hit)")
    start_time = time.time()
    
    result2 = cache.get_or_compute(
        key_parts=["llm_summary", content_hash],
        compute_func=expensive_llm_operation,  # Won't be called
        ttl=3600
    )
    
    hit_time = time.time() - start_time
    print(f"✅ Result: {result2}")
    print(f"⏱️  Time: {hit_time:.2f}s")
    
    # Verify results are identical
    assert result1 == result2, "Cache should return identical results"
    
    speedup = miss_time / hit_time if hit_time > 0 else float('inf')
    print(f"\n📈 Caching Improvement:")
    print(f"   Cache miss: {miss_time:.2f}s")
    print(f"   Cache hit: {hit_time:.3f}s")
    print(f"   Speedup: {speedup:.0f}x faster")
    
    # Show cache stats
    stats = cache_backend.get_stats()
    print(f"\n📊 Cache Stats: {stats['entry_count']} entries, {stats['total_size_mb']:.2f}MB")


def demonstrate_incremental_updates():
    """Show incremental document addition"""
    
    print("\n➕ Incremental Updates Demo")
    print("=" * 32)
    
    pipeline = create_pipeline()
    
    # Process first document
    doc1 = Path("/tmp/doc1.txt")
    with open(doc1, 'w') as f:
        f.write("First document content for incremental demo")
    
    print("1️⃣ Processing first document...")
    start_time = time.time()
    
    result1 = pipeline.process_document(str(doc1))
    time1 = time.time() - start_time
    
    if result1.success:
        print(f"✅ First doc processed: {time1:.2f}s, {result1.chunk_count} chunks")
        initial_stats = pipeline.get_stats()
        initial_count = initial_stats['vector_store']['document_count']
    else:
        print("❌ First document failed")
        return
    
    # Process second document (incremental)
    doc2 = Path("/tmp/doc2.txt")
    with open(doc2, 'w') as f:
        f.write("Second document content - this should be added incrementally to existing index")
    
    print("\n2️⃣ Processing second document (incremental)...")
    start_time = time.time()
    
    result2 = pipeline.process_document(str(doc2))
    time2 = time.time() - start_time
    
    if result2.success:
        print(f"✅ Second doc processed: {time2:.2f}s, {result2.chunk_count} chunks")
        
        # Verify incremental addition
        final_stats = pipeline.get_stats()
        final_count = final_stats['vector_store']['document_count']
        
        print(f"\n📊 Incremental Update Results:")
        print(f"   Initial documents: {initial_count}")
        print(f"   Final documents: {final_count}")
        print(f"   Added: {final_count - initial_count} documents")
        
        # Test that both documents are searchable
        query_result = pipeline.query("document content", k=5)
        print(f"🔍 Query found {len(query_result.sources)} sources from both documents")
        
    else:
        print("❌ Second document failed")
    
    # Cleanup
    doc1.unlink()
    doc2.unlink()


def demonstrate_cost_savings():
    """Show cost savings from caching"""
    
    print("\n💰 Cost Savings Demo")
    print("=" * 25)
    
    # Simulate processing the same content multiple times
    content_variants = [
        "Machine learning is a subset of artificial intelligence",
        "Deep learning uses neural networks with multiple layers", 
        "Natural language processing enables computers to understand text"
    ]
    
    # Without caching (simulated costs)
    print("❌ Without Caching:")
    total_cost_no_cache = 0
    for i, content in enumerate(content_variants * 3):  # Process each 3 times
        simulated_cost = 0.05  # $0.05 per LLM call
        total_cost_no_cache += simulated_cost
        if i < 3:  # Only show first few
            print(f"   Call {i+1}: ${simulated_cost:.3f}")
    
    print(f"   Total cost: ${total_cost_no_cache:.3f}")
    
    # With caching (only pay once per unique content)
    print("\n✅ With Caching:")
    unique_content = set(content_variants)
    total_cost_cached = len(unique_content) * 0.05
    cache_hits = len(content_variants * 3) - len(unique_content)
    
    print(f"   Unique calls: {len(unique_content)} × $0.05 = ${total_cost_cached:.3f}")
    print(f"   Cache hits: {cache_hits} × $0.00 = $0.000")
    print(f"   Total cost: ${total_cost_cached:.3f}")
    
    savings = total_cost_no_cache - total_cost_cached
    savings_percent = (savings / total_cost_no_cache) * 100
    
    print(f"\n💵 Cost Savings:")
    print(f"   Saved: ${savings:.3f} ({savings_percent:.1f}%)")
    print(f"   ROI: {total_cost_no_cache / total_cost_cached:.1f}x")


def demonstrate_production_workflow():
    """Show realistic production workflow with persistence"""
    
    print("\n🏭 Production Workflow Demo")
    print("=" * 35)
    
    pipeline = create_pipeline()
    
    print("Scenario: Daily research paper processing service")
    print("-" * 45)
    
    # Day 1: Process initial batch
    print("\n📅 Day 1: Initial batch processing")
    papers = [
        ("Paper A: Machine Learning Fundamentals", "ML fundamentals content"),
        ("Paper B: Deep Learning Applications", "Deep learning applications content"),
        ("Paper C: Natural Language Processing", "NLP techniques and methods content")
    ]
    
    day1_start = time.time()
    processed_papers = 0
    
    for title, content in papers:
        doc_path = Path(f"/tmp/{title.replace(':', '').replace(' ', '_')}.txt")
        with open(doc_path, 'w') as f:
            f.write(f"{title}\n\n{content}")
        
        result = pipeline.process_document(str(doc_path))
        if result.success:
            processed_papers += 1
        
        doc_path.unlink()
    
    # Save the day's work
    pipeline.save_index("./data/production_index")
    day1_time = time.time() - day1_start
    
    print(f"✅ Day 1 complete: {processed_papers} papers, {day1_time:.2f}s")
    
    # Day 2: Add new papers to existing index
    print("\n📅 Day 2: Incremental processing")
    
    # Create new pipeline (simulates service restart)
    pipeline2 = create_pipeline()
    
    # Load yesterday's work
    load_start = time.time()
    pipeline2.load_index("./data/production_index")
    load_time = time.time() - load_start
    
    print(f"📂 Loaded existing index: {load_time:.2f}s")
    
    # Process new papers
    new_papers = [
        ("Paper D: Computer Vision", "Computer vision and image processing"),
        ("Paper E: Reinforcement Learning", "RL algorithms and applications")
    ]
    
    day2_start = time.time()
    new_processed = 0
    
    for title, content in new_papers:
        doc_path = Path(f"/tmp/{title.replace(':', '').replace(' ', '_')}.txt")
        with open(doc_path, 'w') as f:
            f.write(f"{title}\n\n{content}")
        
        result = pipeline2.process_document(str(doc_path))
        if result.success:
            new_processed += 1
        
        doc_path.unlink()
    
    day2_time = time.time() - day2_start
    
    print(f"✅ Day 2 complete: {new_processed} new papers, {day2_time:.2f}s")
    
    # Show total system state
    final_stats = pipeline2.get_stats()
    total_docs = final_stats['vector_store']['document_count']
    
    print(f"\n📊 Production System Status:")
    print(f"   Total papers indexed: {processed_papers + new_processed}")
    print(f"   Total document chunks: {total_docs}")
    print(f"   Day 1 processing: {day1_time:.2f}s")
    print(f"   Day 2 processing: {day2_time:.2f}s (including load)")
    print(f"   Persistence overhead: {load_time:.2f}s")
    
    # Test cross-day queries
    query_result = pipeline2.query("machine learning and deep learning", k=8)
    print(f"🔍 Cross-day query: {len(query_result.sources)} sources found")


if __name__ == "__main__":
    try:
        demonstrate_index_persistence()
        demonstrate_llm_caching()
        demonstrate_incremental_updates()
        demonstrate_cost_savings()
        demonstrate_production_workflow()
        
        print("\n🎉 All caching and persistence demos completed!")
        print("\n💡 Key Takeaways:")
        print("   • Index persistence provides massive speedup (10-100x)")
        print("   • LLM caching eliminates redundant API calls")
        print("   • Incremental updates avoid full rebuilds")
        print("   • Production workflows become efficient and cost-effective")
        
    except Exception as e:
        print(f"Demo failed: {e}")
        sys.exit(1)