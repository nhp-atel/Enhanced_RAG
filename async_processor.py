"""
Async processing for improved performance and scalability.
"""
import asyncio
import aiohttp
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import logging
from concurrent.futures import ThreadPoolExecutor
import time

from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document

logger = logging.getLogger(__name__)


@dataclass
class ProcessingResult:
    """Result container for async processing."""
    success: bool
    data: Any = None
    error: Optional[str] = None
    processing_time: float = 0.0


class AsyncEmbeddingProcessor:
    """Async embedding generation with batching and rate limiting."""
    
    def __init__(
        self,
        embeddings: OpenAIEmbeddings,
        batch_size: int = 10,
        max_concurrent: int = 5,
        rate_limit_delay: float = 0.1
    ):
        self.embeddings = embeddings
        self.batch_size = batch_size
        self.max_concurrent = max_concurrent
        self.rate_limit_delay = rate_limit_delay
        self.semaphore = asyncio.Semaphore(max_concurrent)
    
    async def embed_documents_batch(
        self, 
        documents: List[Document]
    ) -> List[ProcessingResult]:
        """
        Process documents in batches with async execution.
        
        Args:
            documents: List of documents to embed
            
        Returns:
            List of processing results
        """
        start_time = time.time()
        
        # Split into batches
        batches = [
            documents[i:i + self.batch_size] 
            for i in range(0, len(documents), self.batch_size)
        ]
        
        logger.info(f"Processing {len(documents)} documents in {len(batches)} batches")
        
        # Process batches concurrently
        tasks = [
            self._process_batch(batch, batch_idx) 
            for batch_idx, batch in enumerate(batches)
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Flatten results
        flattened_results = []
        for result in results:
            if isinstance(result, Exception):
                flattened_results.append(
                    ProcessingResult(success=False, error=str(result))
                )
            else:
                flattened_results.extend(result)
        
        total_time = time.time() - start_time
        logger.info(f"Completed embedding in {total_time:.2f}s")
        
        return flattened_results
    
    async def _process_batch(
        self, 
        batch: List[Document], 
        batch_idx: int
    ) -> List[ProcessingResult]:
        """Process a single batch of documents."""
        async with self.semaphore:
            try:
                await asyncio.sleep(self.rate_limit_delay * batch_idx)
                
                # Extract text content
                texts = [doc.page_content for doc in batch]
                
                # Generate embeddings (this is still sync, but limited by semaphore)
                loop = asyncio.get_event_loop()
                with ThreadPoolExecutor() as executor:
                    embeddings = await loop.run_in_executor(
                        executor, 
                        self.embeddings.embed_documents, 
                        texts
                    )
                
                # Create results
                results = []
                for doc, embedding in zip(batch, embeddings):
                    results.append(ProcessingResult(
                        success=True,
                        data={'document': doc, 'embedding': embedding}
                    ))
                
                logger.debug(f"Batch {batch_idx} completed successfully")
                return results
                
            except Exception as e:
                logger.error(f"Batch {batch_idx} failed: {e}")
                return [ProcessingResult(success=False, error=str(e)) for _ in batch]


class AsyncPDFDownloader:
    """Async PDF downloading with concurrent processing."""
    
    def __init__(self, max_concurrent: int = 3):
        self.max_concurrent = max_concurrent
        self.semaphore = asyncio.Semaphore(max_concurrent)
    
    async def download_pdfs(
        self, 
        urls: List[str]
    ) -> Dict[str, ProcessingResult]:
        """
        Download multiple PDFs concurrently.
        
        Args:
            urls: List of PDF URLs to download
            
        Returns:
            Dictionary mapping URLs to download results
        """
        tasks = [self._download_single_pdf(url) for url in urls]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        return {
            url: result if not isinstance(result, Exception) 
            else ProcessingResult(success=False, error=str(result))
            for url, result in zip(urls, results)
        }
    
    async def _download_single_pdf(self, url: str) -> ProcessingResult:
        """Download a single PDF file."""
        async with self.semaphore:
            try:
                start_time = time.time()
                
                async with aiohttp.ClientSession() as session:
                    async with session.get(url) as response:
                        response.raise_for_status()
                        content = await response.read()
                
                # Save to temp file
                import tempfile
                import os
                
                with tempfile.NamedTemporaryFile(
                    delete=False, 
                    suffix='.pdf'
                ) as tmp_file:
                    tmp_file.write(content)
                    tmp_path = tmp_file.name
                
                processing_time = time.time() - start_time
                
                return ProcessingResult(
                    success=True,
                    data={'file_path': tmp_path, 'size': len(content)},
                    processing_time=processing_time
                )
                
            except Exception as e:
                logger.error(f"Failed to download {url}: {e}")
                return ProcessingResult(success=False, error=str(e))


# Caching decorator for expensive operations
from functools import wraps
import pickle
import hashlib
from pathlib import Path

def cached(cache_dir: Path, ttl_seconds: int = 3600):
    """Decorator for caching function results."""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Create cache key from function name and arguments
            cache_key = hashlib.md5(
                f"{func.__name__}_{str(args)}_{str(kwargs)}".encode()
            ).hexdigest()
            
            cache_file = cache_dir / f"{cache_key}.pkl"
            
            # Check if cached result exists and is fresh
            if cache_file.exists():
                cache_age = time.time() - cache_file.stat().st_mtime
                if cache_age < ttl_seconds:
                    try:
                        with open(cache_file, 'rb') as f:
                            cached_result = pickle.load(f)
                        logger.debug(f"Cache hit for {func.__name__}")
                        return cached_result
                    except Exception as e:
                        logger.warning(f"Cache read failed: {e}")
            
            # Execute function and cache result
            result = await func(*args, **kwargs)
            
            try:
                cache_dir.mkdir(parents=True, exist_ok=True)
                with open(cache_file, 'wb') as f:
                    pickle.dump(result, f)
                logger.debug(f"Cached result for {func.__name__}")
            except Exception as e:
                logger.warning(f"Cache write failed: {e}")
            
            return result
        return wrapper
    return decorator


# Usage example
async def main():
    """Example usage of async processing."""
    from langchain_openai import OpenAIEmbeddings
    
    # Setup
    embeddings = OpenAIEmbeddings()
    processor = AsyncEmbeddingProcessor(embeddings)
    downloader = AsyncPDFDownloader()
    
    # Download PDFs
    urls = [
        "https://arxiv.org/pdf/2101.00001",
        "https://arxiv.org/pdf/2102.00002"
    ]
    
    download_results = await downloader.download_pdfs(urls)
    
    # Process documents (assuming you have documents)
    documents = []  # Your documents here
    embedding_results = await processor.embed_documents_batch(documents)
    
    # Report results
    successful_downloads = sum(1 for r in download_results.values() if r.success)
    successful_embeddings = sum(1 for r in embedding_results if r.success)
    
    print(f"Downloads: {successful_downloads}/{len(urls)} successful")
    print(f"Embeddings: {successful_embeddings}/{len(embedding_results)} successful")


if __name__ == "__main__":
    asyncio.run(main())