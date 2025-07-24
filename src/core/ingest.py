"""
Document ingestion module - handles loading and preprocessing of various document types.
"""

import os
import re
import requests
import hashlib
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass
from urllib.parse import urlparse

from pypdf import PdfReader
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document

from ..interfaces import LLMClient, LoggerInterface, MetricsInterface
from ..utils.errors import DocumentError, ValidationError
from ..utils.retry import retry_with_backoff


@dataclass
class DocumentMetadata:
    """Structured document metadata"""
    title: str
    authors: List[str]
    institutions: List[str]
    publication_date: str
    arxiv_id: str
    keywords: List[str]
    abstract: str
    source_url: Optional[str] = None
    file_path: Optional[str] = None
    content_hash: Optional[str] = None
    page_count: Optional[int] = None
    file_size_bytes: Optional[int] = None


class DocumentIngestor:
    """Handles ingestion and preprocessing of research documents"""
    
    def __init__(
        self,
        llm_client: LLMClient,
        logger: LoggerInterface,
        metrics: MetricsInterface,
        config: Dict[str, Any]
    ):
        self.llm_client = llm_client
        self.logger = logger
        self.metrics = metrics
        self.config = config
        
        # Configuration
        self.max_pages_for_metadata = config.get('max_pages_for_metadata', 3)
        self.metadata_char_limit = config.get('metadata_char_limit', 8000)
        self.supported_formats = config.get('supported_formats', ['.pdf'])
        self.download_timeout = config.get('download_timeout', 60)
        self.max_file_size_mb = config.get('max_file_size_mb', 100)
    
    def ingest_document(
        self, 
        source: Union[str, Path], 
        document_id: Optional[str] = None
    ) -> tuple[List[Document], DocumentMetadata]:
        """
        Main ingestion method - handles URLs, local files, and validates input
        
        Args:
            source: URL or file path to document
            document_id: Optional custom document ID
            
        Returns:
            Tuple of (document_chunks, metadata)
        """
        start_time = self._get_timestamp()
        
        try:
            self.logger.info("Starting document ingestion", source=str(source))
            self.metrics.increment_counter("documents.ingest.started")
            
            # Validate input
            self._validate_source(source)
            
            # Download or load file
            file_path = self._get_or_download_file(source)
            
            # Validate file
            self._validate_file(file_path)
            
            # Load document pages
            pages = self._load_pdf_pages(file_path)
            
            # Extract metadata
            metadata = self._extract_metadata(pages, source, file_path)
            
            # Create document chunks
            document_chunks = self._create_document_chunks(pages, metadata)
            
            # Add processing metadata
            self._add_processing_metadata(document_chunks, metadata, document_id)
            
            # Log success
            processing_time = self._get_timestamp() - start_time
            self.logger.info(
                "Document ingestion completed",
                source=str(source),
                pages=len(pages),
                chunks=len(document_chunks),
                processing_time_ms=processing_time
            )
            
            self.metrics.record_histogram("documents.ingest.duration_ms", processing_time)
            self.metrics.increment_counter("documents.ingest.success")
            
            return document_chunks, metadata
            
        except Exception as e:
            processing_time = self._get_timestamp() - start_time
            self.logger.error(
                "Document ingestion failed",
                source=str(source),
                error=str(e),
                processing_time_ms=processing_time
            )
            
            self.metrics.increment_counter("documents.ingest.failed")
            self.metrics.record_histogram("documents.ingest.duration_ms", processing_time)
            
            raise DocumentError(f"Failed to ingest document: {e}") from e
    
    def _validate_source(self, source: Union[str, Path]) -> None:
        """Validate input source"""
        if not source:
            raise ValidationError("Source cannot be empty")
        
        source_str = str(source)
        
        # Check if URL
        if source_str.startswith('http'):
            parsed = urlparse(source_str)
            if not parsed.netloc:
                raise ValidationError("Invalid URL format")
        
        # Check if local file
        elif not source_str.startswith('http'):
            if not Path(source_str).exists():
                raise ValidationError(f"File does not exist: {source_str}")
    
    def _get_or_download_file(self, source: Union[str, Path]) -> Path:
        """Download file from URL or return local path"""
        source_str = str(source)
        
        if source_str.startswith('http'):
            return self._download_file(source_str)
        else:
            return Path(source_str)
    
    @retry_with_backoff(max_retries=3)
    def _download_file(self, url: str) -> Path:
        """Download file from URL with retry logic"""
        self.logger.info("Downloading document", url=url)
        
        try:
            response = requests.get(url, timeout=self.download_timeout, stream=True)
            response.raise_for_status()
            
            # Create filename from URL
            parsed_url = urlparse(url)
            filename = Path(parsed_url.path).name or "downloaded_document.pdf"
            
            # Ensure .pdf extension
            if not filename.endswith('.pdf'):
                filename += '.pdf'
            
            # Create download path
            download_dir = Path('./data/downloads')
            download_dir.mkdir(parents=True, exist_ok=True)
            file_path = download_dir / filename
            
            # Check file size
            if 'content-length' in response.headers:
                size_mb = int(response.headers['content-length']) / (1024 * 1024)
                if size_mb > self.max_file_size_mb:
                    raise ValidationError(f"File too large: {size_mb:.1f}MB > {self.max_file_size_mb}MB")
            
            # Download with progress
            total_size = 0
            with open(file_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        total_size += len(chunk)
                        
                        # Check size during download
                        if total_size > self.max_file_size_mb * 1024 * 1024:
                            f.close()
                            file_path.unlink()  # Delete partial file
                            raise ValidationError(f"File too large during download")
            
            self.logger.info("Download completed", 
                           url=url, 
                           file_path=str(file_path),
                           size_bytes=total_size)
            
            return file_path
            
        except requests.RequestException as e:
            raise DocumentError(f"Failed to download document: {e}") from e
    
    def _validate_file(self, file_path: Path) -> None:
        """Validate downloaded/local file"""
        if not file_path.exists():
            raise ValidationError(f"File does not exist: {file_path}")
        
        if file_path.suffix.lower() not in self.supported_formats:
            raise ValidationError(f"Unsupported file format: {file_path.suffix}")
        
        # Check file size
        size_mb = file_path.stat().st_size / (1024 * 1024)
        if size_mb > self.max_file_size_mb:
            raise ValidationError(f"File too large: {size_mb:.1f}MB > {self.max_file_size_mb}MB")
        
        # Try to open PDF to validate
        try:
            reader = PdfReader(str(file_path))
            if len(reader.pages) == 0:
                raise ValidationError("PDF has no pages")
        except Exception as e:
            raise ValidationError(f"Invalid PDF file: {e}") from e
    
    def _load_pdf_pages(self, file_path: Path) -> List[Document]:
        """Load PDF pages using PyPDFLoader"""
        try:
            loader = PyPDFLoader(str(file_path))
            pages = loader.load()
            
            if not pages:
                raise DocumentError("No content could be extracted from PDF")
            
            self.logger.info("PDF loaded successfully", 
                           file_path=str(file_path),
                           page_count=len(pages))
            
            return pages
            
        except Exception as e:
            raise DocumentError(f"Failed to load PDF: {e}") from e
    
    def _extract_metadata(
        self, 
        pages: List[Document], 
        source: Union[str, Path],
        file_path: Path
    ) -> DocumentMetadata:
        """Extract metadata from PDF using LLM"""
        try:
            # Get first few pages for metadata extraction
            first_pages_text = ""
            pages_to_use = min(self.max_pages_for_metadata, len(pages))
            
            for i in range(pages_to_use):
                first_pages_text += pages[i].page_content + "\n\n"
            
            # Limit text size
            if len(first_pages_text) > self.metadata_char_limit:
                first_pages_text = first_pages_text[:self.metadata_char_limit]
            
            # Use LLM to extract metadata
            metadata_text = self._extract_metadata_with_llm(first_pages_text)
            
            # Parse metadata text into structured format
            parsed_metadata = self._parse_metadata_text(metadata_text)
            
            # Add file information
            parsed_metadata.source_url = str(source) if str(source).startswith('http') else None
            parsed_metadata.file_path = str(file_path)
            parsed_metadata.content_hash = self._calculate_content_hash(first_pages_text)
            parsed_metadata.page_count = len(pages)
            parsed_metadata.file_size_bytes = file_path.stat().st_size
            
            return parsed_metadata
            
        except Exception as e:
            self.logger.warning("Metadata extraction failed, using defaults", error=str(e))
            
            # Return default metadata
            return DocumentMetadata(
                title="Unknown Document",
                authors=["Unknown"],
                institutions=["Unknown"],
                publication_date="Unknown",
                arxiv_id="Not found",
                keywords=["research"],
                abstract="Abstract not available",
                source_url=str(source) if str(source).startswith('http') else None,
                file_path=str(file_path),
                content_hash=self._calculate_content_hash(""),
                page_count=len(pages),
                file_size_bytes=file_path.stat().st_size if file_path.exists() else 0
            )
    
    def _extract_metadata_with_llm(self, text: str) -> str:
        """Use LLM to extract metadata from text"""
        from ..utils.prompts import PromptManager
        
        prompt_manager = PromptManager()
        messages = prompt_manager.get_prompt('metadata_extraction').format(text=text)
        
        response = self.llm_client.generate_with_retry(messages)
        return response.content
    
    def _parse_metadata_text(self, metadata_text: str) -> DocumentMetadata:
        """Parse LLM-generated metadata text into structured format"""
        try:
            # Extract using regex patterns
            title_match = re.search(r'Title:\s*(.+?)(?:\n|$)', metadata_text)
            authors_match = re.search(r'Authors:\s*(.+?)(?:\n|$)', metadata_text)
            institutions_match = re.search(r'Institutions:\s*(.+?)(?:\n|$)', metadata_text)
            date_match = re.search(r'Publication Date:\s*(.+?)(?:\n|$)', metadata_text)
            arxiv_match = re.search(r'ArXiv ID:\s*(.+?)(?:\n|$)', metadata_text)
            keywords_match = re.search(r'Keywords:\s*(.+?)(?:\n|$)', metadata_text)
            abstract_match = re.search(r'Abstract:\s*(.+?)(?:\n--- END OF METADATA ---|$)', metadata_text, re.DOTALL)
            
            return DocumentMetadata(
                title=title_match.group(1).strip() if title_match else "Unknown Document",
                authors=self._parse_list_field(authors_match.group(1) if authors_match else "Unknown"),
                institutions=self._parse_list_field(institutions_match.group(1) if institutions_match else "Unknown"),
                publication_date=date_match.group(1).strip() if date_match else "Unknown",
                arxiv_id=arxiv_match.group(1).strip() if arxiv_match else "Not found",
                keywords=self._parse_list_field(keywords_match.group(1) if keywords_match else "research"),
                abstract=abstract_match.group(1).strip() if abstract_match else "Abstract not available"
            )
            
        except Exception as e:
            self.logger.warning("Failed to parse metadata text", error=str(e))
            raise DocumentError(f"Failed to parse metadata: {e}") from e
    
    def _parse_list_field(self, field_text: str) -> List[str]:
        """Parse comma-separated field into list"""
        if not field_text or field_text.strip().lower() in ['unknown', 'not found', '']:
            return ['Unknown']
        
        # Split by comma and clean
        items = [item.strip() for item in field_text.split(',')]
        return [item for item in items if item and item.lower() != 'unknown']
    
    def _create_document_chunks(self, pages: List[Document], metadata: DocumentMetadata) -> List[Document]:
        """Create document chunks from pages"""
        # Create metadata document
        metadata_content = self._format_metadata_for_storage(metadata)
        metadata_doc = Document(
            page_content=metadata_content,
            metadata={
                "source": metadata.file_path or "unknown",
                "page": "metadata",
                "type": "paper_metadata",
                "chunk_id": "metadata_0"
            }
        )
        
        # Add all page documents
        document_chunks = [metadata_doc]
        
        for i, page in enumerate(pages):
            # Update page metadata
            page.metadata.update({
                "type": "content",
                "chunk_id": f"page_{i}",
                "page_number": i + 1,
                "document_title": metadata.title,
                "document_authors": metadata.authors
            })
            document_chunks.append(page)
        
        return document_chunks
    
    def _format_metadata_for_storage(self, metadata: DocumentMetadata) -> str:
        """Format metadata for vector store storage"""
        authors_str = ', '.join(metadata.authors)
        institutions_str = ', '.join(metadata.institutions)
        keywords_str = ', '.join(metadata.keywords)
        
        return f"""PAPER METADATA:
Title: {metadata.title}
Authors: {authors_str}
Institutions: {institutions_str}
Publication Date: {metadata.publication_date}
ArXiv ID: {metadata.arxiv_id}
Keywords: {keywords_str}
Abstract: {metadata.abstract}
--- END OF METADATA ---

FIRST PAGE CONTENT:
{metadata.abstract}"""
    
    def _add_processing_metadata(
        self, 
        document_chunks: List[Document], 
        metadata: DocumentMetadata,
        document_id: Optional[str]
    ) -> None:
        """Add processing metadata to all chunks"""
        processing_id = document_id or self._generate_document_id(metadata)
        timestamp = self._get_timestamp()
        
        for chunk in document_chunks:
            chunk.metadata.update({
                "document_id": processing_id,
                "processed_at": timestamp,
                "content_hash": metadata.content_hash,
                "ingestion_version": "1.0.0"
            })
    
    def _generate_document_id(self, metadata: DocumentMetadata) -> str:
        """Generate unique document ID"""
        # Use title + first author + date for ID
        id_source = f"{metadata.title}_{metadata.authors[0]}_{metadata.publication_date}"
        return hashlib.md5(id_source.encode()).hexdigest()[:12]
    
    def _calculate_content_hash(self, content: str) -> str:
        """Calculate hash of content for caching"""
        return hashlib.sha256(content.encode()).hexdigest()[:16]
    
    def _get_timestamp(self) -> int:
        """Get current timestamp in milliseconds"""
        import time
        return int(time.time() * 1000)
    
    def get_supported_formats(self) -> List[str]:
        """Get list of supported file formats"""
        return self.supported_formats.copy()
    
    def validate_source(self, source: Union[str, Path]) -> Dict[str, Any]:
        """Validate source without processing - useful for API endpoints"""
        try:
            self._validate_source(source)
            
            # Additional validation info
            source_str = str(source)
            is_url = source_str.startswith('http')
            
            validation_result = {
                "valid": True,
                "source_type": "url" if is_url else "local_file",
                "source": source_str,
                "estimated_size_mb": None,
                "format": None
            }
            
            if not is_url:
                file_path = Path(source_str)
                validation_result.update({
                    "estimated_size_mb": file_path.stat().st_size / (1024 * 1024),
                    "format": file_path.suffix.lower()
                })
            
            return validation_result
            
        except Exception as e:
            return {
                "valid": False,
                "error": str(e),
                "source": str(source)
            }