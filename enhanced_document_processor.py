"""
Enhanced Document Processor with proper typing and documentation.
"""
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path
import logging
from dataclasses import dataclass
from abc import ABC, abstractmethod

from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from pypdf import PdfReader

logger = logging.getLogger(__name__)


@dataclass
class PaperMetadata:
    """Structured representation of research paper metadata."""
    title: str
    authors: List[str]
    institutions: List[str]
    publication_date: str
    arxiv_id: Optional[str] = None
    keywords: List[str] = None
    abstract: Optional[str] = None
    confidence_score: float = 0.0

    def __post_init__(self):
        """Validate and clean metadata after initialization."""
        if self.keywords is None:
            self.keywords = []
        self.title = self.title.strip()
        self.authors = [author.strip() for author in self.authors if author.strip()]


class DocumentProcessor(ABC):
    """Abstract base class for document processing strategies."""
    
    @abstractmethod
    def extract_metadata(self, content: str) -> PaperMetadata:
        """Extract metadata from document content."""
        pass
    
    @abstractmethod
    def process_document(self, source: Union[str, Path]) -> Tuple[List[Document], PaperMetadata]:
        """Process document and return chunks with metadata."""
        pass


class EnhancedPDFProcessor(DocumentProcessor):
    """Enhanced PDF processor with robust error handling and validation."""
    
    def __init__(
        self, 
        llm: ChatOpenAI,
        chunk_size: int = 800,
        chunk_overlap: int = 150,
        max_pages_for_metadata: int = 3
    ) -> None:
        """
        Initialize PDF processor.
        
        Args:
            llm: Language model for metadata extraction
            chunk_size: Size of text chunks for processing
            chunk_overlap: Overlap between consecutive chunks
            max_pages_for_metadata: Maximum pages to use for metadata extraction
        """
        self.llm = llm
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.max_pages_for_metadata = max_pages_for_metadata
        
    def extract_metadata(self, content: str) -> PaperMetadata:
        """
        Extract structured metadata from PDF content using LLM.
        
        Args:
            content: Raw text content from PDF pages
            
        Returns:
            PaperMetadata object with extracted information
            
        Raises:
            ValueError: If content is empty or extraction fails
        """
        if not content.strip():
            raise ValueError("Content cannot be empty")
            
        try:
            # Implementation with robust error handling
            logger.info("Extracting metadata from document content")
            
            # Your existing metadata extraction logic here
            # but wrapped in proper error handling
            
            return PaperMetadata(
                title="Extracted Title",
                authors=["Author 1", "Author 2"],
                institutions=["Institution 1"],
                publication_date="2024",
                confidence_score=0.85
            )
            
        except Exception as e:
            logger.error(f"Metadata extraction failed: {e}")
            return self._create_fallback_metadata()
    
    def _create_fallback_metadata(self) -> PaperMetadata:
        """Create fallback metadata when extraction fails."""
        return PaperMetadata(
            title="Unknown Paper",
            authors=["Unknown Author"],
            institutions=["Unknown Institution"],
            publication_date="Unknown",
            confidence_score=0.1
        )
    
    def process_document(
        self, 
        source: Union[str, Path]
    ) -> Tuple[List[Document], PaperMetadata]:
        """
        Process PDF document into chunks and extract metadata.
        
        Args:
            source: URL or file path to PDF
            
        Returns:
            Tuple of (document_chunks, paper_metadata)
            
        Raises:
            FileNotFoundError: If local file doesn't exist
            ConnectionError: If URL cannot be accessed
        """
        logger.info(f"Processing document from source: {source}")
        
        try:
            # Download or load PDF
            pdf_path = self._get_pdf_path(source)
            
            # Extract text and metadata
            reader = PdfReader(pdf_path)
            metadata_content = self._extract_metadata_content(reader)
            metadata = self.extract_metadata(metadata_content)
            
            # Create document chunks
            chunks = self._create_document_chunks(reader, metadata)
            
            logger.info(f"Successfully processed {len(chunks)} chunks")
            return chunks, metadata
            
        except Exception as e:
            logger.error(f"Document processing failed: {e}")
            raise
    
    def _get_pdf_path(self, source: Union[str, Path]) -> Path:
        """Get PDF file path, downloading if necessary."""
        # Implementation for handling URLs vs local paths
        pass
    
    def _extract_metadata_content(self, reader: PdfReader) -> str:
        """Extract content from first few pages for metadata."""
        content = ""
        max_pages = min(self.max_pages_for_metadata, len(reader.pages))
        
        for i in range(max_pages):
            try:
                content += reader.pages[i].extract_text() + "\n\n"
            except Exception as e:
                logger.warning(f"Failed to extract text from page {i}: {e}")
                
        return content
    
    def _create_document_chunks(
        self, 
        reader: PdfReader, 
        metadata: PaperMetadata
    ) -> List[Document]:
        """Create document chunks with proper metadata."""
        # Implementation for creating properly structured chunks
        pass


# Usage example with proper error handling
def process_research_paper(
    pdf_source: Union[str, Path],
    llm: ChatOpenAI
) -> Tuple[List[Document], PaperMetadata]:
    """
    High-level API for processing research papers.
    
    Args:
        pdf_source: URL or path to PDF file
        llm: Language model for processing
        
    Returns:
        Processed documents and metadata
    """
    processor = EnhancedPDFProcessor(llm)
    
    try:
        return processor.process_document(pdf_source)
    except Exception as e:
        logger.error(f"Failed to process paper {pdf_source}: {e}")
        raise