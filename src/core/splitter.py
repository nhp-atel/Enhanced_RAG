"""
Document splitting module - handles intelligent chunking based on content and model constraints.
"""

import re
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from ..interfaces import LLMClient, LoggerInterface, MetricsInterface
from ..utils.errors import ProcessingError


class SplitStrategy(Enum):
    """Different strategies for document splitting"""
    FIXED_SIZE = "fixed_size"
    SEMANTIC = "semantic"  
    SECTION_AWARE = "section_aware"
    ADAPTIVE = "adaptive"


@dataclass
class ChunkMetadata:
    """Metadata for document chunks"""
    chunk_id: str
    chunk_index: int
    total_chunks: int
    chunk_type: str  # content, metadata, summary, concept
    parent_document_id: str
    section_title: Optional[str] = None
    semantic_score: Optional[float] = None
    token_count: Optional[int] = None
    char_count: int = 0


class DocumentSplitter:
    """Intelligent document splitter with adaptive chunking strategies"""
    
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
        self.chunk_size = config.get('chunk_size', 800)
        self.chunk_overlap = config.get('chunk_overlap', 150)
        self.separators = config.get('separators', ["\n\n", "\n", ". ", " ", ""])
        self.strategy = SplitStrategy(config.get('split_strategy', 'adaptive'))
        self.max_chunk_size = config.get('max_chunk_size', 2000)
        self.min_chunk_size = config.get('min_chunk_size', 100)
        
        # Initialize base splitter
        self.base_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=self.separators,
            length_function=len
        )
        
        # Section patterns for academic papers
        self.section_patterns = [
            r'^\s*(?:Abstract|ABSTRACT)\s*$',
            r'^\s*(?:\d+\.?\s*)?(?:Introduction|INTRODUCTION)\s*$',
            r'^\s*(?:\d+\.?\s*)?(?:Related Work|RELATED WORK|Literature Review)\s*$',
            r'^\s*(?:\d+\.?\s*)?(?:Methodology|METHODOLOGY|Methods|METHODS)\s*$',
            r'^\s*(?:\d+\.?\s*)?(?:Results|RESULTS|Findings|FINDINGS)\s*$',
            r'^\s*(?:\d+\.?\s*)?(?:Discussion|DISCUSSION)\s*$',
            r'^\s*(?:\d+\.?\s*)?(?:Conclusion|CONCLUSIONS|CONCLUSION)\s*$',
            r'^\s*(?:\d+\.?\s*)?(?:References|REFERENCES|Bibliography)\s*$',
            r'^\s*(?:\d+\.?\s*)?(?:Appendix|APPENDIX)\s*$'
        ]
    
    def split_documents(
        self, 
        documents: List[Document], 
        strategy: Optional[SplitStrategy] = None
    ) -> List[Document]:
        """
        Split documents using specified or configured strategy
        
        Args:
            documents: List of documents to split
            strategy: Optional strategy override
            
        Returns:
            List of split document chunks
        """
        start_time = self._get_timestamp()
        
        try:
            self.logger.info("Starting document splitting", 
                           document_count=len(documents),
                           strategy=strategy or self.strategy)
            
            self.metrics.increment_counter("documents.split.started")
            
            # Use provided strategy or default
            split_strategy = strategy or self.strategy
            
            # Route to appropriate splitting method
            if split_strategy == SplitStrategy.FIXED_SIZE:
                chunks = self._split_fixed_size(documents)
            elif split_strategy == SplitStrategy.SEMANTIC:
                chunks = self._split_semantic(documents)
            elif split_strategy == SplitStrategy.SECTION_AWARE:
                chunks = self._split_section_aware(documents)
            elif split_strategy == SplitStrategy.ADAPTIVE:
                chunks = self._split_adaptive(documents)
            else:
                raise ProcessingError(f"Unknown split strategy: {split_strategy}")
            
            # Post-process chunks
            processed_chunks = self._post_process_chunks(chunks)
            
            # Add chunk metadata
            final_chunks = self._add_chunk_metadata(processed_chunks)
            
            # Log success
            processing_time = self._get_timestamp() - start_time
            self.logger.info("Document splitting completed",
                           input_documents=len(documents),
                           output_chunks=len(final_chunks),
                           strategy=str(split_strategy),
                           processing_time_ms=processing_time)
            
            self.metrics.record_histogram("documents.split.duration_ms", processing_time)
            self.metrics.record_histogram("documents.split.chunks_per_doc", len(final_chunks) / len(documents))
            self.metrics.increment_counter("documents.split.success")
            
            return final_chunks
            
        except Exception as e:
            processing_time = self._get_timestamp() - start_time
            self.logger.error("Document splitting failed",
                            error=str(e),
                            processing_time_ms=processing_time)
            
            self.metrics.increment_counter("documents.split.failed")
            raise ProcessingError(f"Failed to split documents: {e}") from e
    
    def _split_fixed_size(self, documents: List[Document]) -> List[Document]:
        """Split using fixed size chunks"""
        self.logger.debug("Using fixed size splitting strategy")
        
        all_chunks = []
        for doc in documents:
            # Skip metadata documents - keep them intact
            if doc.metadata.get('type') == 'paper_metadata':
                all_chunks.append(doc)
                continue
            
            # Split document
            chunks = self.base_splitter.split_documents([doc])
            all_chunks.extend(chunks)
        
        return all_chunks
    
    def _split_semantic(self, documents: List[Document]) -> List[Document]:
        """Split using semantic boundaries (experimental)"""
        self.logger.debug("Using semantic splitting strategy")
        
        # For now, use section-aware splitting as semantic proxy
        # In future, could use sentence embeddings to find semantic boundaries
        return self._split_section_aware(documents)
    
    def _split_section_aware(self, documents: List[Document]) -> List[Document]:
        """Split documents respecting section boundaries"""
        self.logger.debug("Using section-aware splitting strategy")
        
        all_chunks = []
        
        for doc in documents:
            # Skip metadata documents
            if doc.metadata.get('type') == 'paper_metadata':
                all_chunks.append(doc)
                continue
            
            # Find section boundaries
            sections = self._identify_sections(doc.page_content)
            
            if len(sections) <= 1:
                # No clear sections found, use fixed size
                chunks = self.base_splitter.split_documents([doc])
                all_chunks.extend(chunks)
            else:
                # Split by sections, then by size if sections are too large
                for section_title, section_content in sections:
                    if len(section_content) <= self.max_chunk_size:
                        # Section fits in one chunk
                        section_doc = Document(
                            page_content=section_content,
                            metadata={
                                **doc.metadata,
                                "section_title": section_title,
                                "chunk_type": "section"
                            }
                        )
                        all_chunks.append(section_doc)
                    else:
                        # Section too large, split further
                        section_doc = Document(
                            page_content=section_content,
                            metadata={
                                **doc.metadata,
                                "section_title": section_title
                            }
                        )
                        sub_chunks = self.base_splitter.split_documents([section_doc])
                        all_chunks.extend(sub_chunks)
        
        return all_chunks
    
    def _split_adaptive(self, documents: List[Document]) -> List[Document]:
        """Adaptive splitting based on content characteristics"""
        self.logger.debug("Using adaptive splitting strategy")
        
        all_chunks = []
        
        for doc in documents:
            # Skip metadata documents
            if doc.metadata.get('type') == 'paper_metadata':
                all_chunks.append(doc)
                continue
            
            # Analyze document characteristics
            doc_analysis = self._analyze_document_structure(doc)
            
            # Choose splitting approach based on analysis
            if doc_analysis['has_clear_sections']:
                # Use section-aware splitting
                sections = self._identify_sections(doc.page_content)
                for section_title, section_content in sections:
                    if len(section_content) <= self.max_chunk_size:
                        section_doc = Document(
                            page_content=section_content,
                            metadata={
                                **doc.metadata,
                                "section_title": section_title,
                                "chunk_type": "section"
                            }
                        )
                        all_chunks.append(section_doc)
                    else:
                        # Adaptive chunk size based on content density
                        adaptive_size = self._calculate_adaptive_chunk_size(section_content)
                        adaptive_splitter = RecursiveCharacterTextSplitter(
                            chunk_size=adaptive_size,
                            chunk_overlap=min(self.chunk_overlap, adaptive_size // 4),
                            separators=self.separators
                        )
                        section_doc = Document(
                            page_content=section_content,
                            metadata={**doc.metadata, "section_title": section_title}
                        )
                        sub_chunks = adaptive_splitter.split_documents([section_doc])
                        all_chunks.extend(sub_chunks)
            else:
                # Use adaptive fixed-size splitting
                adaptive_size = self._calculate_adaptive_chunk_size(doc.page_content)
                adaptive_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=adaptive_size,
                    chunk_overlap=min(self.chunk_overlap, adaptive_size // 4),
                    separators=self.separators
                )
                chunks = adaptive_splitter.split_documents([doc])
                all_chunks.extend(chunks)
        
        return all_chunks
    
    def _identify_sections(self, text: str) -> List[Tuple[str, str]]:
        """Identify sections in academic paper text"""
        sections = []
        current_section = "Introduction"
        current_content = []
        
        lines = text.split('\n')
        
        for line in lines:
            # Check if line matches a section pattern
            section_match = None
            for pattern in self.section_patterns:
                if re.match(pattern, line.strip(), re.IGNORECASE):
                    section_match = line.strip()
                    break
            
            if section_match:
                # Save previous section
                if current_content:
                    sections.append((current_section, '\n'.join(current_content)))
                
                # Start new section
                current_section = section_match
                current_content = []
            else:
                current_content.append(line)
        
        # Add final section
        if current_content:
            sections.append((current_section, '\n'.join(current_content)))
        
        return sections
    
    def _analyze_document_structure(self, doc: Document) -> Dict[str, Any]:
        """Analyze document structure to inform splitting strategy"""
        text = doc.page_content
        
        # Basic analysis
        analysis = {
            'length': len(text),
            'line_count': len(text.split('\n')),
            'paragraph_count': len([p for p in text.split('\n\n') if p.strip()]),
            'has_clear_sections': False,
            'section_count': 0,
            'avg_paragraph_length': 0,
            'content_density': 0,
            'has_equations': bool(re.search(r'\$.*?\$|\\\[.*?\\\]', text)),
            'has_citations': bool(re.search(r'\[\d+\]|\([^)]+\s+\d{4}\)', text)),
            'has_code': bool(re.search(r'```|def |class |import |#include', text))
        }
        
        # Section analysis
        sections = self._identify_sections(text)
        analysis['section_count'] = len(sections)
        analysis['has_clear_sections'] = len(sections) > 2
        
        # Content density (characters per line)
        if analysis['line_count'] > 0:
            analysis['content_density'] = analysis['length'] / analysis['line_count']
        
        # Average paragraph length
        paragraphs = [p for p in text.split('\n\n') if p.strip()]
        if paragraphs:
            analysis['avg_paragraph_length'] = sum(len(p) for p in paragraphs) / len(paragraphs)
        
        return analysis
    
    def _calculate_adaptive_chunk_size(self, text: str) -> int:
        """Calculate adaptive chunk size based on content characteristics"""
        base_size = self.chunk_size
        
        # Content density adjustment
        lines = text.split('\n')
        if lines:
            avg_line_length = sum(len(line) for line in lines) / len(lines)
            
            # If lines are very long (dense content), use smaller chunks
            if avg_line_length > 100:
                base_size = int(base_size * 0.8)
            # If lines are short (sparse content), use larger chunks  
            elif avg_line_length < 50:
                base_size = int(base_size * 1.2)
        
        # Has equations or code - use smaller chunks for precision
        if re.search(r'\$.*?\$|\\\[.*?\\\]|```|def |class ', text):
            base_size = int(base_size * 0.7)
        
        # Ensure within bounds
        return max(self.min_chunk_size, min(self.max_chunk_size, base_size))
    
    def _post_process_chunks(self, chunks: List[Document]) -> List[Document]:
        """Post-process chunks to ensure quality"""
        processed_chunks = []
        
        for chunk in chunks:
            # Skip empty chunks
            if not chunk.page_content.strip():
                continue
            
            # Skip chunks that are too small (unless they're metadata)
            if (len(chunk.page_content) < self.min_chunk_size and 
                chunk.metadata.get('type') != 'paper_metadata'):
                # Try to merge with previous chunk
                if processed_chunks and len(processed_chunks[-1].page_content) < self.max_chunk_size:
                    prev_chunk = processed_chunks[-1]
                    merged_content = prev_chunk.page_content + "\n\n" + chunk.page_content
                    
                    if len(merged_content) <= self.max_chunk_size:
                        prev_chunk.page_content = merged_content
                        continue
            
            # Clean up content
            cleaned_content = self._clean_chunk_content(chunk.page_content)
            chunk.page_content = cleaned_content
            
            processed_chunks.append(chunk)
        
        return processed_chunks
    
    def _clean_chunk_content(self, content: str) -> str:
        """Clean up chunk content"""
        # Remove excessive whitespace
        content = re.sub(r'\n\s*\n\s*\n', '\n\n', content)
        content = re.sub(r'[ \t]+', ' ', content)
        
        # Remove page headers/footers (common patterns)
        content = re.sub(r'^\d+\s*$', '', content, flags=re.MULTILINE)
        content = re.sub(r'^Page \d+ of \d+\s*$', '', content, flags=re.MULTILINE)
        
        return content.strip()
    
    def _add_chunk_metadata(self, chunks: List[Document]) -> List[Document]:
        """Add comprehensive chunk metadata"""
        total_chunks = len(chunks)
        
        for i, chunk in enumerate(chunks):
            # Count tokens if possible
            token_count = None
            try:
                token_count = self.llm_client.count_tokens(chunk.page_content)
            except:
                # Rough estimation if token counting fails
                token_count = len(chunk.page_content) // 4
            
            # Create chunk metadata
            chunk_metadata = ChunkMetadata(
                chunk_id=f"{chunk.metadata.get('document_id', 'unknown')}_{i}",
                chunk_index=i,
                total_chunks=total_chunks,
                chunk_type=chunk.metadata.get('type', 'content'),
                parent_document_id=chunk.metadata.get('document_id', 'unknown'),
                section_title=chunk.metadata.get('section_title'),
                token_count=token_count,
                char_count=len(chunk.page_content)
            )
            
            # Update chunk metadata
            chunk.metadata.update({
                'chunk_id': chunk_metadata.chunk_id,
                'chunk_index': chunk_metadata.chunk_index,
                'total_chunks': chunk_metadata.total_chunks,
                'token_count': chunk_metadata.token_count,
                'char_count': chunk_metadata.char_count,
                'split_strategy': str(self.strategy)
            })
        
        return chunks
    
    def get_optimal_chunk_size(self, model_context_length: int, overhead_tokens: int = 500) -> int:
        """Calculate optimal chunk size based on model context window"""
        # Reserve space for query, prompt, and response
        available_tokens = model_context_length - overhead_tokens
        
        # Convert tokens to characters (rough estimate: 1 token ≈ 4 characters)
        optimal_chars = available_tokens * 4
        
        # Ensure within configured bounds
        return max(self.min_chunk_size, min(self.max_chunk_size, optimal_chars))
    
    def estimate_chunks(self, text_length: int, strategy: Optional[SplitStrategy] = None) -> Dict[str, Any]:
        """Estimate chunking results without actually splitting"""
        split_strategy = strategy or self.strategy
        
        if split_strategy == SplitStrategy.FIXED_SIZE:
            estimated_chunks = (text_length // self.chunk_size) + 1
        else:
            # More complex estimation for adaptive strategies
            estimated_chunks = max(1, int(text_length / (self.chunk_size * 0.8)))
        
        return {
            'estimated_chunks': estimated_chunks,
            'strategy': str(split_strategy),
            'avg_chunk_size': text_length / estimated_chunks if estimated_chunks > 0 else 0,
            'total_characters': text_length
        }
    
    def _get_timestamp(self) -> int:
        """Get current timestamp in milliseconds"""
        import time
        return int(time.time() * 1000)