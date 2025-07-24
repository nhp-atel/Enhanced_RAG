"""
FAISS vector store implementation with persistence and optimization.
"""

import os
import json
import pickle
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import numpy as np

import faiss
from langchain_core.documents import Document

from ..interfaces import VectorStore, EmbeddingClient
from ..utils.errors import VectorStoreError
from ..utils.logging import get_logger


class FAISVectorStore(VectorStore):
    """FAISS-based vector store with persistence and optimization"""
    
    def __init__(self, config: Dict[str, Any], embedding_client: EmbeddingClient):
        self.config = config
        self.embedding_client = embedding_client
        self.logger = get_logger("faiss_store")
        
        # Configuration
        self.index_type = config.get('index_type', 'IndexFlatIP')
        self.similarity_metric = config.get('similarity_metric', 'cosine')
        self.persistence_enabled = config.get('persistence', {}).get('enabled', True)
        self.persistence_dir = Path(config.get('persistence', {}).get('directory', './data/indices'))
        self.save_every_n_docs = config.get('persistence', {}).get('save_every_n_docs', 100)
        
        # Initialize storage
        self.index: Optional[faiss.Index] = None
        self.documents: List[Document] = []
        self.document_ids: List[str] = []
        self.metadata: List[Dict[str, Any]] = []
        self.dimensions: Optional[int] = None
        
        # Statistics
        self.doc_count = 0
        self.last_save_count = 0
        
        # Create persistence directory
        if self.persistence_enabled:
            self.persistence_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger.info("FAISS vector store initialized",
                        index_type=self.index_type,
                        persistence_enabled=self.persistence_enabled,
                        persistence_dir=str(self.persistence_dir))
    
    def add_documents(self, documents: List[Document], **kwargs) -> List[str]:
        """Add documents to vector store"""
        if not documents:
            return []
        
        try:
            self.logger.info("Adding documents to vector store", count=len(documents))
            
            # Extract text content
            texts = [doc.page_content for doc in documents]
            
            # Generate embeddings
            embedding_response = self.embedding_client.embed_with_retry(texts)
            embeddings = embedding_response.embeddings
            
            # Initialize index if needed
            if self.index is None:
                self.dimensions = len(embeddings[0])
                self._initialize_index()
            
            # Convert embeddings to numpy array
            embedding_matrix = np.array(embeddings, dtype=np.float32)
            
            # Normalize for cosine similarity if needed
            if self.similarity_metric == 'cosine':
                faiss.normalize_L2(embedding_matrix)
            
            # Add to FAISS index
            start_id = self.doc_count
            self.index.add(embedding_matrix)
            
            # Store documents and metadata
            document_ids = []
            for i, doc in enumerate(documents):
                doc_id = f"doc_{start_id + i}"
                document_ids.append(doc_id)
                
                self.documents.append(doc)
                self.document_ids.append(doc_id)
                self.metadata.append(doc.metadata)
            
            self.doc_count += len(documents)
            
            # Auto-save if threshold reached
            if (self.persistence_enabled and 
                self.doc_count - self.last_save_count >= self.save_every_n_docs):
                self._auto_save()
            
            self.logger.info("Documents added successfully",
                           added_count=len(documents),
                           total_count=self.doc_count,
                           embedding_cost=embedding_response.cost_usd)
            
            return document_ids
            
        except Exception as e:
            self.logger.error("Failed to add documents", error=str(e))
            raise VectorStoreError(f"Failed to add documents: {e}", operation="add_documents")
    
    def similarity_search(self, query: str, k: int = 6, **kwargs) -> List[Document]:
        """Search for similar documents"""
        try:
            if self.index is None or self.doc_count == 0:
                self.logger.warning("No documents in vector store")
                return []
            
            # Get query embedding
            query_embedding = self.embedding_client.embed_query(query)
            query_vector = np.array([query_embedding], dtype=np.float32)
            
            # Normalize for cosine similarity
            if self.similarity_metric == 'cosine':
                faiss.normalize_L2(query_vector)
            
            # Perform search
            k = min(k, self.doc_count)  # Don't search for more docs than we have
            scores, indices = self.index.search(query_vector, k)
            
            # Convert results to documents
            results = []
            for i, (score, doc_idx) in enumerate(zip(scores[0], indices[0])):
                if doc_idx == -1:  # FAISS returns -1 for missing results
                    continue
                
                if doc_idx >= len(self.documents):
                    self.logger.warning("Invalid document index", index=doc_idx)
                    continue
                
                # Get document and add similarity score
                doc = self.documents[doc_idx]
                doc_copy = Document(
                    page_content=doc.page_content,
                    metadata={
                        **doc.metadata,
                        'similarity_score': float(score),
                        'search_rank': i
                    }
                )
                results.append(doc_copy)
            
            self.logger.debug("Similarity search completed",
                            query_length=len(query),
                            k=k,
                            results_count=len(results))
            
            return results
            
        except Exception as e:
            self.logger.error("Similarity search failed", error=str(e))
            raise VectorStoreError(f"Similarity search failed: {e}", operation="similarity_search")
    
    def similarity_search_with_score(self, query: str, k: int = 6, **kwargs) -> List[Tuple[Document, float]]:
        """Search with similarity scores"""
        try:
            documents = self.similarity_search(query, k, **kwargs)
            
            results = []
            for doc in documents:
                score = doc.metadata.get('similarity_score', 0.0)
                # Remove the score from metadata to avoid duplication
                clean_metadata = {k: v for k, v in doc.metadata.items() if k != 'similarity_score'}
                clean_doc = Document(page_content=doc.page_content, metadata=clean_metadata)
                results.append((clean_doc, score))
            
            return results
            
        except Exception as e:
            raise VectorStoreError(f"Similarity search with score failed: {e}")
    
    def delete_documents(self, document_ids: List[str]) -> bool:
        """Delete documents from vector store"""
        try:
            # FAISS doesn't support deletion directly
            # We would need to rebuild the index without the deleted documents
            # For now, just mark them as deleted in metadata
            
            deleted_count = 0
            for doc_id in document_ids:
                if doc_id in self.document_ids:
                    idx = self.document_ids.index(doc_id)
                    self.metadata[idx]['deleted'] = True
                    deleted_count += 1
            
            self.logger.info("Documents marked as deleted", count=deleted_count)
            return deleted_count > 0
            
        except Exception as e:
            self.logger.error("Failed to delete documents", error=str(e))
            return False
    
    def save_local(self, path: str) -> bool:
        """Save index to local storage"""
        try:
            save_path = Path(path)
            save_path.mkdir(parents=True, exist_ok=True)
            
            if self.index is None:
                self.logger.warning("No index to save")
                return False
            
            # Save FAISS index
            index_file = save_path / 'faiss_index.bin'
            faiss.write_index(self.index, str(index_file))
            
            # Save documents and metadata
            documents_file = save_path / 'documents.pkl'
            with open(documents_file, 'wb') as f:
                pickle.dump({
                    'documents': self.documents,
                    'document_ids': self.document_ids,
                    'metadata': self.metadata,
                    'doc_count': self.doc_count,
                    'dimensions': self.dimensions
                }, f)
            
            # Save configuration
            config_file = save_path / 'config.json'
            with open(config_file, 'w', encoding='utf-8') as f:
                json.dump({
                    'index_type': self.index_type,
                    'similarity_metric': self.similarity_metric,
                    'dimensions': self.dimensions,
                    'doc_count': self.doc_count,
                    'embedding_model': self.embedding_client.model
                }, f, indent=2)
            
            self.last_save_count = self.doc_count
            
            self.logger.info("Vector store saved successfully",
                           path=str(save_path),
                           doc_count=self.doc_count)
            
            return True
            
        except Exception as e:
            self.logger.error("Failed to save vector store", path=path, error=str(e))
            return False
    
    def load_local(self, path: str) -> bool:
        """Load index from local storage"""
        try:
            load_path = Path(path)
            
            if not load_path.exists():
                self.logger.warning("Vector store path does not exist", path=str(load_path))
                return False
            
            # Load FAISS index
            index_file = load_path / 'faiss_index.bin'
            if not index_file.exists():
                self.logger.error("FAISS index file not found", path=str(index_file))
                return False
            
            self.index = faiss.read_index(str(index_file))
            
            # Load documents and metadata
            documents_file = load_path / 'documents.pkl'
            if documents_file.exists():
                with open(documents_file, 'rb') as f:
                    data = pickle.load(f)
                    self.documents = data['documents']
                    self.document_ids = data['document_ids']
                    self.metadata = data['metadata']
                    self.doc_count = data['doc_count']
                    self.dimensions = data['dimensions']
            
            # Load configuration
            config_file = load_path / 'config.json'
            if config_file.exists():
                with open(config_file, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                    # Validate compatibility
                    if config.get('embedding_model') != self.embedding_client.model:
                        self.logger.warning("Embedding model mismatch",
                                          saved_model=config.get('embedding_model'),
                                          current_model=self.embedding_client.model)
            
            self.last_save_count = self.doc_count
            
            self.logger.info("Vector store loaded successfully",
                           path=str(load_path),
                           doc_count=self.doc_count,
                           dimensions=self.dimensions)
            
            return True
            
        except Exception as e:
            self.logger.error("Failed to load vector store", path=path, error=str(e))
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """Get vector store statistics"""
        stats = {
            'provider': 'faiss',
            'index_type': self.index_type,
            'similarity_metric': self.similarity_metric,
            'document_count': self.doc_count,
            'dimensions': self.dimensions,
            'index_initialized': self.index is not None,
            'persistence_enabled': self.persistence_enabled,
            'last_save_count': self.last_save_count
        }
        
        if self.index is not None:
            stats.update({
                'index_size_mb': self._estimate_index_size_mb(),
                'is_trained': self.index.is_trained if hasattr(self.index, 'is_trained') else True
            })
        
        return stats
    
    def _initialize_index(self) -> None:
        """Initialize FAISS index based on configuration"""
        if self.dimensions is None:
            raise VectorStoreError("Cannot initialize index without dimensions")
        
        try:
            if self.index_type == 'IndexFlatIP':
                # Inner product (good for cosine similarity with normalized vectors)
                self.index = faiss.IndexFlatIP(self.dimensions)
            elif self.index_type == 'IndexFlatL2':
                # L2 distance
                self.index = faiss.IndexFlatL2(self.dimensions)
            elif self.index_type == 'IndexIVFFlat':
                # IVF with flat vectors (good for larger datasets)
                nlist = min(100, max(1, self.doc_count // 10))  # Heuristic for nlist
                quantizer = faiss.IndexFlatIP(self.dimensions)
                self.index = faiss.IndexIVFFlat(quantizer, self.dimensions, nlist)
            elif self.index_type == 'IndexHNSW':
                # HNSW index (good balance of speed and accuracy)
                self.index = faiss.IndexHNSWFlat(self.dimensions, 32)
                self.index.hnsw.efConstruction = 40
                self.index.hnsw.efSearch = 16
            else:
                # Default to flat inner product
                self.logger.warning("Unknown index type, using IndexFlatIP", 
                                  index_type=self.index_type)
                self.index = faiss.IndexFlatIP(self.dimensions)
            
            self.logger.info("FAISS index initialized",
                           index_type=type(self.index).__name__,
                           dimensions=self.dimensions)
            
        except Exception as e:
            raise VectorStoreError(f"Failed to initialize FAISS index: {e}")
    
    def _auto_save(self) -> None:
        """Automatically save index when threshold is reached"""
        if self.persistence_enabled:
            save_path = self.persistence_dir / 'auto_save'
            success = self.save_local(str(save_path))
            if success:
                self.logger.info("Auto-save completed", path=str(save_path))
            else:
                self.logger.warning("Auto-save failed", path=str(save_path))
    
    def _estimate_index_size_mb(self) -> float:
        """Estimate index size in MB"""
        if self.index is None or self.dimensions is None:
            return 0.0
        
        # Rough estimation based on index type and dimensions
        bytes_per_vector = self.dimensions * 4  # float32
        total_bytes = self.doc_count * bytes_per_vector
        
        # Add overhead for index structure
        if isinstance(self.index, faiss.IndexHNSWFlat):
            # HNSW has more overhead
            total_bytes *= 1.5
        elif 'IVF' in type(self.index).__name__:
            # IVF has some overhead
            total_bytes *= 1.2
        
        return total_bytes / (1024 * 1024)
    
    def optimize_index(self) -> bool:
        """Optimize index for better performance"""
        try:
            if self.index is None or self.doc_count == 0:
                return False
            
            # Train index if needed (for IVF indices)
            if hasattr(self.index, 'is_trained') and not self.index.is_trained:
                if self.doc_count >= 100:  # Need enough data to train
                    # Get all embeddings for training
                    texts = [doc.page_content for doc in self.documents[:1000]]  # Limit for training
                    embedding_response = self.embedding_client.embed_with_retry(texts)
                    training_data = np.array(embedding_response.embeddings, dtype=np.float32)
                    
                    if self.similarity_metric == 'cosine':
                        faiss.normalize_L2(training_data)
                    
                    self.index.train(training_data)
                    self.logger.info("Index training completed")
            
            # Set search parameters for HNSW
            if isinstance(self.index, faiss.IndexHNSWFlat):
                # Tune ef parameter based on dataset size
                if self.doc_count > 10000:
                    self.index.hnsw.efSearch = 32
                elif self.doc_count > 1000:
                    self.index.hnsw.efSearch = 24
                else:
                    self.index.hnsw.efSearch = 16
            
            return True
            
        except Exception as e:
            self.logger.error("Index optimization failed", error=str(e))
            return False