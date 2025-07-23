"""
Advanced FAISS vector store management with persistence and incremental updates.
"""
import faiss
import numpy as np
import pickle
import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import logging
from dataclasses import dataclass, asdict
from datetime import datetime
import threading

from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings

logger = logging.getLogger(__name__)


@dataclass
class IndexMetadata:
    """Metadata for FAISS index management."""
    created_at: str
    last_updated: str
    total_documents: int
    embedding_dimension: int
    index_type: str
    version: str = "1.0"
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'IndexMetadata':
        return cls(**data)


class VectorStoreManager:
    """Advanced FAISS vector store with persistence and incremental updates."""
    
    def __init__(
        self,
        embeddings: OpenAIEmbeddings,
        index_path: Path,
        index_type: str = "flat",
        enable_gpu: bool = False
    ):
        self.embeddings = embeddings
        self.index_path = Path(index_path)
        self.index_type = index_type
        self.enable_gpu = enable_gpu
        
        # Initialize index components
        self.index: Optional[faiss.Index] = None
        self.documents: List[Document] = []
        self.metadata: Optional[IndexMetadata] = None
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Ensure directory exists
        self.index_path.mkdir(parents=True, exist_ok=True)
        
        # Load existing index if available
        self._load_index()
    
    def _create_index(self, dimension: int, num_docs: int) -> faiss.Index:
        """Create appropriate FAISS index based on configuration."""
        logger.info(f"Creating {self.index_type} index for {num_docs} documents")
        
        if self.index_type == "flat":
            index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
            
        elif self.index_type == "ivf":
            # IVF (Inverted File) for large datasets
            nlist = min(100, max(1, num_docs // 100))  # Adaptive cluster count
            quantizer = faiss.IndexFlatIP(dimension)
            index = faiss.IndexIVFFlat(quantizer, dimension, nlist)
            
        elif self.index_type == "hnsw":
            # HNSW for fast approximate search
            M = 32  # Number of connections
            index = faiss.IndexHNSWFlat(dimension, M)
            index.hnsw.efConstruction = 64
            index.hnsw.efSearch = 32
            
        else:
            raise ValueError(f"Unsupported index type: {self.index_type}")
        
        # GPU support
        if self.enable_gpu and faiss.get_num_gpus() > 0:
            logger.info("Moving index to GPU")
            index = faiss.index_cpu_to_gpu(faiss.StandardGpuResources(), 0, index)
        
        return index
    
    def add_documents(
        self, 
        documents: List[Document], 
        batch_size: int = 100
    ) -> None:
        """
        Add documents to the vector store with batched processing.
        
        Args:
            documents: List of documents to add
            batch_size: Size of processing batches
        """
        with self._lock:
            logger.info(f"Adding {len(documents)} documents to vector store")
            
            # Generate embeddings in batches
            all_embeddings = []
            for i in range(0, len(documents), batch_size):
                batch = documents[i:i + batch_size]
                texts = [doc.page_content for doc in batch]
                
                logger.debug(f"Processing batch {i//batch_size + 1}")
                batch_embeddings = self.embeddings.embed_documents(texts)
                all_embeddings.extend(batch_embeddings)
            
            # Convert to numpy array
            embeddings_array = np.array(all_embeddings, dtype=np.float32)
            
            # Normalize for cosine similarity
            faiss.normalize_L2(embeddings_array)
            
            # Initialize index if needed
            if self.index is None:
                dimension = embeddings_array.shape[1]
                self.index = self._create_index(dimension, len(documents))
                
                # Train index if necessary (IVF)
                if hasattr(self.index, 'train'):
                    logger.info("Training index...")
                    self.index.train(embeddings_array)
            
            # Add vectors to index
            start_id = len(self.documents)
            self.index.add(embeddings_array)
            
            # Store documents
            self.documents.extend(documents)
            
            # Update metadata
            self._update_metadata()
            
            logger.info(f"Successfully added {len(documents)} documents")
    
    def search(
        self, 
        query: str, 
        k: int = 6,
        score_threshold: float = 0.0
    ) -> List[Tuple[Document, float]]:
        """
        Search for similar documents.
        
        Args:
            query: Search query
            k: Number of results to return
            score_threshold: Minimum similarity score
            
        Returns:
            List of (document, score) tuples
        """
        with self._lock:
            if self.index is None or len(self.documents) == 0:
                logger.warning("No documents in index")
                return []
            
            # Generate query embedding
            query_embedding = np.array(
                [self.embeddings.embed_query(query)], 
                dtype=np.float32
            )
            faiss.normalize_L2(query_embedding)
            
            # Search
            scores, indices = self.index.search(query_embedding, k)
            
            # Filter and format results
            results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx != -1 and score >= score_threshold:
                    doc = self.documents[idx]
                    results.append((doc, float(score)))
            
            logger.debug(f"Found {len(results)} results for query")
            return results
    
    def update_document(self, doc_id: int, new_document: Document) -> None:
        """Update a document in the vector store."""
        with self._lock:
            if doc_id >= len(self.documents):
                raise IndexError(f"Document ID {doc_id} out of range")
            
            # Note: FAISS doesn't support in-place updates efficiently
            # For production, consider rebuilding periodically or using a hybrid approach
            logger.warning("Document updates require index rebuild for optimal performance")
            
            self.documents[doc_id] = new_document
            # TODO: Implement efficient update strategy
    
    def save_index(self) -> None:
        """Save index and metadata to disk."""
        with self._lock:
            if self.index is None:
                logger.warning("No index to save")
                return
            
            logger.info(f"Saving index to {self.index_path}")
            
            # Save FAISS index
            index_file = self.index_path / "index.faiss"
            faiss.write_index(self.index, str(index_file))
            
            # Save documents
            docs_file = self.index_path / "documents.pkl"
            with open(docs_file, 'wb') as f:
                pickle.dump(self.documents, f)
            
            # Save metadata
            metadata_file = self.index_path / "metadata.json"
            with open(metadata_file, 'w') as f:
                json.dump(self.metadata.to_dict(), f, indent=2)
            
            logger.info("Index saved successfully")
    
    def _load_index(self) -> None:
        """Load existing index from disk."""
        try:
            index_file = self.index_path / "index.faiss"
            docs_file = self.index_path / "documents.pkl"
            metadata_file = self.index_path / "metadata.json"
            
            if not all(f.exists() for f in [index_file, docs_file, metadata_file]):
                logger.info("No existing index found")
                return
            
            # Load FAISS index
            self.index = faiss.read_index(str(index_file))
            
            # Load documents
            with open(docs_file, 'rb') as f:
                self.documents = pickle.load(f)
            
            # Load metadata
            with open(metadata_file, 'r') as f:
                metadata_dict = json.load(f)
                self.metadata = IndexMetadata.from_dict(metadata_dict)
            
            logger.info(f"Loaded index with {len(self.documents)} documents")
            
        except Exception as e:
            logger.error(f"Failed to load index: {e}")
            self.index = None
            self.documents = []
            self.metadata = None
    
    def _update_metadata(self) -> None:
        """Update index metadata."""
        dimension = self.index.d if self.index else 0
        
        self.metadata = IndexMetadata(
            created_at=self.metadata.created_at if self.metadata else datetime.now().isoformat(),
            last_updated=datetime.now().isoformat(),
            total_documents=len(self.documents),
            embedding_dimension=dimension,
            index_type=self.index_type
        )
    
    def get_stats(self) -> Dict[str, Any]:
        """Get vector store statistics."""
        with self._lock:
            return {
                'total_documents': len(self.documents),
                'index_type': self.index_type,
                'embedding_dimension': self.index.d if self.index else 0,
                'index_size_mb': self._get_index_size_mb(),
                'metadata': self.metadata.to_dict() if self.metadata else None
            }
    
    def _get_index_size_mb(self) -> float:
        """Calculate approximate index size in MB."""
        if not self.index:
            return 0.0
        
        # Rough estimation based on number of vectors and dimension
        num_vectors = self.index.ntotal
        dimension = self.index.d
        bytes_per_vector = dimension * 4  # float32
        total_bytes = num_vectors * bytes_per_vector
        return total_bytes / (1024 * 1024)
    
    def cleanup(self) -> None:
        """Clean up resources."""
        with self._lock:
            if self.index and hasattr(self.index, 'reset'):
                self.index.reset()
            self.index = None
            self.documents.clear()
            logger.info("Vector store cleaned up")


# Usage example
def main():
    """Example usage of VectorStoreManager."""
    from langchain_openai import OpenAIEmbeddings
    from langchain_core.documents import Document
    
    # Initialize
    embeddings = OpenAIEmbeddings()
    manager = VectorStoreManager(
        embeddings=embeddings,
        index_path=Path("./vector_store"),
        index_type="hnsw"
    )
    
    # Add documents
    docs = [
        Document(page_content="Sample document 1", metadata={"id": 1}),
        Document(page_content="Sample document 2", metadata={"id": 2}),
    ]
    
    manager.add_documents(docs)
    
    # Search
    results = manager.search("sample", k=5)
    print(f"Found {len(results)} results")
    
    # Save
    manager.save_index()
    
    # Stats
    stats = manager.get_stats()
    print(f"Index stats: {stats}")


if __name__ == "__main__":
    main()