"""
Document Store Coordination Layer
Coordinates between MinIO storage and other storage backends
"""

import logging
from typing import Dict, Any, Optional, List, Union
from abc import ABC, abstractmethod

from .minio_store import MinIODocumentStore, get_document_store

logger = logging.getLogger("rag.storage.document")


class DocumentStoreInterface(ABC):
    """Abstract interface for document storage backends"""
    
    @abstractmethod
    def store_chunk(self, rag_id: str, chunk_id: str, content: str, 
                   metadata: Dict[str, Any] = None, doc_id: Optional[str] = None) -> str:
        """Store chunk content and return URI"""
        pass
    
    @abstractmethod
    def get_chunk_content(self, uri: str) -> Optional[Dict[str, Any]]:
        """Retrieve chunk content by URI"""
        pass
    
    @abstractmethod
    def delete_rag_chunks(self, rag_id: str) -> Dict[str, Any]:
        """Delete all chunks for a RAG system"""
        pass
    
    @abstractmethod
    def health_check(self) -> Dict[str, Any]:
        """Check storage backend health"""
        pass


class DocumentStoreManager:
    """
    Document storage manager with multiple backend support
    
    Features:
    - Primary/fallback storage backends
    - Automatic failover
    - Health monitoring
    - Backend selection by URI scheme
    """
    
    def __init__(self, 
                 primary_backend: str = "minio",
                 minio_config: Dict[str, Any] = None):
        """
        Initialize document store manager
        
        Args:
            primary_backend: Primary storage backend ("minio", "local")
            minio_config: MinIO configuration parameters
        """
        self.primary_backend = primary_backend
        self.backends = {}
        
        # Initialize MinIO backend
        minio_config = minio_config or {}
        try:
            self.backends["minio"] = MinIODocumentStore(**minio_config)
            logger.info("MinIO document store initialized")
        except Exception as e:
            logger.error(f"Failed to initialize MinIO backend: {e}")
            self.backends["minio"] = None
        
        # Set active backend
        self.active_backend = self.backends.get(primary_backend)
        if not self.active_backend:
            logger.warning(f"Primary backend '{primary_backend}' not available, using fallback")
            # MinIO will automatically use local fallback if needed
            self.active_backend = self.backends.get("minio")
        
        logger.info(f"Document store manager ready with backend: {primary_backend}")
    
    def store_chunk(self, 
                   rag_id: str, 
                   chunk_id: str, 
                   content: str, 
                   metadata: Dict[str, Any] = None,
                   doc_id: Optional[str] = None) -> str:
        """
        Store chunk content using active backend
        
        Args:
            rag_id: RAG system identifier
            chunk_id: Unique chunk identifier
            content: Full chunk content
            metadata: Additional chunk metadata
            doc_id: Source document identifier
            
        Returns:
            URI for accessing the stored content
        """
        if not self.active_backend:
            raise RuntimeError("No storage backend available")
        
        return self.active_backend.store_chunk(rag_id, chunk_id, content, metadata, doc_id)
    
    def get_chunk_content(self, uri: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve chunk content by URI
        
        Args:
            uri: Chunk URI
            
        Returns:
            Chunk data dict or None if not found
        """
        # Route by URI scheme
        if uri.startswith("minio://") or uri.startswith("file://"):
            backend = self.backends.get("minio")
            if backend:
                return backend.get_chunk_content(uri)
        
        # Fallback to active backend
        if self.active_backend:
            return self.active_backend.get_chunk_content(uri)
        
        logger.error(f"No backend available to retrieve: {uri}")
        return None
    
    def store_chunks_batch(self, 
                          rag_id: str, 
                          chunks_data: List[Dict[str, Any]]) -> List[str]:
        """
        Store multiple chunks in batch
        
        Args:
            rag_id: RAG system identifier
            chunks_data: List of chunk data dicts
            
        Returns:
            List of URIs for stored chunks
        """
        if not self.active_backend:
            raise RuntimeError("No storage backend available")
        
        if hasattr(self.active_backend, 'store_chunks_batch'):
            return self.active_backend.store_chunks_batch(rag_id, chunks_data)
        else:
            # Fallback to individual storage
            uris = []
            for chunk_data in chunks_data:
                try:
                    uri = self.store_chunk(
                        rag_id=rag_id,
                        chunk_id=chunk_data["chunk_id"],
                        content=chunk_data["content"],
                        metadata=chunk_data.get("metadata", {}),
                        doc_id=chunk_data.get("doc_id")
                    )
                    uris.append(uri)
                except Exception as e:
                    logger.error(f"Failed to store chunk {chunk_data.get('chunk_id')}: {e}")
                    uris.append(None)
            return uris
    
    def delete_rag_chunks(self, rag_id: str) -> Dict[str, Any]:
        """
        Delete all chunks for a RAG system
        
        Args:
            rag_id: RAG system identifier
            
        Returns:
            Deletion summary
        """
        if not self.active_backend:
            raise RuntimeError("No storage backend available")
        
        return self.active_backend.delete_rag_chunks(rag_id)
    
    def get_rag_stats(self, rag_id: str) -> Dict[str, Any]:
        """
        Get storage statistics for a RAG system
        
        Args:
            rag_id: RAG system identifier
            
        Returns:
            Storage statistics
        """
        if not self.active_backend:
            return {"error": "No storage backend available"}
        
        if hasattr(self.active_backend, 'get_rag_stats'):
            return self.active_backend.get_rag_stats(rag_id)
        else:
            return {"error": "Storage stats not supported by backend"}
    
    def health_check(self) -> Dict[str, Any]:
        """
        Check all storage backends health
        
        Returns:
            Health status for all backends
        """
        health_status = {
            "primary_backend": self.primary_backend,
            "backends": {},
            "overall_healthy": False
        }
        
        healthy_count = 0
        
        for backend_name, backend in self.backends.items():
            if backend:
                try:
                    status = backend.health_check()
                    health_status["backends"][backend_name] = status
                    if status.get("healthy", False):
                        healthy_count += 1
                except Exception as e:
                    health_status["backends"][backend_name] = {
                        "healthy": False,
                        "error": str(e)
                    }
            else:
                health_status["backends"][backend_name] = {
                    "healthy": False,
                    "error": "Backend not initialized"
                }
        
        health_status["overall_healthy"] = healthy_count > 0
        health_status["healthy_backends"] = healthy_count
        
        return health_status
    
    def get_object_content(self, uri: str) -> str:
        """
        Get object content as string (compatibility method for MilvusRAGStore)
        
        Args:
            uri: Object URI
            
        Returns:
            Content as string
        """
        chunk_data = self.get_chunk_content(uri)
        if chunk_data and "content" in chunk_data:
            return chunk_data["content"]
        return ""


# Global instance management
_document_store_manager = None

def get_document_store_manager(**kwargs) -> DocumentStoreManager:
    """Get global document store manager instance"""
    global _document_store_manager
    if _document_store_manager is None:
        _document_store_manager = DocumentStoreManager(**kwargs)
    return _document_store_manager