"""
MinIO Document Store Implementation
S3-compatible document storage for RAG chunk content
"""

import json
import logging
import uuid
from typing import Dict, Any, Optional, List, Union
from datetime import datetime
from pathlib import Path

try:
    from minio import Minio
    from minio.error import S3Error, InvalidResponseError
    MINIO_AVAILABLE = True
except ImportError:
    MINIO_AVAILABLE = False

logger = logging.getLogger("rag.storage.minio")


class MinIODocumentStore:
    """
    S3-compatible document storage using MinIO
    
    Features:
    - Chunk content storage with URI-based access
    - Metadata preservation and versioning
    - Bucket lifecycle management
    - Error handling and retry logic
    - Development fallback to local storage
    """
    
    def __init__(self, 
                 endpoint: str = "localhost:9000",
                 access_key: str = "minioadmin",
                 secret_key: str = "minioadmin",
                 bucket_name: str = "rag-storage",
                 secure: bool = False,
                 region: Optional[str] = None):
        """
        Initialize MinIO document store
        
        Args:
            endpoint: MinIO server endpoint
            access_key: Access key for authentication
            secret_key: Secret key for authentication
            bucket_name: Bucket name for document storage
            secure: Whether to use HTTPS
            region: AWS region (optional)
        """
        if not MINIO_AVAILABLE:
            logger.warning("MinIO library not available, using local fallback")
            self.client = None
        else:
            try:
                self.client = Minio(
                    endpoint=endpoint,
                    access_key=access_key,
                    secret_key=secret_key,
                    secure=secure,
                    region=region
                )
                logger.info(f"MinIO client initialized: {endpoint}")
            except Exception as e:
                logger.error(f"Failed to initialize MinIO client: {e}")
                self.client = None
        
        self.bucket_name = bucket_name
        self.local_fallback_dir = Path("./rag_storage_fallback")
        
        # Ensure bucket exists
        self._ensure_bucket()
        
        logger.info(f"MinIO Document Store ready: bucket='{bucket_name}'")
    
    def _ensure_bucket(self):
        """Create bucket if it doesn't exist"""
        if self.client:
            try:
                if not self.client.bucket_exists(self.bucket_name):
                    self.client.make_bucket(self.bucket_name)
                    logger.info(f"Created bucket: {self.bucket_name}")
                else:
                    logger.debug(f"Bucket already exists: {self.bucket_name}")
            except S3Error as e:
                logger.error(f"Failed to create/check bucket: {e}")
                self.client = None
        
        # Create local fallback directory
        if not self.client:
            self.local_fallback_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Using local fallback storage: {self.local_fallback_dir}")
    
    def store_chunk(self, 
                   rag_id: str, 
                   chunk_id: str, 
                   content: str, 
                   metadata: Dict[str, Any] = None,
                   doc_id: Optional[str] = None) -> str:
        """
        Store chunk content and return URI
        
        Args:
            rag_id: RAG system identifier
            chunk_id: Unique chunk identifier
            content: Full chunk content
            metadata: Additional chunk metadata
            doc_id: Source document identifier
            
        Returns:
            URI for accessing the stored content
        """
        # Prepare chunk data
        chunk_data = {
            "rag_id": rag_id,
            "chunk_id": chunk_id,
            "doc_id": doc_id,
            "content": content,
            "metadata": metadata or {},
            "stored_at": datetime.now().isoformat(),
            "version": "1.0"
        }
        
        # Generate object key
        object_key = f"chunks/{rag_id}/{doc_id or 'unknown'}/{chunk_id}.json"
        
        try:
            if self.client:
                # Store in MinIO
                chunk_json = json.dumps(chunk_data, ensure_ascii=False, indent=None)
                chunk_bytes = chunk_json.encode('utf-8')
                
                self.client.put_object(
                    bucket_name=self.bucket_name,
                    object_name=object_key,
                    data=chunk_bytes,
                    length=len(chunk_bytes),
                    content_type="application/json"
                )
                
                uri = f"minio://{self.bucket_name}/{object_key}"
                logger.debug(f"Stored chunk in MinIO: {uri}")
                
            else:
                # Local fallback
                local_path = self.local_fallback_dir / object_key
                local_path.parent.mkdir(parents=True, exist_ok=True)
                
                with open(local_path, 'w', encoding='utf-8') as f:
                    json.dump(chunk_data, f, ensure_ascii=False, indent=2)
                
                uri = f"file://{local_path.absolute()}"
                logger.debug(f"Stored chunk locally: {uri}")
            
            return uri
            
        except Exception as e:
            logger.error(f"Failed to store chunk {chunk_id}: {e}")
            raise
    
    def get_chunk_content(self, uri: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve chunk content by URI
        
        Args:
            uri: Chunk URI (minio:// or file://)
            
        Returns:
            Chunk data dict or None if not found
        """
        try:
            if uri.startswith("minio://"):
                return self._get_from_minio(uri)
            elif uri.startswith("file://"):
                return self._get_from_file(uri)
            else:
                logger.error(f"Unsupported URI scheme: {uri}")
                return None
                
        except Exception as e:
            logger.error(f"Failed to retrieve chunk content from {uri}: {e}")
            return None
    
    def _get_from_minio(self, uri: str) -> Optional[Dict[str, Any]]:
        """Get chunk from MinIO storage"""
        if not self.client:
            logger.error("MinIO client not available")
            return None
        
        try:
            # Parse URI: minio://bucket/path/to/object.json
            parts = uri.replace("minio://", "").split("/", 1)
            bucket = parts[0]
            object_key = parts[1]
            
            response = self.client.get_object(bucket, object_key)
            chunk_json = response.read().decode('utf-8')
            response.close()
            response.release_conn()
            
            return json.loads(chunk_json)
            
        except S3Error as e:
            if e.code == "NoSuchKey":
                logger.debug(f"Chunk not found: {uri}")
            else:
                logger.error(f"MinIO error retrieving {uri}: {e}")
            return None
    
    def _get_from_file(self, uri: str) -> Optional[Dict[str, Any]]:
        """Get chunk from local file"""
        try:
            file_path = Path(uri.replace("file://", ""))
            
            if not file_path.exists():
                logger.debug(f"File not found: {file_path}")
                return None
            
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
                
        except Exception as e:
            logger.error(f"Error reading local file {uri}: {e}")
            return None
    
    def store_chunks_batch(self, 
                          rag_id: str, 
                          chunks_data: List[Dict[str, Any]]) -> List[str]:
        """
        Store multiple chunks in batch
        
        Args:
            rag_id: RAG system identifier
            chunks_data: List of chunk data dicts with 'chunk_id', 'content', etc.
            
        Returns:
            List of URIs for stored chunks
        """
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
                uris.append(None)  # Placeholder for failed chunks
        
        logger.info(f"Batch stored {len([u for u in uris if u])} of {len(chunks_data)} chunks")
        return uris
    
    def delete_rag_chunks(self, rag_id: str) -> Dict[str, Any]:
        """
        Delete all chunks for a RAG system
        
        Args:
            rag_id: RAG system identifier
            
        Returns:
            Deletion summary
        """
        deleted_count = 0
        errors = []
        
        try:
            prefix = f"chunks/{rag_id}/"
            
            if self.client:
                # Delete from MinIO
                objects = self.client.list_objects(self.bucket_name, prefix=prefix, recursive=True)
                
                for obj in objects:
                    try:
                        self.client.remove_object(self.bucket_name, obj.object_name)
                        deleted_count += 1
                    except S3Error as e:
                        errors.append(f"Failed to delete {obj.object_name}: {e}")
                        
            else:
                # Delete from local fallback
                rag_dir = self.local_fallback_dir / "chunks" / rag_id
                if rag_dir.exists():
                    import shutil
                    shutil.rmtree(rag_dir)
                    deleted_count = 1  # Directory deletion
            
            summary = {
                "rag_id": rag_id,
                "deleted_objects": deleted_count,
                "errors": errors,
                "success": len(errors) == 0
            }
            
            logger.info(f"Deleted {deleted_count} objects for RAG {rag_id}")
            return summary
            
        except Exception as e:
            logger.error(f"Failed to delete chunks for RAG {rag_id}: {e}")
            return {
                "rag_id": rag_id,
                "deleted_objects": 0,
                "errors": [str(e)],
                "success": False
            }
    
    def get_rag_stats(self, rag_id: str) -> Dict[str, Any]:
        """
        Get storage statistics for a RAG system
        
        Args:
            rag_id: RAG system identifier
            
        Returns:
            Storage statistics
        """
        try:
            prefix = f"chunks/{rag_id}/"
            object_count = 0
            total_size = 0
            
            if self.client:
                objects = self.client.list_objects(self.bucket_name, prefix=prefix, recursive=True)
                
                for obj in objects:
                    object_count += 1
                    total_size += obj.size or 0
                    
            else:
                # Count local files
                rag_dir = self.local_fallback_dir / "chunks" / rag_id
                if rag_dir.exists():
                    for file_path in rag_dir.rglob("*.json"):
                        object_count += 1
                        total_size += file_path.stat().st_size
            
            return {
                "rag_id": rag_id,
                "object_count": object_count,
                "total_size_bytes": total_size,
                "total_size_mb": total_size / (1024 * 1024),
                "storage_backend": "minio" if self.client else "local"
            }
            
        except Exception as e:
            logger.error(f"Failed to get stats for RAG {rag_id}: {e}")
            return {
                "rag_id": rag_id,
                "object_count": 0,
                "total_size_bytes": 0,
                "total_size_mb": 0,
                "error": str(e)
            }
    
    def health_check(self) -> Dict[str, Any]:
        """
        Check MinIO connection health
        
        Returns:
            Health status
        """
        try:
            if self.client:
                # Test connection by listing bucket
                self.client.bucket_exists(self.bucket_name)
                
                return {
                    "healthy": True,
                    "backend": "minio",
                    "bucket": self.bucket_name,
                    "timestamp": datetime.now().isoformat()
                }
            else:
                return {
                    "healthy": True,
                    "backend": "local_fallback",
                    "directory": str(self.local_fallback_dir),
                    "timestamp": datetime.now().isoformat()
                }
                
        except Exception as e:
            return {
                "healthy": False,
                "backend": "minio" if self.client else "local_fallback",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }


# Global instance management
_document_store_instance = None

def get_document_store(**kwargs) -> MinIODocumentStore:
    """Get global document store instance"""
    global _document_store_instance
    if _document_store_instance is None:
        _document_store_instance = MinIODocumentStore(**kwargs)
    return _document_store_instance