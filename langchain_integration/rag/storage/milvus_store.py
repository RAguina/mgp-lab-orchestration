"""
Milvus Vector Store Implementation
Single collection with partitions for multi-tenancy
"""

from typing import List, Dict, Any, Optional
import logging
from datetime import datetime
import json

try:
    from pymilvus import (
        Collection, CollectionSchema, FieldSchema, DataType, 
        utility, connections
    )
    MILVUS_AVAILABLE = True
except ImportError:
    MILVUS_AVAILABLE = False

logger = logging.getLogger("rag.storage.milvus")


class MilvusRAGStore:
    """Milvus vector store with partition-based RAG isolation"""
    
    def __init__(self, 
                 host: str = "localhost", 
                 port: int = 19530,
                 embed_dim: int = 1024,
                 collection_name: str = "ai_lab_chunks"):
        
        if not MILVUS_AVAILABLE:
            raise ImportError("pymilvus not installed. Run: pip install pymilvus>=2.4.0")
            
        self.host = host
        self.port = port
        self.embed_dim = embed_dim
        self.collection_name = collection_name
        
        self._connect()
        self._ensure_collection()
    
    def _connect(self):
        """Connect to Milvus server"""
        try:
            # Check if already connected
            if connections.has_connection("default"):
                connections.remove_connection("default")
                
            connections.connect(
                alias="default",
                host=self.host,
                port=self.port
            )
            logger.info(f"Connected to Milvus at {self.host}:{self.port}")
            
        except Exception as e:
            logger.error(f"Failed to connect to Milvus: {e}")
            raise
    
    def _ensure_collection(self):
        """Create collection if it doesn't exist"""
        try:
            if utility.has_collection(self.collection_name):
                logger.info(f"Collection '{self.collection_name}' already exists")
                return
                
            # Define schema
            schema = CollectionSchema([
                FieldSchema("id", DataType.INT64, is_primary=True, auto_id=True),
                FieldSchema("vector", DataType.FLOAT_VECTOR, dim=self.embed_dim),
                FieldSchema("rag_id", DataType.VARCHAR, max_length=64),
                FieldSchema("doc_id", DataType.VARCHAR, max_length=64),
                FieldSchema("chunk_id", DataType.VARCHAR, max_length=64),
                # Storage optimization: URI + excerpt instead of full content
                FieldSchema("uri", DataType.VARCHAR, max_length=256),
                FieldSchema("excerpt", DataType.VARCHAR, max_length=512),
                FieldSchema("metadata", DataType.JSON),
                FieldSchema("quality_score", DataType.FLOAT),
                FieldSchema("created_at", DataType.VARCHAR, max_length=32),
            ], enable_dynamic_field=True)
            
            # Create collection
            collection = Collection(self.collection_name, schema)
            logger.info(f"Created collection '{self.collection_name}'")
            
            # Create HNSW index (GPT-5 recommended parameters)
            index_params = {
                "index_type": "HNSW",
                "metric_type": "COSINE",
                "params": {
                    "M": 32,
                    "efConstruction": 200
                }
            }
            
            collection.create_index("vector", index_params)
            logger.info("Created HNSW index on vector field")
            
        except Exception as e:
            logger.error(f"Failed to ensure collection: {e}")
            raise
    
    def create_rag_partition(self, rag_id: str):
        """Create partition for RAG isolation"""
        try:
            collection = Collection(self.collection_name)
            partition_name = f"rag_{rag_id}"
            
            if not collection.has_partition(partition_name):
                collection.create_partition(partition_name)
                logger.info(f"Created partition: {partition_name}")
            else:
                logger.info(f"Partition already exists: {partition_name}")
                
        except Exception as e:
            logger.error(f"Failed to create partition for {rag_id}: {e}")
            raise
    
    def store_chunks(self, 
                    rag_id: str, 
                    chunks: List[Dict[str, Any]], 
                    embeddings: List[List[float]]):
        """Store chunks with embeddings in partitioned collection"""
        try:
            if len(chunks) != len(embeddings):
                raise ValueError("Chunks and embeddings length mismatch")
                
            # Ensure partition exists
            self.create_rag_partition(rag_id)
            
            collection = Collection(self.collection_name)
            partition_name = f"rag_{rag_id}"
            
            # Prepare data
            data = self._prepare_insert_data(rag_id, chunks, embeddings)
            
            # Insert data
            collection.insert(data, partition_name=partition_name)
            collection.flush()
            
            logger.info(f"Stored {len(chunks)} chunks in partition {partition_name}")
            
        except Exception as e:
            logger.error(f"Failed to store chunks for {rag_id}: {e}")
            raise
    
    def search(self, 
              rag_id: str, 
              query_embedding: List[float], 
              top_k: int = 50,
              ef_search: int = 96,
              include_full_content: bool = False) -> List[Dict[str, Any]]:
        """Search within specific RAG partition"""
        try:
            collection = Collection(self.collection_name)
            partition_name = f"rag_{rag_id}"
            
            # Load partition
            collection.load(partition_names=[partition_name])
            
            # Search parameters
            search_params = {
                "metric_type": "COSINE",
                "params": {"ef": ef_search}
            }
            
            # Output fields
            output_fields = ["uri", "excerpt", "metadata", "quality_score", 
                           "doc_id", "chunk_id", "created_at"]
            
            # Perform search
            results = collection.search(
                data=[query_embedding],
                anns_field="vector",
                param=search_params,
                limit=top_k,
                partition_names=[partition_name],
                output_fields=output_fields
            )
            
            return self._format_results(results, include_full_content)
            
        except Exception as e:
            logger.error(f"Failed to search in {rag_id}: {e}")
            raise
    
    def delete_rag(self, rag_id: str):
        """Delete RAG partition and all its data"""
        try:
            collection = Collection(self.collection_name)
            partition_name = f"rag_{rag_id}"
            
            if collection.has_partition(partition_name):
                collection.drop_partition(partition_name)
                logger.info(f"Deleted partition: {partition_name}")
            else:
                logger.warning(f"Partition does not exist: {partition_name}")
                
        except Exception as e:
            logger.error(f"Failed to delete RAG {rag_id}: {e}")
            raise
    
    def _prepare_insert_data(self, 
                           rag_id: str, 
                           chunks: List[Dict[str, Any]], 
                           embeddings: List[List[float]]) -> List[List]:
        """Prepare data for insertion"""
        current_time = datetime.now().isoformat()
        
        vectors = []
        rag_ids = []
        doc_ids = []
        chunk_ids = []
        uris = []
        excerpts = []
        metadata_list = []
        quality_scores = []
        created_ats = []
        
        for chunk, embedding in zip(chunks, embeddings):
            vectors.append(embedding)
            rag_ids.append(rag_id)
            doc_ids.append(chunk.get("doc_id", "unknown"))
            chunk_ids.append(chunk.get("chunk_id", f"chunk_{len(vectors)}"))
            uris.append(chunk.get("uri", ""))
            excerpts.append(chunk.get("content", "")[:500])  # Truncate for excerpt
            metadata_list.append(chunk.get("metadata", {}))
            quality_scores.append(chunk.get("quality_score", 0.5))
            created_ats.append(current_time)
        
        return [
            vectors,
            rag_ids,
            doc_ids,
            chunk_ids,
            uris,
            excerpts,
            metadata_list,
            quality_scores,
            created_ats
        ]
    
    def _format_results(self, 
                       results, 
                       include_full_content: bool = False) -> List[Dict[str, Any]]:
        """Format search results"""
        formatted = []
        
        for hits in results:
            for hit in hits:
                entity = hit.entity
                item = {
                    "uri": entity.get("uri"),
                    "excerpt": entity.get("excerpt"),
                    "metadata": entity.get("metadata", {}),
                    "quality_score": entity.get("quality_score", 0.0),
                    "similarity_score": hit.score,
                    "doc_id": entity.get("doc_id"),
                    "chunk_id": entity.get("chunk_id"),
                    "created_at": entity.get("created_at")
                }
                
                # Include full content if requested
                if include_full_content:
                    item["content"] = self._retrieve_full_content(item["uri"])
                    
                formatted.append(item)
        
        return formatted
    
    def _retrieve_full_content(self, uri: str) -> str:
        """Retrieve full content from URI (placeholder for MinIO integration)"""
        # TODO: Implement MinIO/S3 content retrieval
        # For now, return excerpt as fallback
        logger.warning(f"Full content retrieval not implemented for: {uri}")
        return "Full content retrieval pending MinIO integration"
    
    def get_stats(self) -> Dict[str, Any]:
        """Get collection statistics"""
        try:
            collection = Collection(self.collection_name)
            stats = utility.get_query_segment_info(self.collection_name)
            
            return {
                "collection_name": self.collection_name,
                "embed_dim": self.embed_dim,
                "partitions": [p.name for p in collection.partitions],
                "total_entities": collection.num_entities,
                "stats": stats
            }
        except Exception as e:
            logger.error(f"Failed to get stats: {e}")
            return {}
    
    def get_partition_stats(self, partition_name: str) -> Dict[str, Any]:
        """
        Get statistics for a specific partition
        
        Args:
            partition_name: Name of the partition
            
        Returns:
            Partition statistics including entity count
        """
        try:
            collection = Collection(self.collection_name)
            
            # Load partition for querying
            collection.load(partition_names=[partition_name])
            
            # Get partition statistics
            stats = collection.get_partition_stats(partition_name)
            
            return {
                "partition_name": partition_name,
                "entity_count": stats.row_count if hasattr(stats, 'row_count') else 0,
                "status": "loaded" if collection.has_partition(partition_name) else "not_found"
            }
            
        except Exception as e:
            logger.debug(f"Failed to get partition stats for {partition_name}: {e}")
            return {
                "partition_name": partition_name,
                "entity_count": 0,
                "status": "error",
                "error": str(e)
            }