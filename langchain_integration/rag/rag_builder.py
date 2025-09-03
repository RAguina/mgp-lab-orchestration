"""
RAG Builder - Complete RAG Creation Workflow
Orchestrates document processing, embedding generation, and vector indexing
"""

from typing import List, Dict, Any, Optional, Union
import logging
import uuid
import asyncio
import time
from dataclasses import dataclass, asdict
from pathlib import Path

from .processing.document_pipeline import DocumentProcessingPipeline, PipelineConfig, ProcessingResult
from .embeddings.embedding_manager import get_embedding_manager
from .storage.milvus_store import MilvusRAGStore

logger = logging.getLogger("rag.builder")


@dataclass
class RAGBuildConfig:
    """Complete RAG build configuration"""
    # Processing config
    chunk_size: int = 800
    chunk_overlap: int = 100
    min_chunk_size: int = 120
    min_quality_score: float = 0.3
    enable_deduplication: bool = True
    
    # Embedding config
    embedding_model: str = "bge-m3"
    embedding_device: str = "cuda"
    batch_size: int = 32
    
    # Vector store config
    milvus_host: str = "localhost"
    milvus_port: int = 19530
    collection_name: str = "ai_lab_chunks"
    
    # Build limits
    max_documents: int = 1000
    max_chunks_per_doc: int = 500
    max_total_chunks: int = 50000


@dataclass
class RAGBuildResult:
    """Result of RAG building process"""
    rag_id: str
    success: bool
    
    # Processing results
    documents_processed: int
    chunks_created: int
    chunks_indexed: int
    
    # Timing
    processing_time: float
    embedding_time: float
    indexing_time: float
    total_time: float
    
    # Quality metrics
    avg_chunk_quality: float
    avg_embedding_dim: int
    
    # Storage info
    milvus_collection: str
    milvus_partition: str
    index_size_mb: float
    
    # Errors and warnings
    errors: List[str]
    warnings: List[str]
    
    # Detailed stats
    processing_stats: Dict[str, Any]
    embedding_stats: Dict[str, Any]
    indexing_stats: Dict[str, Any]


class RAGBuilder:
    """
    Complete RAG creation workflow orchestrator
    
    Workflow:
    1. Document Processing (parse, chunk, deduplicate)
    2. Embedding Generation (BGE-M3)
    3. Vector Indexing (Milvus with partitions)
    4. Quality validation and statistics
    """
    
    def __init__(self, config: Optional[RAGBuildConfig] = None):
        self.config = config or RAGBuildConfig()
        
        # Initialize components
        self._init_components()
        
        logger.info("RAG Builder initialized")
        logger.info(f"Config: {asdict(self.config)}")
    
    def _init_components(self):
        """Initialize processing pipeline, embeddings, and vector store"""
        
        # Document processing pipeline
        pipeline_config = PipelineConfig(
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap,
            min_chunk_size=self.config.min_chunk_size,
            min_quality_score=self.config.min_quality_score,
            enable_deduplication=self.config.enable_deduplication,
            max_chunks_per_doc=self.config.max_chunks_per_doc,
            max_total_chunks=self.config.max_total_chunks
        )
        
        self.pipeline = DocumentProcessingPipeline(pipeline_config)
        logger.info("Document processing pipeline initialized")
        
        # Embedding manager
        try:
            self.embedding_manager = get_embedding_manager()
            self.embedder = self.embedding_manager.get_provider(
                self.config.embedding_model, 
                device=self.config.embedding_device
            )
            logger.info(f"Embedding provider initialized: {self.config.embedding_model}")
        except Exception as e:
            logger.error(f"Failed to initialize embedding provider: {e}")
            raise
        
        # Vector store
        try:
            embed_dim = self.embedder.get_embedding_dimension()
            self.vector_store = MilvusRAGStore(
                host=self.config.milvus_host,
                port=self.config.milvus_port,
                embed_dim=embed_dim,
                collection_name=self.config.collection_name
            )
            logger.info(f"Milvus vector store initialized: {embed_dim}D embeddings")
        except Exception as e:
            logger.error(f"Failed to initialize vector store: {e}")
            raise
    
    def build_from_files(self, 
                        file_paths: List[str],
                        rag_name: str,
                        rag_description: Optional[str] = None) -> RAGBuildResult:
        """
        Build RAG from file paths
        
        Args:
            file_paths: List of document file paths
            rag_name: Human-readable name for the RAG
            rag_description: Optional description
            
        Returns:
            RAGBuildResult with comprehensive statistics
        """
        rag_id = f"rag_{uuid.uuid4().hex[:8]}"
        start_time = time.time()
        
        logger.info(f"[{rag_id}] Starting RAG build from {len(file_paths)} files")
        logger.info(f"[{rag_id}] RAG name: {rag_name}")
        
        try:
            # Step 1: Document Processing
            logger.info(f"[{rag_id}] Step 1: Document processing...")
            processing_start = time.time()
            
            processing_result = self.pipeline.process_files(file_paths)
            processing_time = time.time() - processing_start
            
            if not processing_result.success:
                return self._create_failure_result(
                    rag_id, start_time, 
                    errors=processing_result.errors + ["Document processing failed"]
                )
            
            logger.info(f"[{rag_id}] Processed {len(processing_result.chunks)} chunks in {processing_time:.2f}s")
            
            # Step 2: Embedding Generation
            logger.info(f"[{rag_id}] Step 2: Generating embeddings...")
            embedding_start = time.time()
            
            embeddings = self._generate_embeddings(processing_result.chunks, rag_id)
            embedding_time = time.time() - embedding_start
            
            logger.info(f"[{rag_id}] Generated {len(embeddings)} embeddings in {embedding_time:.2f}s")
            
            # Step 3: Vector Indexing
            logger.info(f"[{rag_id}] Step 3: Vector indexing...")
            indexing_start = time.time()
            
            indexing_result = self._index_chunks(rag_id, processing_result.chunks, embeddings)
            indexing_time = time.time() - indexing_start
            
            logger.info(f"[{rag_id}] Indexed {indexing_result['chunks_indexed']} chunks in {indexing_time:.2f}s")
            
            # Step 4: Build result
            total_time = time.time() - start_time
            
            result = RAGBuildResult(
                rag_id=rag_id,
                success=True,
                documents_processed=len(processing_result.parsed_documents),
                chunks_created=len(processing_result.chunks),
                chunks_indexed=indexing_result['chunks_indexed'],
                processing_time=processing_time,
                embedding_time=embedding_time,
                indexing_time=indexing_time,
                total_time=total_time,
                avg_chunk_quality=self._calculate_avg_quality(processing_result.chunks),
                avg_embedding_dim=self.embedder.get_embedding_dimension(),
                milvus_collection=self.config.collection_name,
                milvus_partition=f"rag_{rag_id}",
                index_size_mb=indexing_result.get('index_size_mb', 0.0),
                errors=[],
                warnings=[],
                processing_stats=processing_result.processing_stats,
                embedding_stats=self._get_embedding_stats(embeddings),
                indexing_stats=indexing_result
            )
            
            logger.info(f"[{rag_id}] RAG build completed successfully in {total_time:.2f}s")
            return result
            
        except Exception as e:
            error_msg = f"RAG build failed: {str(e)}"
            logger.error(f"[{rag_id}] {error_msg}")
            return self._create_failure_result(rag_id, start_time, errors=[error_msg])
    
    def build_from_uploaded_content(self,
                                  file_contents: List[Dict[str, Any]],
                                  rag_name: str,
                                  rag_description: Optional[str] = None) -> RAGBuildResult:
        """
        Build RAG from uploaded file contents
        
        Args:
            file_contents: List of file content dicts
            rag_name: Human-readable name for the RAG
            rag_description: Optional description
            
        Returns:
            RAGBuildResult with comprehensive statistics
        """
        rag_id = f"rag_{uuid.uuid4().hex[:8]}"
        start_time = time.time()
        
        logger.info(f"[{rag_id}] Starting RAG build from {len(file_contents)} uploaded files")
        
        try:
            # Step 1: Document Processing
            processing_result = self.pipeline.process_uploaded_content(file_contents)
            
            if not processing_result.success:
                return self._create_failure_result(
                    rag_id, start_time, 
                    errors=processing_result.errors + ["Upload processing failed"]
                )
            
            # Continue with same workflow as file-based build
            return self._complete_build(rag_id, rag_name, processing_result, start_time)
            
        except Exception as e:
            error_msg = f"Upload RAG build failed: {str(e)}"
            logger.error(f"[{rag_id}] {error_msg}")
            return self._create_failure_result(rag_id, start_time, errors=[error_msg])
    
    def _generate_embeddings(self, chunks, rag_id: str) -> List[List[float]]:
        """Generate embeddings for chunks with batching"""
        
        if not chunks:
            return []
        
        # Extract content from chunks
        chunk_contents = [chunk.content for chunk in chunks]
        
        # Generate embeddings in batches
        all_embeddings = []
        batch_size = self.config.batch_size
        
        for i in range(0, len(chunk_contents), batch_size):
            batch = chunk_contents[i:i + batch_size]
            logger.debug(f"[{rag_id}] Embedding batch {i//batch_size + 1}/{(len(chunk_contents) + batch_size - 1)//batch_size}")
            
            try:
                batch_embeddings = self.embedder.embed_documents(batch)
                all_embeddings.extend(batch_embeddings)
            except Exception as e:
                logger.error(f"[{rag_id}] Embedding batch failed: {e}")
                raise
        
        return all_embeddings
    
    def _index_chunks(self, rag_id: str, chunks, embeddings) -> Dict[str, Any]:
        """Index chunks with embeddings in Milvus"""
        
        if len(chunks) != len(embeddings):
            raise ValueError(f"Chunks ({len(chunks)}) and embeddings ({len(embeddings)}) count mismatch")
        
        try:
            # Prepare chunk data for Milvus
            chunk_data = []
            for chunk, embedding in zip(chunks, embeddings):
                chunk_dict = {
                    "content": chunk.content,
                    "doc_id": chunk.doc_id,
                    "chunk_id": chunk.chunk_id,
                    "uri": f"milvus://{self.config.collection_name}/rag_{rag_id}/{chunk.chunk_id}",
                    "metadata": {
                        **chunk.metadata,
                        "section_type": chunk.section_type,
                        "section_title": chunk.section_title,
                        "token_count": chunk.metrics.token_count,
                        "quality_score": chunk.metrics.quality_score
                    },
                    "quality_score": chunk.metrics.quality_score
                }
                chunk_data.append(chunk_dict)
            
            # Store in Milvus
            self.vector_store.store_chunks(rag_id, chunk_data, embeddings)
            
            # Calculate index stats
            total_vectors = len(embeddings)
            vector_dim = len(embeddings[0]) if embeddings else 0
            estimated_size_mb = (total_vectors * vector_dim * 4) / (1024 * 1024)  # 4 bytes per float
            
            return {
                "chunks_indexed": len(chunks),
                "total_vectors": total_vectors,
                "vector_dimension": vector_dim,
                "index_size_mb": estimated_size_mb,
                "partition_name": f"rag_{rag_id}"
            }
            
        except Exception as e:
            logger.error(f"[{rag_id}] Indexing failed: {e}")
            raise
    
    def _complete_build(self, rag_id: str, rag_name: str, processing_result, start_time: float) -> RAGBuildResult:
        """Complete the RAG build workflow"""
        
        # Generate embeddings
        embedding_start = time.time()
        embeddings = self._generate_embeddings(processing_result.chunks, rag_id)
        embedding_time = time.time() - embedding_start
        
        # Index chunks
        indexing_start = time.time()
        indexing_result = self._index_chunks(rag_id, processing_result.chunks, embeddings)
        indexing_time = time.time() - indexing_start
        
        total_time = time.time() - start_time
        
        return RAGBuildResult(
            rag_id=rag_id,
            success=True,
            documents_processed=len(processing_result.parsed_documents),
            chunks_created=len(processing_result.chunks),
            chunks_indexed=indexing_result['chunks_indexed'],
            processing_time=embedding_start - start_time,  # Processing time
            embedding_time=embedding_time,
            indexing_time=indexing_time,
            total_time=total_time,
            avg_chunk_quality=self._calculate_avg_quality(processing_result.chunks),
            avg_embedding_dim=self.embedder.get_embedding_dimension(),
            milvus_collection=self.config.collection_name,
            milvus_partition=f"rag_{rag_id}",
            index_size_mb=indexing_result.get('index_size_mb', 0.0),
            errors=[],
            warnings=[],
            processing_stats=processing_result.processing_stats,
            embedding_stats=self._get_embedding_stats(embeddings),
            indexing_stats=indexing_result
        )
    
    def _calculate_avg_quality(self, chunks) -> float:
        """Calculate average chunk quality score"""
        if not chunks:
            return 0.0
        
        total_quality = sum(chunk.metrics.quality_score for chunk in chunks)
        return total_quality / len(chunks)
    
    def _get_embedding_stats(self, embeddings) -> Dict[str, Any]:
        """Calculate embedding statistics"""
        if not embeddings:
            return {}
        
        total_embeddings = len(embeddings)
        embedding_dim = len(embeddings[0]) if embeddings else 0
        
        # Calculate some basic stats
        return {
            "total_embeddings": total_embeddings,
            "embedding_dimension": embedding_dim,
            "model_used": self.config.embedding_model,
            "device_used": self.config.embedding_device,
            "batch_size": self.config.batch_size,
            "estimated_memory_mb": (total_embeddings * embedding_dim * 4) / (1024 * 1024)
        }
    
    def _create_failure_result(self, rag_id: str, start_time: float, errors: List[str]) -> RAGBuildResult:
        """Create a failure result"""
        total_time = time.time() - start_time
        
        return RAGBuildResult(
            rag_id=rag_id,
            success=False,
            documents_processed=0,
            chunks_created=0,
            chunks_indexed=0,
            processing_time=0.0,
            embedding_time=0.0,
            indexing_time=0.0,
            total_time=total_time,
            avg_chunk_quality=0.0,
            avg_embedding_dim=0,
            milvus_collection=self.config.collection_name,
            milvus_partition="",
            index_size_mb=0.0,
            errors=errors,
            warnings=[],
            processing_stats={},
            embedding_stats={},
            indexing_stats={}
        )
    
    def get_build_status(self, rag_id: str) -> Dict[str, Any]:
        """Get status of a built RAG"""
        try:
            stats = self.vector_store.get_stats()
            
            # Check if RAG partition exists
            partition_name = f"rag_{rag_id}"
            partition_exists = any(p == partition_name for p in stats.get('partitions', []))
            
            return {
                "rag_id": rag_id,
                "exists": partition_exists,
                "partition_name": partition_name,
                "collection_stats": stats
            }
        except Exception as e:
            return {
                "rag_id": rag_id,
                "exists": False,
                "error": str(e)
            }
    
    def delete_rag(self, rag_id: str) -> Dict[str, Any]:
        """Delete a RAG and all its data"""
        try:
            self.vector_store.delete_rag(rag_id)
            return {
                "rag_id": rag_id,
                "deleted": True,
                "timestamp": time.time()
            }
        except Exception as e:
            return {
                "rag_id": rag_id,
                "deleted": False,
                "error": str(e)
            }