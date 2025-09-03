"""
Document Processing Pipeline
Coordinates parsing, chunking, and deduplication
"""

from typing import List, Dict, Any, Optional, Union
import logging
from pathlib import Path
import uuid
from dataclasses import dataclass, asdict

from .document_parser import DocumentParser, ParsedDocument
from .smart_chunker import SmartChunker, DocumentChunk
from .deduplicator import SemanticDeduplicator, DuplicateGroup

logger = logging.getLogger("rag.processing.pipeline")


@dataclass
class ProcessingResult:
    """Result of document processing pipeline"""
    chunks: List[DocumentChunk]
    parsed_documents: List[ParsedDocument]
    duplicate_groups: List[DuplicateGroup]
    processing_stats: Dict[str, Any]
    pipeline_id: str
    success: bool
    errors: List[str]


@dataclass 
class PipelineConfig:
    """Configuration for document processing pipeline"""
    # Chunking config
    chunk_size: int = 800
    chunk_overlap: int = 100
    min_chunk_size: int = 120
    
    # Parsing config
    preserve_structure: bool = True
    
    # Deduplication config
    enable_deduplication: bool = True
    exact_threshold: float = 0.95
    fuzzy_threshold: float = 0.85
    semantic_threshold: float = 0.90
    
    # Quality filtering
    min_quality_score: float = 0.3
    
    # Processing limits
    max_chunks_per_doc: int = 1000
    max_total_chunks: int = 10000


class DocumentProcessingPipeline:
    """
    Complete document processing pipeline
    File Upload → Parse → Chunk → Deduplicate → Quality Filter
    """
    
    def __init__(self, config: Optional[PipelineConfig] = None):
        self.config = config or PipelineConfig()
        
        # Initialize processors
        self.parser = DocumentParser(preserve_structure=self.config.preserve_structure)
        
        self.chunker = SmartChunker(
            chunk_size=self.config.chunk_size,
            overlap=self.config.chunk_overlap,
            min_chunk_size=self.config.min_chunk_size
        )
        
        if self.config.enable_deduplication:
            self.deduplicator = SemanticDeduplicator(
                exact_threshold=self.config.exact_threshold,
                fuzzy_threshold=self.config.fuzzy_threshold,
                semantic_threshold=self.config.semantic_threshold
            )
        else:
            self.deduplicator = None
        
        logger.info("Document processing pipeline initialized")
        logger.info(f"Config: {asdict(self.config)}")
    
    def process_files(self, 
                     file_paths: List[str],
                     doc_ids: Optional[List[str]] = None) -> ProcessingResult:
        """
        Process multiple documents from file paths
        
        Args:
            file_paths: List of file paths to process
            doc_ids: Optional document IDs, defaults to filenames
            
        Returns:
            ProcessingResult with chunks and metadata
        """
        pipeline_id = f"pipeline_{uuid.uuid4().hex[:8]}"
        logger.info(f"[{pipeline_id}] Starting file processing: {len(file_paths)} files")
        
        # Validate inputs
        if not file_paths:
            return ProcessingResult(
                chunks=[], parsed_documents=[], duplicate_groups=[],
                processing_stats={}, pipeline_id=pipeline_id,
                success=False, errors=["No files provided"]
            )
        
        if doc_ids and len(doc_ids) != len(file_paths):
            return ProcessingResult(
                chunks=[], parsed_documents=[], duplicate_groups=[],
                processing_stats={}, pipeline_id=pipeline_id,
                success=False, errors=["doc_ids length mismatch with file_paths"]
            )
        
        # Process each file
        parsed_documents = []
        all_chunks = []
        errors = []
        
        for i, file_path in enumerate(file_paths):
            doc_id = doc_ids[i] if doc_ids else Path(file_path).stem
            
            try:
                # Parse document
                parsed_doc = self.parser.parse_file(file_path, doc_id)
                parsed_documents.append(parsed_doc)
                
                # Chunk document
                doc_chunks = self.chunker.chunk_document(
                    text=parsed_doc.content,
                    doc_id=doc_id,
                    doc_metadata=parsed_doc.metadata
                )
                
                # Apply per-document limits
                if len(doc_chunks) > self.config.max_chunks_per_doc:
                    logger.warning(f"Document {doc_id} has {len(doc_chunks)} chunks, limiting to {self.config.max_chunks_per_doc}")
                    doc_chunks = doc_chunks[:self.config.max_chunks_per_doc]
                
                all_chunks.extend(doc_chunks)
                logger.info(f"[{pipeline_id}] Processed {file_path}: {len(doc_chunks)} chunks")
                
            except Exception as e:
                error_msg = f"Failed to process {file_path}: {str(e)}"
                logger.error(f"[{pipeline_id}] {error_msg}")
                errors.append(error_msg)
                continue
        
        # Apply global chunk limit
        if len(all_chunks) > self.config.max_total_chunks:
            logger.warning(f"Total chunks {len(all_chunks)} exceeds limit, truncating to {self.config.max_total_chunks}")
            all_chunks = all_chunks[:self.config.max_total_chunks]
        
        # Process chunks
        return self._process_chunks(all_chunks, parsed_documents, pipeline_id, errors)
    
    def process_uploaded_content(self,
                               file_contents: List[Dict[str, Any]]) -> ProcessingResult:
        """
        Process documents from uploaded content (FastAPI UploadFile)
        
        Args:
            file_contents: List of dicts with keys: content, filename, file_type, doc_id?
            
        Returns:
            ProcessingResult with chunks and metadata
        """
        pipeline_id = f"upload_{uuid.uuid4().hex[:8]}"
        logger.info(f"[{pipeline_id}] Starting upload processing: {len(file_contents)} files")
        
        parsed_documents = []
        all_chunks = []
        errors = []
        
        for file_info in file_contents:
            try:
                content = file_info["content"]
                filename = file_info["filename"]
                file_type = file_info["file_type"]
                doc_id = file_info.get("doc_id", Path(filename).stem)
                
                # Parse content
                if file_type in {'.txt', '.md', '.markdown'}:
                    parsed_doc = self.parser.parse_content(
                        content=content if isinstance(content, str) else content.decode('utf-8'),
                        filename=filename,
                        file_type=file_type,
                        doc_id=doc_id
                    )
                else:
                    # For binary files, would need temporary file approach
                    logger.warning(f"Skipping binary file type {file_type} for upload processing")
                    continue
                
                parsed_documents.append(parsed_doc)
                
                # Chunk document
                doc_chunks = self.chunker.chunk_document(
                    text=parsed_doc.content,
                    doc_id=doc_id,
                    doc_metadata=parsed_doc.metadata
                )
                
                all_chunks.extend(doc_chunks)
                logger.info(f"[{pipeline_id}] Processed upload {filename}: {len(doc_chunks)} chunks")
                
            except Exception as e:
                error_msg = f"Failed to process upload {file_info.get('filename', 'unknown')}: {str(e)}"
                logger.error(f"[{pipeline_id}] {error_msg}")
                errors.append(error_msg)
                continue
        
        return self._process_chunks(all_chunks, parsed_documents, pipeline_id, errors)
    
    def _process_chunks(self,
                       chunks: List[DocumentChunk],
                       parsed_documents: List[ParsedDocument],
                       pipeline_id: str,
                       errors: List[str]) -> ProcessingResult:
        """Process chunks through deduplication and quality filtering"""
        
        original_chunk_count = len(chunks)
        duplicate_groups = []
        
        # Step 1: Quality filtering
        quality_filtered_chunks = self._apply_quality_filter(chunks)
        logger.info(f"[{pipeline_id}] Quality filter: {len(chunks)} → {len(quality_filtered_chunks)}")
        
        # Step 2: Deduplication
        if self.deduplicator and len(quality_filtered_chunks) > 1:
            try:
                deduplicated_chunks, duplicate_groups = self.deduplicator.deduplicate_chunks(
                    quality_filtered_chunks
                )
                final_chunks = deduplicated_chunks
                logger.info(f"[{pipeline_id}] Deduplication: {len(quality_filtered_chunks)} → {len(final_chunks)}")
            except Exception as e:
                logger.warning(f"[{pipeline_id}] Deduplication failed: {e}")
                final_chunks = quality_filtered_chunks
                errors.append(f"Deduplication failed: {str(e)}")
        else:
            final_chunks = quality_filtered_chunks
        
        # Calculate processing stats
        processing_stats = self._calculate_stats(
            original_chunks=chunks,
            final_chunks=final_chunks,
            parsed_documents=parsed_documents,
            duplicate_groups=duplicate_groups,
            pipeline_id=pipeline_id
        )
        
        success = len(final_chunks) > 0 and len(errors) == 0
        
        logger.info(f"[{pipeline_id}] Pipeline complete: {original_chunk_count} → {len(final_chunks)} chunks")
        
        return ProcessingResult(
            chunks=final_chunks,
            parsed_documents=parsed_documents,
            duplicate_groups=duplicate_groups,
            processing_stats=processing_stats,
            pipeline_id=pipeline_id,
            success=success,
            errors=errors
        )
    
    def _apply_quality_filter(self, chunks: List[DocumentChunk]) -> List[DocumentChunk]:
        """Filter chunks by quality score"""
        if self.config.min_quality_score <= 0:
            return chunks
        
        filtered = []
        for chunk in chunks:
            if chunk.metrics.quality_score >= self.config.min_quality_score:
                filtered.append(chunk)
            else:
                logger.debug(f"Filtered low quality chunk: {chunk.chunk_id} (score: {chunk.metrics.quality_score:.3f})")
        
        return filtered
    
    def _calculate_stats(self,
                        original_chunks: List[DocumentChunk],
                        final_chunks: List[DocumentChunk], 
                        parsed_documents: List[ParsedDocument],
                        duplicate_groups: List[DuplicateGroup],
                        pipeline_id: str) -> Dict[str, Any]:
        """Calculate comprehensive processing statistics"""
        
        # Basic counts
        total_docs = len(parsed_documents)
        total_original_chunks = len(original_chunks)
        total_final_chunks = len(final_chunks)
        
        # Document stats
        doc_word_counts = [doc.word_count or 0 for doc in parsed_documents]
        total_words = sum(doc_word_counts)
        avg_words_per_doc = total_words / max(total_docs, 1)
        
        # Chunk stats
        chunk_token_counts = [chunk.metrics.token_count for chunk in final_chunks]
        chunk_quality_scores = [chunk.metrics.quality_score for chunk in final_chunks]
        
        avg_tokens_per_chunk = sum(chunk_token_counts) / max(len(chunk_token_counts), 1)
        avg_quality_score = sum(chunk_quality_scores) / max(len(chunk_quality_scores), 1)
        
        # Section type distribution
        section_types = {}
        for chunk in final_chunks:
            section_type = chunk.section_type
            section_types[section_type] = section_types.get(section_type, 0) + 1
        
        # Deduplication stats
        dedup_stats = {}
        if duplicate_groups:
            total_duplicates = sum(len(group.duplicates) for group in duplicate_groups)
            dedup_stats = {
                "duplicate_groups": len(duplicate_groups),
                "duplicates_removed": total_duplicates,
                "deduplication_ratio": total_duplicates / max(total_original_chunks, 1)
            }
        
        return {
            "pipeline_id": pipeline_id,
            "documents": {
                "total_documents": total_docs,
                "total_words": total_words,
                "avg_words_per_document": avg_words_per_doc,
                "file_types": [doc.file_type for doc in parsed_documents]
            },
            "chunks": {
                "original_count": total_original_chunks,
                "final_count": total_final_chunks,
                "reduction_ratio": (total_original_chunks - total_final_chunks) / max(total_original_chunks, 1),
                "avg_tokens_per_chunk": avg_tokens_per_chunk,
                "avg_quality_score": avg_quality_score,
                "section_type_distribution": section_types,
                "token_counts": {
                    "min": min(chunk_token_counts) if chunk_token_counts else 0,
                    "max": max(chunk_token_counts) if chunk_token_counts else 0,
                    "avg": avg_tokens_per_chunk
                }
            },
            "deduplication": dedup_stats,
            "quality_filtering": {
                "min_quality_threshold": self.config.min_quality_score,
                "chunks_above_threshold": len([c for c in final_chunks 
                                             if c.metrics.quality_score >= self.config.min_quality_score])
            },
            "config": asdict(self.config)
        }
    
    def get_supported_formats(self) -> Dict[str, bool]:
        """Get supported file formats"""
        return self.parser.get_supported_types()
    
    def validate_config(self) -> List[str]:
        """Validate pipeline configuration"""
        issues = []
        
        if self.config.chunk_size <= 0:
            issues.append("chunk_size must be positive")
        
        if self.config.chunk_overlap >= self.config.chunk_size:
            issues.append("chunk_overlap must be less than chunk_size")
        
        if self.config.min_chunk_size >= self.config.chunk_size:
            issues.append("min_chunk_size must be less than chunk_size")
        
        if not 0 <= self.config.min_quality_score <= 1:
            issues.append("min_quality_score must be between 0 and 1")
        
        # Threshold validations
        for threshold_name in ["exact_threshold", "fuzzy_threshold", "semantic_threshold"]:
            threshold_val = getattr(self.config, threshold_name)
            if not 0 <= threshold_val <= 1:
                issues.append(f"{threshold_name} must be between 0 and 1")
        
        return issues