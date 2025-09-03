"""
Semantic Deduplicator Implementation
Advanced deduplication using embeddings and similarity metrics
"""

from typing import List, Dict, Any, Set, Tuple
import logging
from dataclasses import dataclass
import hashlib

from .smart_chunker import DocumentChunk

logger = logging.getLogger("rag.processing.deduplicator")


@dataclass
class DuplicateGroup:
    """Group of similar chunks"""
    representative: DocumentChunk  # Best chunk in the group
    duplicates: List[DocumentChunk]  # Similar chunks
    similarity_scores: List[float]  # Similarity to representative
    merge_strategy: str  # How duplicates were handled


class SemanticDeduplicator:
    """
    Advanced deduplication using multiple similarity metrics
    - Exact text matching
    - Fuzzy string similarity  
    - Semantic embedding similarity (when available)
    - Content overlap detection
    """
    
    def __init__(self,
                 exact_threshold: float = 0.95,
                 fuzzy_threshold: float = 0.85,
                 semantic_threshold: float = 0.90,
                 min_overlap_ratio: float = 0.70):
        
        self.exact_threshold = exact_threshold
        self.fuzzy_threshold = fuzzy_threshold
        self.semantic_threshold = semantic_threshold
        self.min_overlap_ratio = min_overlap_ratio
        
        # Import optional dependencies
        self.embedding_available = False
        self.fuzzy_available = False
        
        try:
            from difflib import SequenceMatcher
            self.sequence_matcher = SequenceMatcher
            self.fuzzy_available = True
        except ImportError:
            logger.warning("difflib not available - fuzzy matching disabled")
        
        # Initialize embedding provider if available
        self.embedder = None
        self._try_init_embedder()
    
    def _try_init_embedder(self):
        """Try to initialize embedding provider for semantic similarity"""
        try:
            from ..embeddings.embedding_manager import get_embedding_manager
            manager = get_embedding_manager()
            self.embedder = manager.get_provider("bge-m3", device="cpu")
            self.embedding_available = True
            logger.info("Semantic deduplication enabled with BGE-M3")
        except Exception as e:
            logger.info(f"Semantic deduplication disabled: {e}")
            self.embedding_available = False
    
    def deduplicate_chunks(self, chunks: List[DocumentChunk]) -> Tuple[List[DocumentChunk], List[DuplicateGroup]]:
        """
        Deduplicate chunks using multiple similarity methods
        
        Returns:
            (unique_chunks, duplicate_groups)
        """
        if not chunks:
            return [], []
        
        logger.info(f"Starting deduplication of {len(chunks)} chunks")
        
        # Step 1: Exact deduplication (fastest)
        chunks_after_exact, exact_groups = self._exact_deduplication(chunks)
        logger.info(f"After exact dedup: {len(chunks_after_exact)} chunks")
        
        # Step 2: Fuzzy deduplication
        if self.fuzzy_available:
            chunks_after_fuzzy, fuzzy_groups = self._fuzzy_deduplication(chunks_after_exact)
            logger.info(f"After fuzzy dedup: {len(chunks_after_fuzzy)} chunks")
        else:
            chunks_after_fuzzy, fuzzy_groups = chunks_after_exact, []
        
        # Step 3: Semantic deduplication
        if self.embedding_available:
            final_chunks, semantic_groups = self._semantic_deduplication(chunks_after_fuzzy)
            logger.info(f"After semantic dedup: {len(final_chunks)} chunks")
        else:
            final_chunks, semantic_groups = chunks_after_fuzzy, []
        
        # Combine all duplicate groups
        all_duplicate_groups = exact_groups + fuzzy_groups + semantic_groups
        
        logger.info(f"Deduplication complete: {len(chunks)} → {len(final_chunks)} chunks")
        logger.info(f"Found {len(all_duplicate_groups)} duplicate groups")
        
        return final_chunks, all_duplicate_groups
    
    def _exact_deduplication(self, chunks: List[DocumentChunk]) -> Tuple[List[DocumentChunk], List[DuplicateGroup]]:
        """Remove chunks with identical content"""
        content_to_chunk: Dict[str, List[DocumentChunk]] = {}
        
        # Group by normalized content
        for chunk in chunks:
            normalized = self._normalize_content(chunk.content)
            content_hash = hashlib.md5(normalized.encode()).hexdigest()
            
            if content_hash not in content_to_chunk:
                content_to_chunk[content_hash] = []
            content_to_chunk[content_hash].append(chunk)
        
        # Select best representative from each group
        unique_chunks = []
        duplicate_groups = []
        
        for content_hash, chunk_group in content_to_chunk.items():
            if len(chunk_group) == 1:
                unique_chunks.append(chunk_group[0])
            else:
                # Select best chunk as representative
                representative = self._select_best_chunk(chunk_group)
                unique_chunks.append(representative)
                
                # Create duplicate group
                duplicates = [c for c in chunk_group if c != representative]
                duplicate_groups.append(DuplicateGroup(
                    representative=representative,
                    duplicates=duplicates,
                    similarity_scores=[1.0] * len(duplicates),  # Exact matches
                    merge_strategy="exact_match"
                ))
        
        return unique_chunks, duplicate_groups
    
    def _fuzzy_deduplication(self, chunks: List[DocumentChunk]) -> Tuple[List[DocumentChunk], List[DuplicateGroup]]:
        """Remove chunks with high fuzzy similarity"""
        if not self.fuzzy_available:
            return chunks, []
        
        remaining_chunks = chunks.copy()
        duplicate_groups = []
        
        i = 0
        while i < len(remaining_chunks):
            current_chunk = remaining_chunks[i]
            similar_chunks = []
            similarity_scores = []
            
            # Compare with remaining chunks
            j = i + 1
            while j < len(remaining_chunks):
                other_chunk = remaining_chunks[j]
                similarity = self._calculate_fuzzy_similarity(
                    current_chunk.content, 
                    other_chunk.content
                )
                
                if similarity >= self.fuzzy_threshold:
                    similar_chunks.append(other_chunk)
                    similarity_scores.append(similarity)
                    # Remove similar chunk
                    remaining_chunks.pop(j)
                else:
                    j += 1
            
            # Create duplicate group if similar chunks found
            if similar_chunks:
                # Select best representative from current + similar
                all_candidates = [current_chunk] + similar_chunks
                representative = self._select_best_chunk(all_candidates)
                
                # Update remaining chunks with representative
                remaining_chunks[i] = representative
                
                # Create duplicate group
                duplicates = [c for c in all_candidates if c != representative]
                if duplicates:
                    duplicate_groups.append(DuplicateGroup(
                        representative=representative,
                        duplicates=duplicates,
                        similarity_scores=similarity_scores,
                        merge_strategy="fuzzy_similarity"
                    ))
            
            i += 1
        
        return remaining_chunks, duplicate_groups
    
    def _semantic_deduplication(self, chunks: List[DocumentChunk]) -> Tuple[List[DocumentChunk], List[DuplicateGroup]]:
        """Remove chunks with high semantic similarity"""
        if not self.embedding_available or len(chunks) < 2:
            return chunks, []
        
        try:
            # Generate embeddings for all chunks
            logger.info("Generating embeddings for semantic deduplication...")
            chunk_texts = [chunk.content for chunk in chunks]
            embeddings = self.embedder.embed_documents(chunk_texts)
            
            # Find similar pairs using cosine similarity
            similar_pairs = []
            for i in range(len(chunks)):
                for j in range(i + 1, len(chunks)):
                    similarity = self._cosine_similarity(embeddings[i], embeddings[j])
                    if similarity >= self.semantic_threshold:
                        similar_pairs.append((i, j, similarity))
            
            # Group similar chunks
            chunk_groups = self._group_similar_chunks(chunks, similar_pairs)
            
            # Select representatives
            unique_chunks = []
            duplicate_groups = []
            
            for chunk_group in chunk_groups:
                if len(chunk_group) == 1:
                    unique_chunks.append(chunk_group[0])
                else:
                    representative = self._select_best_chunk(chunk_group)
                    unique_chunks.append(representative)
                    
                    duplicates = [c for c in chunk_group if c != representative]
                    if duplicates:
                        # Calculate similarities to representative
                        rep_embedding = self.embedder.embed_query(representative.content)
                        similarities = []
                        for dup in duplicates:
                            dup_embedding = self.embedder.embed_query(dup.content)
                            sim = self._cosine_similarity(rep_embedding, dup_embedding)
                            similarities.append(sim)
                        
                        duplicate_groups.append(DuplicateGroup(
                            representative=representative,
                            duplicates=duplicates,
                            similarity_scores=similarities,
                            merge_strategy="semantic_similarity"
                        ))
            
            return unique_chunks, duplicate_groups
            
        except Exception as e:
            logger.warning(f"Semantic deduplication failed: {e}")
            return chunks, []
    
    def _normalize_content(self, content: str) -> str:
        """Normalize content for exact matching"""
        # Remove extra whitespace, normalize line endings
        normalized = " ".join(content.split())
        normalized = normalized.lower().strip()
        return normalized
    
    def _calculate_fuzzy_similarity(self, text1: str, text2: str) -> float:
        """Calculate fuzzy string similarity using SequenceMatcher"""
        if not self.fuzzy_available:
            return 0.0
        
        # Normalize texts
        norm1 = self._normalize_content(text1)
        norm2 = self._normalize_content(text2)
        
        if not norm1 or not norm2:
            return 0.0
        
        # Calculate similarity ratio
        matcher = self.sequence_matcher(None, norm1, norm2)
        return matcher.ratio()
    
    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors"""
        import math
        
        if len(vec1) != len(vec2):
            return 0.0
        
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        magnitude1 = math.sqrt(sum(a * a for a in vec1))
        magnitude2 = math.sqrt(sum(a * a for a in vec2))
        
        if magnitude1 == 0 or magnitude2 == 0:
            return 0.0
        
        return dot_product / (magnitude1 * magnitude2)
    
    def _group_similar_chunks(self, chunks: List[DocumentChunk], similar_pairs: List[Tuple[int, int, float]]) -> List[List[DocumentChunk]]:
        """Group chunks based on similarity pairs using Union-Find"""
        
        # Initialize Union-Find structure
        parent = list(range(len(chunks)))
        
        def find(x):
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]
        
        def union(x, y):
            px, py = find(x), find(y)
            if px != py:
                parent[px] = py
        
        # Union similar chunks
        for i, j, _ in similar_pairs:
            union(i, j)
        
        # Group chunks by their root parent
        groups = {}
        for i, chunk in enumerate(chunks):
            root = find(i)
            if root not in groups:
                groups[root] = []
            groups[root].append(chunk)
        
        return list(groups.values())
    
    def _select_best_chunk(self, chunks: List[DocumentChunk]) -> DocumentChunk:
        """
        Select the best chunk as representative from a group
        Criteria: highest quality score, longest content, latest timestamp
        """
        if not chunks:
            raise ValueError("Cannot select best chunk from empty list")
        
        if len(chunks) == 1:
            return chunks[0]
        
        # Sort by multiple criteria
        def chunk_score(chunk: DocumentChunk) -> Tuple[float, int, str]:
            return (
                chunk.metrics.quality_score,  # Higher is better
                chunk.metrics.token_count,    # Longer is better
                chunk.chunk_id               # Lexicographic for stability
            )
        
        best_chunk = max(chunks, key=chunk_score)
        return best_chunk
    
    def _calculate_content_overlap(self, content1: str, content2: str) -> float:
        """Calculate content overlap ratio"""
        words1 = set(self._normalize_content(content1).split())
        words2 = set(self._normalize_content(content2).split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union) if union else 0.0
    
    def get_deduplication_stats(self, original_count: int, final_count: int, duplicate_groups: List[DuplicateGroup]) -> Dict[str, Any]:
        """Get deduplication statistics"""
        total_duplicates = sum(len(group.duplicates) for group in duplicate_groups)
        
        strategy_counts = {}
        for group in duplicate_groups:
            strategy = group.merge_strategy
            strategy_counts[strategy] = strategy_counts.get(strategy, 0) + len(group.duplicates)
        
        return {
            "original_chunks": original_count,
            "final_chunks": final_count,
            "duplicates_removed": total_duplicates,
            "deduplication_ratio": (original_count - final_count) / original_count if original_count > 0 else 0.0,
            "duplicate_groups": len(duplicate_groups),
            "strategies_used": strategy_counts,
            "methods_available": {
                "exact": True,
                "fuzzy": self.fuzzy_available,
                "semantic": self.embedding_available
            }
        }