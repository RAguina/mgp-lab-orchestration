"""
Smart Chunking Implementation
Token-aware chunking aligned to BGE-M3 embedding model
"""

from typing import List, Dict, Any, Optional, Tuple
import logging
import re
from dataclasses import dataclass

logger = logging.getLogger("rag.processing.smart_chunker")

# Remove redundant tokenizer import - we'll reuse from embedding manager
HF_TOKENIZER_AVAILABLE = True


@dataclass
class ChunkMetrics:
    """Quality metrics for a chunk"""
    token_count: int
    char_count: int
    sentence_count: int
    avg_sentence_length: float
    information_density: float
    coherence_score: float
    quality_score: float


@dataclass
class DocumentChunk:
    """Structured chunk with metadata"""
    content: str
    chunk_id: str
    doc_id: str
    start_char: int
    end_char: int
    section_type: str  # paragraph, header, list, code, table
    section_title: Optional[str]
    metrics: ChunkMetrics
    metadata: Dict[str, Any]


class SmartChunker:
    """
    GPT-5 style chunking with BGE-M3 tokenizer alignment
    - Token-aware chunking (not character-based)
    - Preserves paragraph boundaries  
    - Maintains heading context
    - Quality scoring for each chunk
    """
    
    def __init__(self, 
                 chunk_size: int = 800,
                 overlap: int = 100, 
                 min_chunk_size: int = 120,
                 model_name: str = "BAAI/bge-m3"):
        
        self.chunk_size = chunk_size  # in tokens
        self.overlap = overlap
        self.min_chunk_size = min_chunk_size
        self.model_name = model_name
        
        # Initialize tokenizer
        self.tokenizer = None
        self.token_aware = False
        self._init_tokenizer()
        
        # Quality thresholds
        self.min_quality_score = 0.3
        self.ideal_sentence_length = 20  # tokens
        
    def _init_tokenizer(self):
        """Initialize BGE-M3 tokenizer by reusing from embedding manager"""
        try:
            # Reuse tokenizer from embedding manager to avoid redundancy
            from ..embeddings.embedding_manager import get_embedding_manager
            embedder = get_embedding_manager().get_provider("bge-m3", device="cpu")
            
            # Access internal tokenizer from SentenceTransformer
            if hasattr(embedder.model, 'tokenizer'):
                self.tokenizer = embedder.model.tokenizer
                self.token_aware = True
                logger.info(f"Reusing BGE-M3 tokenizer from embedding manager")
            else:
                # Fallback to word estimation
                self.token_aware = False
                logger.warning("Could not access tokenizer from embedding model, using word estimation")
        except Exception as e:
            logger.warning(f"Failed to get tokenizer from embedding manager: {e}")
            self.token_aware = False
    
    def count_tokens(self, text: str) -> int:
        """Count tokens using BGE-M3 tokenizer for accurate chunking"""
        if not text.strip():
            return 0
            
        if self.token_aware and self.tokenizer:
            try:
                tokens = self.tokenizer.encode(text, add_special_tokens=False)
                return len(tokens)
            except Exception as e:
                logger.warning(f"Tokenizer failed, falling back to estimation: {e}")
        
        # Fallback: rough word-to-token estimation
        word_count = len(text.split())
        return int(word_count * 1.3)  # Approximate token ratio
    
    def chunk_document(self, 
                      text: str, 
                      doc_id: str,
                      doc_metadata: Optional[Dict[str, Any]] = None) -> List[DocumentChunk]:
        """
        Main chunking method with quality scoring
        """
        if not text.strip():
            return []
            
        doc_metadata = doc_metadata or {}
        
        # Step 1: Detect document structure
        sections = self._detect_sections(text)
        logger.info(f"Detected {len(sections)} sections in document {doc_id}")
        
        # Step 2: Chunk within sections
        chunks = []
        for section in sections:
            section_chunks = self._chunk_section(section, doc_id)
            chunks.extend(section_chunks)
        
        # Step 3: Quality scoring and filtering
        scored_chunks = self._score_chunks(chunks)
        
        # Step 4: Deduplication (basic)
        unique_chunks = self._basic_deduplication(scored_chunks)
        
        logger.info(f"Generated {len(unique_chunks)} chunks from {doc_id}")
        return unique_chunks
    
    def _detect_sections(self, text: str) -> List[Dict[str, Any]]:
        """
        Detect document structure: headers, paragraphs, lists, code blocks
        """
        sections = []
        lines = text.split('\n')
        current_section = {"type": "paragraph", "content": "", "title": None, "start_line": 0}
        
        for i, line in enumerate(lines):
            line_stripped = line.strip()
            
            # Empty line - potential section boundary
            if not line_stripped:
                if current_section["content"].strip():
                    current_section["end_line"] = i
                    sections.append(current_section.copy())
                    current_section = {"type": "paragraph", "content": "", "title": None, "start_line": i+1}
                continue
            
            # Header detection (Markdown style)
            if line_stripped.startswith('#'):
                # Save previous section
                if current_section["content"].strip():
                    current_section["end_line"] = i-1
                    sections.append(current_section.copy())
                
                # Start header section
                header_level = len(line_stripped) - len(line_stripped.lstrip('#'))
                current_section = {
                    "type": "header",
                    "content": line_stripped,
                    "title": line_stripped.lstrip('#').strip(),
                    "level": header_level,
                    "start_line": i
                }
                continue
            
            # List detection
            if re.match(r'^[\s]*[-*+]\s+|^\s*\d+\.\s+', line_stripped):
                if current_section["type"] != "list":
                    # Save previous section
                    if current_section["content"].strip():
                        current_section["end_line"] = i-1
                        sections.append(current_section.copy())
                    
                    # Start list section
                    current_section = {"type": "list", "content": "", "title": None, "start_line": i}
            
            # Code block detection
            elif line_stripped.startswith('```') or line_stripped.startswith('    '):
                if current_section["type"] != "code":
                    # Save previous section
                    if current_section["content"].strip():
                        current_section["end_line"] = i-1
                        sections.append(current_section.copy())
                    
                    # Start code section
                    current_section = {"type": "code", "content": "", "title": None, "start_line": i}
            
            # Add line to current section
            current_section["content"] += line + '\n'
        
        # Add final section
        if current_section["content"].strip():
            current_section["end_line"] = len(lines)
            sections.append(current_section)
        
        return sections
    
    def _chunk_section(self, section: Dict[str, Any], doc_id: str) -> List[DocumentChunk]:
        """
        Chunk within a section, respecting structure
        """
        content = section["content"].strip()
        if not content:
            return []
        
        section_type = section.get("type", "paragraph")
        section_title = section.get("title")
        
        # For headers, keep them intact if small enough
        if section_type == "header":
            token_count = self.count_tokens(content)
            if token_count <= self.chunk_size:
                return [self._create_chunk(
                    content=content,
                    doc_id=doc_id,
                    chunk_id=f"{doc_id}_header_{section.get('start_line', 0)}",
                    section_type=section_type,
                    section_title=section_title,
                    start_char=0,
                    end_char=len(content)
                )]
        
        # For other sections, use sliding window chunking
        return self._sliding_window_chunk(
            content=content,
            doc_id=doc_id,
            section_type=section_type,
            section_title=section_title
        )
    
    def _sliding_window_chunk(self, 
                             content: str, 
                             doc_id: str,
                             section_type: str,
                             section_title: Optional[str]) -> List[DocumentChunk]:
        """
        Sliding window chunking with overlap
        """
        chunks = []
        
        # Split into sentences for better boundaries
        sentences = self._split_sentences(content)
        if not sentences:
            return []
        
        current_chunk_sentences = []
        current_token_count = 0
        chunk_counter = 0
        
        for i, sentence in enumerate(sentences):
            sentence_tokens = self.count_tokens(sentence)
            
            # If single sentence exceeds chunk size, split it
            if sentence_tokens > self.chunk_size:
                # Save current chunk first
                if current_chunk_sentences:
                    chunk = self._create_chunk_from_sentences(
                        sentences=current_chunk_sentences,
                        doc_id=doc_id,
                        chunk_id=f"{doc_id}_chunk_{chunk_counter}",
                        section_type=section_type,
                        section_title=section_title
                    )
                    chunks.append(chunk)
                    chunk_counter += 1
                    current_chunk_sentences = []
                    current_token_count = 0
                
                # Split long sentence by words
                word_chunks = self._split_long_sentence(sentence, doc_id, chunk_counter)
                chunks.extend(word_chunks)
                chunk_counter += len(word_chunks)
                continue
            
            # Check if adding sentence exceeds chunk size
            if current_token_count + sentence_tokens > self.chunk_size and current_chunk_sentences:
                # Create chunk
                chunk = self._create_chunk_from_sentences(
                    sentences=current_chunk_sentences,
                    doc_id=doc_id,
                    chunk_id=f"{doc_id}_chunk_{chunk_counter}",
                    section_type=section_type,
                    section_title=section_title
                )
                chunks.append(chunk)
                chunk_counter += 1
                
                # Start new chunk with overlap
                overlap_sentences = self._get_overlap_sentences(
                    current_chunk_sentences, 
                    self.overlap
                )
                current_chunk_sentences = overlap_sentences + [sentence]
                current_token_count = sum(self.count_tokens(s) for s in current_chunk_sentences)
            else:
                # Add sentence to current chunk
                current_chunk_sentences.append(sentence)
                current_token_count += sentence_tokens
        
        # Add final chunk
        if current_chunk_sentences:
            chunk = self._create_chunk_from_sentences(
                sentences=current_chunk_sentences,
                doc_id=doc_id,
                chunk_id=f"{doc_id}_chunk_{chunk_counter}",
                section_type=section_type,
                section_title=section_title
            )
            chunks.append(chunk)
        
        return chunks
    
    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences with better boundary detection"""
        # Simple sentence splitting (could be improved with spacy/nltk)
        sentence_endings = r'[.!?]+\s+'
        sentences = re.split(sentence_endings, text)
        
        # Filter out empty sentences and add back endings
        result = []
        for i, sentence in enumerate(sentences):
            sentence = sentence.strip()
            if sentence:
                # Add back sentence ending (except for last)
                if i < len(sentences) - 1:
                    sentence += '.'
                result.append(sentence)
        
        return result
    
    def _split_long_sentence(self, sentence: str, doc_id: str, chunk_counter: int) -> List[DocumentChunk]:
        """Split very long sentence by words when it exceeds chunk_size"""
        words = sentence.split()
        chunks = []
        current_words = []
        current_token_count = 0
        
        for word in words:
            word_tokens = self.count_tokens(word + " ")
            
            if current_token_count + word_tokens > self.chunk_size and current_words:
                # Create chunk from current words
                chunk_content = " ".join(current_words)
                chunk = self._create_chunk(
                    content=chunk_content,
                    doc_id=doc_id,
                    chunk_id=f"{doc_id}_longchunk_{chunk_counter}",
                    section_type="paragraph",
                    section_title=None,
                    start_char=0,
                    end_char=len(chunk_content)
                )
                chunks.append(chunk)
                
                # Start new chunk with some overlap
                overlap_size = min(10, len(current_words) // 2)
                current_words = current_words[-overlap_size:] + [word]
                current_token_count = sum(self.count_tokens(w + " ") for w in current_words)
            else:
                current_words.append(word)
                current_token_count += word_tokens
        
        # Add final chunk
        if current_words:
            chunk_content = " ".join(current_words)
            chunk = self._create_chunk(
                content=chunk_content,
                doc_id=doc_id,
                chunk_id=f"{doc_id}_longchunk_{chunk_counter + len(chunks)}",
                section_type="paragraph", 
                section_title=None,
                start_char=0,
                end_char=len(chunk_content)
            )
            chunks.append(chunk)
        
        return chunks
    
    def _get_overlap_sentences(self, sentences: List[str], overlap_tokens: int) -> List[str]:
        """Get sentences for overlap based on token count"""
        if not sentences or overlap_tokens <= 0:
            return []
        
        overlap_sentences = []
        token_count = 0
        
        # Start from the end and work backwards
        for sentence in reversed(sentences):
            sentence_tokens = self.count_tokens(sentence)
            if token_count + sentence_tokens <= overlap_tokens:
                overlap_sentences.insert(0, sentence)
                token_count += sentence_tokens
            else:
                break
        
        return overlap_sentences
    
    def _create_chunk_from_sentences(self, 
                                   sentences: List[str],
                                   doc_id: str,
                                   chunk_id: str,
                                   section_type: str,
                                   section_title: Optional[str]) -> DocumentChunk:
        """Create DocumentChunk from list of sentences"""
        content = " ".join(sentences)
        return self._create_chunk(
            content=content,
            doc_id=doc_id,
            chunk_id=chunk_id,
            section_type=section_type,
            section_title=section_title,
            start_char=0,
            end_char=len(content)
        )
    
    def _create_chunk(self, 
                     content: str,
                     doc_id: str, 
                     chunk_id: str,
                     section_type: str,
                     section_title: Optional[str],
                     start_char: int,
                     end_char: int) -> DocumentChunk:
        """Create DocumentChunk with metrics"""
        
        # Calculate metrics
        metrics = self._calculate_metrics(content)
        
        return DocumentChunk(
            content=content,
            chunk_id=chunk_id,
            doc_id=doc_id,
            start_char=start_char,
            end_char=end_char,
            section_type=section_type,
            section_title=section_title,
            metrics=metrics,
            metadata={
                "created_by": "smart_chunker",
                "chunk_method": "sliding_window",
                "tokenizer": self.model_name if self.token_aware else "word_estimation"
            }
        )
    
    def _calculate_metrics(self, content: str) -> ChunkMetrics:
        """Calculate quality metrics for chunk"""
        token_count = self.count_tokens(content)
        char_count = len(content)
        
        # Sentence analysis
        sentences = self._split_sentences(content)
        sentence_count = len(sentences)
        avg_sentence_length = token_count / max(sentence_count, 1)
        
        # Information density (very basic)
        unique_words = len(set(content.lower().split()))
        total_words = len(content.split())
        information_density = unique_words / max(total_words, 1)
        
        # Coherence score (placeholder - could use embedding similarity)
        coherence_score = min(1.0, information_density * 2)  # Simple heuristic
        
        # Overall quality score
        quality_factors = [
            min(1.0, token_count / self.chunk_size),  # Size appropriateness
            min(1.0, abs(avg_sentence_length - self.ideal_sentence_length) / self.ideal_sentence_length),
            information_density,
            coherence_score
        ]
        
        quality_score = sum(quality_factors) / len(quality_factors)
        
        return ChunkMetrics(
            token_count=token_count,
            char_count=char_count,
            sentence_count=sentence_count,
            avg_sentence_length=avg_sentence_length,
            information_density=information_density,
            coherence_score=coherence_score,
            quality_score=quality_score
        )
    
    def _score_chunks(self, chunks: List[DocumentChunk]) -> List[DocumentChunk]:
        """Apply additional scoring and filtering"""
        scored_chunks = []
        
        for chunk in chunks:
            # Filter by minimum quality
            if chunk.metrics.quality_score >= self.min_quality_score:
                scored_chunks.append(chunk)
            else:
                logger.debug(f"Filtered low quality chunk: {chunk.chunk_id} (score: {chunk.metrics.quality_score:.2f})")
        
        logger.info(f"Kept {len(scored_chunks)}/{len(chunks)} chunks after quality filtering")
        return scored_chunks
    
    def _basic_deduplication(self, chunks: List[DocumentChunk]) -> List[DocumentChunk]:
        """Basic deduplication based on content similarity"""
        if len(chunks) <= 1:
            return chunks
        
        unique_chunks = []
        seen_content = set()
        
        for chunk in chunks:
            # Simple dedup: normalize whitespace and check exact matches
            normalized = " ".join(chunk.content.split())
            
            if normalized not in seen_content:
                seen_content.add(normalized)
                unique_chunks.append(chunk)
            else:
                logger.debug(f"Deduplicated chunk: {chunk.chunk_id}")
        
        logger.info(f"Deduplicated {len(chunks) - len(unique_chunks)} chunks")
        return unique_chunks