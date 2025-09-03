"""
BGE-M3 Embedding Provider
Production implementation with normalized embeddings
"""

from typing import List, Union
import logging
from sentence_transformers import SentenceTransformer

logger = logging.getLogger("rag.embeddings.bge_m3")


class BGEM3EmbeddingProvider:
    """BGE-M3 embedding provider with L2 normalization"""
    
    def __init__(self, device: str = "cuda"):
        self.device = device
        self.model = None
        self._load_model()
        
    def _load_model(self):
        """Load BGE-M3 model with proper error handling"""
        try:
            logger.info(f"Loading BGE-M3 model on {self.device}")
            self.model = SentenceTransformer("BAAI/bge-m3", device=self.device)
            logger.info(f"BGE-M3 loaded successfully, embedding dim: {self.model.get_sentence_embedding_dimension()}")
        except Exception as e:
            logger.error(f"Failed to load BGE-M3 model: {e}")
            if self.device == "cuda":
                logger.warning("Falling back to CPU")
                self.device = "cpu"
                self.model = SentenceTransformer("BAAI/bge-m3", device=self.device)
            else:
                raise
    
    def embed_chunks(self, chunks: List[dict]) -> List[List[float]]:
        """
        Generate normalized embeddings for document chunks
        
        Args:
            chunks: List of dicts with 'content' field
            
        Returns:
            List of normalized embedding vectors
        """
        if not chunks:
            return []
            
        texts = [chunk["content"] for chunk in chunks]
        return self._encode_texts(texts)
    
    def embed_documents(self, documents: List[str]) -> List[List[float]]:
        """Generate normalized embeddings for documents"""
        if not documents:
            return []
        return self._encode_texts(documents)
        
    def embed_query(self, query: str) -> List[float]:
        """Generate normalized query embedding"""
        if not query.strip():
            raise ValueError("Query cannot be empty")
            
        embeddings = self._encode_texts([query])
        return embeddings[0]
    
    def _encode_texts(self, texts: List[str]) -> List[List[float]]:
        """Internal encoding with normalization"""
        try:
            embeddings = self.model.encode(
                texts,
                normalize_embeddings=True,  # L2 normalization for cosine similarity
                show_progress_bar=len(texts) > 10,  # Progress for large batches
                batch_size=32  # Efficient batch size
            )
            return embeddings.tolist()
        except Exception as e:
            logger.error(f"Failed to encode {len(texts)} texts: {e}")
            raise
    
    def get_embedding_dimension(self) -> int:
        """Get the embedding dimension (1024 for BGE-M3)"""
        return self.model.get_sentence_embedding_dimension()
    
    def __del__(self):
        """Cleanup CUDA memory on destruction"""
        if hasattr(self, 'model') and self.model and self.device == "cuda":
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except:
                pass