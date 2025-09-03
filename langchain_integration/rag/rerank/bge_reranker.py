"""
BGE Reranker Implementation
Enhanced reranking with batch processing and GPU support
"""

import logging
from typing import List, Dict, Any, Optional, Union
import time

try:
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

logger = logging.getLogger("rag.rerank")


class BGEReranker:
    """
    BGE-based reranker for improving RAG search results
    
    Features:
    - Batch processing for efficiency
    - GPU/CPU support with automatic fallback
    - Memory management and cleanup
    - Configurable ranking parameters
    """
    
    def __init__(self, 
                 model_name: str = "BAAI/bge-reranker-base",
                 device: str = "auto",
                 max_length: int = 512,
                 batch_size: int = 16):
        """
        Initialize BGE reranker
        
        Args:
            model_name: HuggingFace model name
            device: Device to use ("auto", "cuda", "cpu") 
            max_length: Maximum token length for input
            batch_size: Batch size for processing
        """
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("transformers library not available. Install with: pip install transformers")
        
        self.model_name = model_name
        self.max_length = max_length
        self.batch_size = batch_size
        
        # Device selection
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        logger.info(f"Initializing BGE Reranker on {self.device}")
        
        # Load model and tokenizer
        self._load_model()
        
        logger.info(f"BGE Reranker loaded: {model_name}")
    
    def _load_model(self):
        """Load tokenizer and model"""
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
            
            # Move to device and set to eval mode
            self.model.to(self.device)
            self.model.eval()
            
            logger.info(f"Model loaded successfully on {self.device}")
            
        except Exception as e:
            logger.error(f"Failed to load reranker model: {e}")
            raise
    
    def rerank(self, 
               query: str, 
               passages: List[Dict[str, Any]], 
               top_k: int = 5,
               max_passages: int = 64,
               return_scores: bool = False) -> Union[List[Dict[str, Any]], tuple]:
        """
        Rerank passages based on query relevance
        
        Args:
            query: Search query
            passages: List of passage dicts with 'content' or 'excerpt' keys
            top_k: Number of top results to return
            max_passages: Maximum passages to process (for performance)
            return_scores: Whether to return relevance scores
            
        Returns:
            Reranked passages (and scores if requested)
        """
        if not passages:
            return [] if not return_scores else ([], [])
        
        # Limit input size for performance
        passages = passages[:max_passages]
        
        logger.debug(f"Reranking {len(passages)} passages for query: '{query[:50]}...'")
        
        start_time = time.time()
        
        try:
            # Prepare input pairs
            pairs = []
            valid_passages = []
            
            for passage in passages:
                # Extract content (prefer full content, fallback to excerpt)
                content = passage.get("content") or passage.get("excerpt", "")
                if content.strip():  # Only process non-empty content
                    pairs.append((query, content))
                    valid_passages.append(passage)
            
            if not pairs:
                logger.warning("No valid content found in passages")
                return [] if not return_scores else ([], [])
            
            # Calculate relevance scores in batches
            scores = self._calculate_scores_batched(pairs)
            
            # Sort by relevance score (descending)
            scored_passages = list(zip(valid_passages, scores))
            scored_passages.sort(key=lambda x: x[1], reverse=True)
            
            # Return top-k results
            top_passages = scored_passages[:top_k]
            
            rerank_time = time.time() - start_time
            logger.debug(f"Reranking completed in {rerank_time:.3f}s")
            
            if return_scores:
                passages_only = [passage for passage, score in top_passages]
                scores_only = [score for passage, score in top_passages]
                return passages_only, scores_only
            else:
                return [passage for passage, score in top_passages]
                
        except Exception as e:
            logger.error(f"Reranking failed: {e}")
            # Fallback: return original order
            return passages[:top_k] if not return_scores else (passages[:top_k], [0.0] * min(top_k, len(passages)))
    
    def _calculate_scores_batched(self, pairs: List[tuple]) -> List[float]:
        """Calculate relevance scores using batched inference"""
        
        all_scores = []
        
        with torch.inference_mode():  # More efficient than no_grad()
            for i in range(0, len(pairs), self.batch_size):
                batch_pairs = pairs[i:i + self.batch_size]
                
                # Tokenize batch
                inputs = self.tokenizer(
                    batch_pairs,
                    padding=True,
                    truncation=True,
                    return_tensors="pt",
                    max_length=self.max_length
                ).to(self.device)
                
                # Forward pass
                outputs = self.model(**inputs, return_dict=True)
                batch_scores = outputs.logits.view(-1).float()
                
                # Convert to CPU and extend results
                all_scores.extend(batch_scores.cpu().tolist())
                
                # Free GPU memory after each batch
                if self.device.startswith("cuda"):
                    torch.cuda.empty_cache()
        
        return all_scores
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get reranker model information"""
        return {
            "model_name": self.model_name,
            "device": self.device,
            "max_length": self.max_length,
            "batch_size": self.batch_size,
            "parameters": sum(p.numel() for p in self.model.parameters()) if hasattr(self, 'model') else 0
        }
    
    def cleanup(self):
        """Clean up model resources"""
        if hasattr(self, 'model'):
            del self.model
        if hasattr(self, 'tokenizer'):
            del self.tokenizer
            
        if self.device.startswith("cuda"):
            torch.cuda.empty_cache()
            
        logger.info("BGE Reranker resources cleaned up")


class RerankerManager:
    """
    Singleton manager for reranker instances
    Provides caching and efficient resource management
    """
    
    _instance = None
    _rerankers = {}
    
    @classmethod
    def get_instance(cls):
        """Get singleton instance"""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
    
    def get_reranker(self, 
                    model_name: str = "bge-base",
                    device: str = "auto",
                    **kwargs) -> BGEReranker:
        """
        Get cached reranker instance
        
        Args:
            model_name: Short name or full model path
            device: Device preference
            **kwargs: Additional reranker arguments
            
        Returns:
            BGEReranker instance
        """
        # Normalize model name
        model_map = {
            "bge-base": "BAAI/bge-reranker-base",
            "bge-large": "BAAI/bge-reranker-large", 
            "bge-v2": "BAAI/bge-reranker-v2-m3"
        }
        
        full_model_name = model_map.get(model_name, model_name)
        cache_key = f"{full_model_name}_{device}"
        
        if cache_key not in self._rerankers:
            try:
                logger.info(f"Loading new reranker: {full_model_name}")
                self._rerankers[cache_key] = BGEReranker(
                    model_name=full_model_name,
                    device=device,
                    **kwargs
                )
            except Exception as e:
                logger.error(f"Failed to load reranker {full_model_name}: {e}")
                raise
        
        return self._rerankers[cache_key]
    
    def cleanup_all(self):
        """Cleanup all cached rerankers"""
        for reranker in self._rerankers.values():
            reranker.cleanup()
        self._rerankers.clear()
        logger.info("All rerankers cleaned up")
    
    def list_rerankers(self) -> List[str]:
        """List currently loaded rerankers"""
        return list(self._rerankers.keys())


# Global singleton accessor
def get_reranker_manager() -> RerankerManager:
    """Get global reranker manager instance"""
    return RerankerManager.get_instance()


# Convenience function
def rerank_passages(query: str, 
                   passages: List[Dict[str, Any]], 
                   top_k: int = 5,
                   model: str = "bge-base",
                   **kwargs) -> List[Dict[str, Any]]:
    """
    Convenience function for reranking passages
    
    Args:
        query: Search query
        passages: List of passages to rerank
        top_k: Number of results to return
        model: Reranker model to use
        **kwargs: Additional arguments
        
    Returns:
        Reranked passages
    """
    manager = get_reranker_manager()
    reranker = manager.get_reranker(model, **kwargs)
    return reranker.rerank(query, passages, top_k=top_k)