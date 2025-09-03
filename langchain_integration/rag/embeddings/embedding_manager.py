"""
Embedding Manager - Singleton pattern for cached model instances
Prevents recreating heavy models on each request
"""

from typing import Dict, Optional
import logging
from .bge_m3_provider import BGEM3EmbeddingProvider

logger = logging.getLogger("rag.embeddings.manager")


class EmbeddingManager:
    """Singleton manager for cached embedding providers"""
    
    _instance: Optional['EmbeddingManager'] = None
    _providers: Dict[str, BGEM3EmbeddingProvider] = {}
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            logger.info("EmbeddingManager singleton created")
        return cls._instance
    
    @classmethod
    def get_instance(cls) -> 'EmbeddingManager':
        """Get singleton instance"""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
    
    def get_provider(self, model_name: str, device: str = "cuda") -> BGEM3EmbeddingProvider:
        """
        Get cached embedding provider
        
        Args:
            model_name: Currently only supports "bge-m3"
            device: "cuda" or "cpu"
            
        Returns:
            Cached embedding provider instance
        """
        cache_key = f"{model_name}_{device}"
        
        if cache_key not in self._providers:
            logger.info(f"Creating new embedding provider: {cache_key}")
            
            if model_name == "bge-m3":
                self._providers[cache_key] = BGEM3EmbeddingProvider(device=device)
            else:
                raise ValueError(f"Unsupported embedding model: {model_name}")
                
        return self._providers[cache_key]
    
    def clear_cache(self):
        """Clear all cached providers (useful for testing)"""
        logger.info("Clearing embedding provider cache")
        self._providers.clear()
    
    def get_cached_models(self) -> Dict[str, str]:
        """Get list of currently cached models"""
        return {key: type(provider).__name__ for key, provider in self._providers.items()}


# Global singleton accessor
def get_embedding_manager() -> EmbeddingManager:
    """Get the global embedding manager instance"""
    return EmbeddingManager.get_instance()