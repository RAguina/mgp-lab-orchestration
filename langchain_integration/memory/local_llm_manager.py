# langchain_integration/memory/local_llm_manager.py

from typing import Dict, Optional, Any
from langchain_integration.wrappers.local_model_wrapper import LocalModelWrapper

class LocalLLMManager:
    """
    Manager centralizado para crear y cachear wrappers de modelos locales
    """
    
    def __init__(self):
        self._cache: Dict[str, LocalModelWrapper] = {}
        
    def get_llm(self, model_key: str, strategy: str = "optimized", **kwargs) -> LocalModelWrapper:
        """
        Obtiene o crea un wrapper de modelo local
        
        Args:
            model_key: Clave del modelo (ej: "mistral7b", "llama3")
            strategy: Estrategia de carga ("standard", "optimized", "streaming")
            **kwargs: Argumentos adicionales para el wrapper
            
        Returns:
            LocalModelWrapper configurado y listo para usar
        """
        # Crear clave única para el cache basada en configuración
        cache_key = f"{model_key}_{strategy}_{hash(frozenset(kwargs.items()))}"
        
        if cache_key not in self._cache:
            print(f"📦 Creando nuevo wrapper para {model_key} con estrategia {strategy}")
            
            # IMPORTANTE: Asegurar que model_key y strategy se pasen correctamente
            # Crear un nuevo diccionario con todos los argumentos necesarios
            wrapper_args = {
                'model_key': model_key,
                'strategy': strategy,
                **kwargs  # Agregar kwargs adicionales
            }
            
            self._cache[cache_key] = LocalModelWrapper(**wrapper_args)
        else:
            print(f"♻️ Reutilizando wrapper existente para {model_key}")
            
        return self._cache[cache_key]
    
    def clear_cache(self):
        """Limpia el cache de modelos"""
        print(f"🧹 Limpiando cache con {len(self._cache)} modelos")
        for wrapper in self._cache.values():
            del wrapper  # Invoca __del__ para liberar recursos
        self._cache.clear()
        
    def list_cached_models(self) -> list:
        """Lista los modelos actualmente en cache"""
        return list(self._cache.keys())


# Instancia global del manager
_global_manager = LocalLLMManager()

def get_global_manager() -> LocalLLMManager:
    """Obtiene la instancia global del manager"""
    return _global_manager