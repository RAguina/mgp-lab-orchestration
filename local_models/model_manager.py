# local_models/model_manager.py
"""
ModelManager: Singleton para gestión de cache y pool de modelos LLM.
Responsabilidades:
- Cargar modelos bajo demanda
- Mantener cache en memoria
- Liberar modelos cuando sea necesario
- Gestión inteligente de VRAM
"""

import gc
import time
import threading
from typing import Dict, Tuple, Optional, Any
from dataclasses import dataclass
from datetime import datetime

from utils.logger import get_logger
from utils.gpu_guard import get_gpu_info, clear_gpu_memory
from langchain_integration.wrappers.hf_pipeline_wrappers import (
    standard,
    optimized,
    streaming,
)

STRATEGY_LOADERS = {
    "standard": standard.load_model,
    "optimized": optimized.load_model,
    "streaming": streaming.load_model,
}

MODELS = {
    "llama3": "meta-llama/Meta-Llama-3-8B-Instruct",
    "mistral7b": "mistralai/Mistral-7B-Instruct-v0.2",
    "deepseek7b": "deepseek-ai/deepseek-coder-6.7b-instruct",
    "deepseek-coder": "deepseek-ai/deepseek-coder-6.7b-instruct"
}

@dataclass
class LoadedModel:
    """Contenedor para modelos cargados con metadata"""
    model: Any
    tokenizer: Any
    pipeline_obj: Any
    model_key: str
    model_name: str
    strategy: str
    loaded_at: datetime
    last_used: datetime
    memory_usage_gb: float


class ModelManager:
    """Singleton para gestión de modelos LLM"""
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if hasattr(self, '_initialized'):
            return
            
        self._cache: Dict[Tuple[str, str], LoadedModel] = {}
        self._lock = threading.Lock()
        self._logger = get_logger("model_manager", "logs/model_manager", enable_file_logging=True)
        self._max_vram_usage = 6.0  # GB - ajustable según tu GPU
        self._initialized = True
        
        self._logger.info("manager_initialized", max_vram=self._max_vram_usage)
    
    def get_cache_key(self, model_key: str, strategy: str) -> Tuple[str, str]:
        """Genera clave única para el cache"""
        return (model_key, strategy)
    
    def is_model_loaded(self, model_key: str, strategy: str) -> bool:
        """Verifica si un modelo está cargado en cache"""
        cache_key = self.get_cache_key(model_key, strategy)
        return cache_key in self._cache
    
    def get_loaded_models(self) -> Dict[str, Dict[str, Any]]:
        """Retorna información de modelos cargados"""
        models_info = {}
        for (model_key, strategy), loaded_model in self._cache.items():
            models_info[f"{model_key}_{strategy}"] = {
                "model_key": model_key,
                "strategy": strategy,
                "loaded_at": loaded_model.loaded_at.isoformat(),
                "last_used": loaded_model.last_used.isoformat(),
                "memory_usage_gb": loaded_model.memory_usage_gb
            }
        return models_info
    
    def _check_memory_pressure(self) -> bool:
        """Verifica si hay presión de memoria"""
        gpu_info = get_gpu_info()
        current_usage = gpu_info.get("allocated_gb", 0)
        return current_usage >= self._max_vram_usage
    
    def _free_least_recently_used(self):
        """Libera el modelo menos recientemente usado"""
        if not self._cache:
            return
            
        # Encontrar modelo LRU
        lru_key = min(self._cache.keys(), 
                     key=lambda k: self._cache[k].last_used)
        
        self._logger.info("freeing_lru_model", 
                         model=f"{lru_key[0]}_{lru_key[1]}")
        self._unload_model_internal(lru_key)
    
    def _unload_model_internal(self, cache_key: Tuple[str, str]):
        """Libera un modelo específico del cache"""
        if cache_key not in self._cache:
            return
            
        loaded_model = self._cache[cache_key]
        
        try:
            # Liberar objetos
            del loaded_model.model
            del loaded_model.tokenizer
            del loaded_model.pipeline_obj
            
            # Remover del cache
            del self._cache[cache_key]
            
            # Limpieza de memoria
            gc.collect()
            clear_gpu_memory()
            
            self._logger.info("model_unloaded", 
                            model=f"{cache_key[0]}_{cache_key[1]}",
                            gpu_info=get_gpu_info())
            
        except Exception as e:
            self._logger.error("unload_error", error=str(e))
    
    def load_model(self, model_key: str, strategy: str = "optimized", **kwargs) -> LoadedModel:
        """
        Carga un modelo, reutilizando del cache si está disponible.
        """
        cache_key = self.get_cache_key(model_key, strategy)
        
        with self._lock:
            # Si está en cache, actualizar last_used y retornar
            if cache_key in self._cache:
                loaded_model = self._cache[cache_key]
                loaded_model.last_used = datetime.now()
                self._logger.info("model_cache_hit", 
                                model=f"{model_key}_{strategy}")
                return loaded_model
            
            # Verificar que el modelo existe
            if model_key not in MODELS:
                raise ValueError(f"Modelo no encontrado: {model_key}")
            
            model_name = MODELS[model_key]
            load_fn = STRATEGY_LOADERS.get(strategy, optimized.load_model)
            
            # Verificar presión de memoria antes de cargar
            while self._check_memory_pressure() and self._cache:
                self._free_least_recently_used()
            
            self._logger.info("loading_model", 
                            model=model_key, 
                            strategy=strategy,
                            gpu_before=get_gpu_info())
            
            try:
                start_time = time.time()
                
                # Cargar modelo
                model, tokenizer, pipeline_obj = load_fn(
                    model_name, self._logger, **kwargs
                )
                
                load_time = time.time() - start_time
                gpu_after = get_gpu_info()
                memory_usage = gpu_after.get("allocated_gb", 0)
                
                # Crear objeto LoadedModel
                loaded_model = LoadedModel(
                    model=model,
                    tokenizer=tokenizer,
                    pipeline_obj=pipeline_obj,
                    model_key=model_key,
                    model_name=model_name,
                    strategy=strategy,
                    loaded_at=datetime.now(),
                    last_used=datetime.now(),
                    memory_usage_gb=memory_usage
                )
                
                # Agregar al cache
                self._cache[cache_key] = loaded_model
                
                self._logger.info("model_loaded_successfully",
                                model=f"{model_key}_{strategy}",
                                load_time=round(load_time, 2),
                                memory_usage_gb=memory_usage,
                                gpu_after=gpu_after)
                
                return loaded_model
                
            except Exception as e:
                self._logger.error("model_load_failed",
                                 model=f"{model_key}_{strategy}",
                                 error=str(e))
                raise
    
    def get_model(self, model_key: str, strategy: str = "optimized") -> Optional[LoadedModel]:
        """
        Obtiene un modelo del cache sin cargarlo.
        Retorna None si no está cargado.
        """
        cache_key = self.get_cache_key(model_key, strategy)
        return self._cache.get(cache_key)
    
    def unload_model(self, model_key: str, strategy: str):
        """Libera un modelo específico"""
        cache_key = self.get_cache_key(model_key, strategy)
        
        with self._lock:
            if cache_key in self._cache:
                self._logger.info("unloading_model", 
                                model=f"{model_key}_{strategy}")
                self._unload_model_internal(cache_key)
            else:
                self._logger.warning("unload_model_not_found",
                                   model=f"{model_key}_{strategy}")
    
    def cleanup_all(self):
        """Libera todos los modelos del cache"""
        with self._lock:
            self._logger.info("cleanup_all_models", count=len(self._cache))
            
            cache_keys = list(self._cache.keys())
            for cache_key in cache_keys:
                self._unload_model_internal(cache_key)
            
            # Limpieza final
            gc.collect()
            clear_gpu_memory()
            
            self._logger.info("cleanup_complete", gpu_info=get_gpu_info())
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Retorna estadísticas de memoria y cache"""
        gpu_info = get_gpu_info()
        return {
            "cache_size": len(self._cache),
            "loaded_models": list(self._cache.keys()),
            "gpu_info": gpu_info,
            "memory_pressure": self._check_memory_pressure(),
            "max_vram_limit_gb": self._max_vram_usage
        }
    
    def set_max_vram_usage(self, max_gb: float):
        """Configura el límite máximo de VRAM"""
        self._max_vram_usage = max_gb
        self._logger.info("vram_limit_updated", max_vram=max_gb)


# Función de conveniencia para obtener el singleton
def get_model_manager() -> ModelManager:
    """Retorna la instancia singleton del ModelManager"""
    return ModelManager()