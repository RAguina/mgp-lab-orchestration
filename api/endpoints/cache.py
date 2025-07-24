# 1. api/endpoints/cache.py
"""
Endpoints para manejo de cache de modelos
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Dict, Any
import logging

from local_models.model_manager import MODELS
from utils.gpu_guard import get_gpu_info

logger = logging.getLogger("lab-api.cache")

router = APIRouter(prefix="/cache", tags=["Model Cache"])

# Global instances (inicializadas en server.py)
model_manager = None

def init_cache_service(_model_manager):
    """Inicializar el servicio con las instancias globales"""
    global model_manager
    model_manager = _model_manager

class CacheStatus(BaseModel):
    cached_models: Dict[str, Dict[str, Any]]
    memory_stats: Dict[str, Any]
    cache_size: int

@router.get("/", response_model=CacheStatus)
async def get_cache_status():
    """Obtiene el estado del cache de modelos"""
    try:
        memory_stats = model_manager.get_memory_stats()
        loaded_models = model_manager.get_loaded_models()
        
        return CacheStatus(
            cached_models=loaded_models,
            memory_stats=memory_stats,
            cache_size=memory_stats["cache_size"]
        )
    except Exception as e:
        logger.error(f"Cache status failed: {e}")
        raise HTTPException(status_code=500, detail="Failed to get cache status")

@router.post("/clear")
async def clear_cache():
    """Limpia todo el cache de modelos"""
    try:
        logger.info("Manual cache clear requested")
        model_manager.cleanup_all()
        
        return {
            "success": True,
            "message": "Cache cleared successfully",
            "gpu_info": get_gpu_info()
        }
    except Exception as e:
        logger.error(f"Cache clear failed: {e}")
        raise HTTPException(status_code=500, detail=f"Cache clear failed: {str(e)}")

@router.delete("/{model_key}")
async def unload_model(model_key: str, strategy: str = "optimized"):
    """Descarga un modelo específico del cache"""
    try:
        if model_key not in MODELS:
            raise HTTPException(status_code=404, detail=f"Model '{model_key}' not found")
        
        logger.info(f"Manual unload requested: {model_key}_{strategy}")
        model_manager.unload_model(model_key, strategy)
        
        return {
            "success": True,
            "message": f"Model {model_key} with strategy {strategy} unloaded",
            "gpu_info": get_gpu_info()
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Model unload failed: {e}")
        raise HTTPException(status_code=500, detail=f"Unload failed: {str(e)}")