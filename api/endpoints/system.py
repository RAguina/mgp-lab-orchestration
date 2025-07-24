# 2. api/endpoints/system.py
"""
Endpoints para métricas y limpieza del sistema
"""

from fastapi import APIRouter, HTTPException
from datetime import datetime
import logging
import gc
import torch

from local_models.model_manager import MODELS
from utils.gpu_guard import get_gpu_info, clear_gpu_memory

logger = logging.getLogger("lab-api.system")

router = APIRouter(prefix="/system", tags=["System Management"])

# Global instances (inicializadas en server.py)
executor = None
start_time = None

def init_system_service(_executor, _start_time):
    """Inicializar el servicio con las instancias globales"""
    global executor, start_time
    executor = _executor
    start_time = _start_time

@router.get("/metrics")
async def get_metrics():
    """Endpoint para métricas del sistema"""
    try:
        execution_stats = executor.get_execution_stats()
        
        return {
            "gpu": get_gpu_info(),
            "uptime": str(datetime.now() - start_time),
            "models_available": len(MODELS),
            "execution_stats": execution_stats,
            "api_version": "2.0.0"
        }
    except Exception as e:
        logger.error(f"Metrics failed: {e}")
        raise HTTPException(status_code=500, detail="Failed to get metrics")

@router.post("/cleanup")
async def cleanup_memory():
    """Fuerza limpieza de memoria GPU (legacy endpoint)"""
    try:
        logger.info("Forcing GPU memory cleanup...")
        
        # Limpieza agresiva
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        clear_gpu_memory()
        
        gpu_info = get_gpu_info()
        logger.info(f"Memory cleanup completed. GPU status: {gpu_info}")
        
        return {
            "success": True,
            "message": "GPU memory cleaned (consider using /cache/clear for model cache)",
            "gpu_info": gpu_info
        }
    except Exception as e:
        logger.error(f"Cleanup failed: {e}")
        raise HTTPException(status_code=500, detail=f"Cleanup failed: {str(e)}")