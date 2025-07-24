#  api/endpoints/inference.py
"""
Endpoint para inferencia simple con ModelManager
EXTRAÍDO del server.py original
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Optional, Dict, List, Any
from datetime import datetime
import logging

from local_models.model_executor import ModelExecutor
from local_models.model_manager import get_model_manager, MODELS
from utils.gpu_guard import get_gpu_info

logger = logging.getLogger("lab-api.inference")

# Router para este módulo
router = APIRouter(prefix="/inference", tags=["Simple LLM Inference"])

# Global instances (inicializadas en server.py)
executor = None
model_manager = None

def init_inference_service(_executor: ModelExecutor, _model_manager):
    """Inicializar el servicio con las instancias globales"""
    global executor, model_manager
    executor = _executor
    model_manager = _model_manager

# Schemas (movidos desde server.py)
class InferenceRequest(BaseModel):
    prompt: str = Field(..., description="Texto a procesar")
    model: str = Field("mistral7b", description="Modelo a usar")
    strategy: str = Field("optimized", description="Estrategia de carga")
    max_tokens: int = Field(128, ge=1, le=4096)
    temperature: Optional[float] = Field(0.7, ge=0.0, le=1.0)

class InferenceResponse(BaseModel):
    id: str
    timestamp: datetime
    model: str
    strategy: str
    prompt: str
    output: str
    metrics: Dict[str, Any]
    success: bool

class ModelInfo(BaseModel):
    key: str
    name: str
    available: bool
    loaded: bool

@router.post("/", response_model=InferenceResponse)
async def run_inference(request: InferenceRequest):
    """
    ✅ MOVIDO DESDE server.py
    Ejecuta inferencia en un modelo local con cache inteligente
    """
    logger.info(f"Inference request: model={request.model}, strategy={request.strategy}")
    
    try:
        # Validar modelo
        if request.model not in MODELS:
            raise HTTPException(
                status_code=400, 
                detail=f"Model '{request.model}' not available. Use /models to see available models."
            )
        
        # Ejecutar usando ModelExecutor
        result = executor.execute(
            model_key=request.model,
            prompt=request.prompt,
            max_tokens=request.max_tokens,
            strategy=request.strategy,
            temperature=request.temperature
        )
        
        if not result["success"]:
            raise HTTPException(
                status_code=500, 
                detail=f"Model execution failed: {result.get('error', 'Unknown error')}"
            )
        
        # Preparar respuesta
        response = InferenceResponse(
            id=result["run_id"],
            timestamp=datetime.now(),
            model=request.model,
            strategy=request.strategy,
            prompt=request.prompt,
            output=result["output"],
            metrics=result["metrics"],
            success=result["success"]
        )
        
        execution_time = result["metrics"]["total_time_sec"]
        cache_hit = result["metrics"]["cache_hit"]
        cache_status = "HIT" if cache_hit else "MISS"
        
        logger.info(f"Inference completed successfully in {execution_time:.2f}s [CACHE {cache_status}]")
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Inference failed: {e}")
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")

@router.get("/models", response_model=List[ModelInfo])
async def list_models():
    """Lista todos los modelos disponibles"""
    try:
        models_list = []
        for key, name in MODELS.items():
            # Verificar si está cargado en cache
            is_loaded = model_manager.is_model_loaded(key, "optimized") or \
                       model_manager.is_model_loaded(key, "standard") or \
                       model_manager.is_model_loaded(key, "streaming")
            
            models_list.append(ModelInfo(
                key=key,
                name=name,
                available=True,
                loaded=is_loaded
            ))
        
        return models_list
    except Exception as e:
        logger.error(f"Error listing models: {e}")
        raise HTTPException(status_code=500, detail="Failed to list models")