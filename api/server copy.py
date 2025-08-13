# ai-agent-lab/api/server.py

"""
Servidor principal - solo configuración y routing
"""

import os
import sys

# ✅ FIX: Configurar encoding UTF-8 globalmente
os.environ['PYTHONIOENCODING'] = 'utf-8'
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.stderr.reconfigure(encoding='utf-8', errors='replace')

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict
from datetime import datetime
import logging
from contextlib import asynccontextmanager

# Agregar path del lab para imports
LAB_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, LAB_ROOT)

from local_models.model_executor import ModelExecutor
from local_models.model_manager import get_model_manager, MODELS
from utils.gpu_guard import get_gpu_info

# ✅ IMPORTAR TODOS LOS ENDPOINTS
from api.endpoints import inference, orchestrator, cache, system

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("lab-api")

# Global instances
executor = ModelExecutor(save_results=True)
model_manager = get_model_manager()

# ✅ NUEVO: Lifespan context manager para reemplazar @app.on_event
@asynccontextmanager
async def lifespan(app: FastAPI):
    # ✅ STARTUP
    logger.info("AI Agent Lab API v2.0 starting up...")
    logger.info(f"Available models: {list(MODELS.keys())}")
    logger.info(f"GPU Info: {get_gpu_info()}")
    logger.info("Features: Modular endpoints, model caching, LangGraph orchestration")
    
    # Inicializar todos los servicios
    inference.init_inference_service(executor, model_manager)
    cache.init_cache_service(model_manager)
    system.init_system_service(executor, datetime.now())
    
    logger.info("All services initialized successfully")
    
    # ✅ App está corriendo
    yield
    
    # ✅ SHUTDOWN
    logger.info("AI Agent Lab API shutting down...")
    logger.info("Cleaning up model cache...")
    try:
        model_manager.cleanup_all()
        logger.info("Cache cleanup completed")
    except Exception as e:
        logger.error(f"Cleanup error during shutdown: {e}")

# FastAPI app con lifespan
app = FastAPI(
    title="AI Agent Lab API",
    version="2.0.0",
    description="API para modelos locales y orquestación de agentes con cache inteligente",
    lifespan=lifespan  # ✅ NUEVO: usar lifespan en lugar de @app.on_event
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:8000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ✅ REGISTRAR TODOS LOS ROUTERS
app.include_router(inference.router)
app.include_router(orchestrator.router)
app.include_router(cache.router)
app.include_router(system.router)

# Schemas para health
class HealthResponse(BaseModel):
    status: str
    gpu_info: Dict
    models_loaded: int
    models_cached: int
    uptime: str
    services: Dict[str, str]

# Global state para uptime
start_time = datetime.now()

# ✅ ENDPOINTS MÍNIMOS EN SERVER PRINCIPAL
@app.get("/")
async def root():
    """Root endpoint con info básica de todos los servicios"""
    return {
        "service": "AI Agent Lab API",
        "version": "2.0.0",
        "status": "running",
        "endpoints": {
            "inference": "/inference",
            "orchestrator": "/orchestrate", 
            "cache": "/cache",
            "system": "/system",
            "health": "/health"
        },
        "features": [
            "model_caching", 
            "intelligent_memory_management", 
            "langgraph_orchestration",
            "modular_endpoints"
        ]
    }

@app.get("/health", response_model=HealthResponse)
async def health():
    """Health check consolidado de todos los servicios"""
    try:
        gpu_info = get_gpu_info()
        uptime = str(datetime.now() - start_time)
        memory_stats = model_manager.get_memory_stats()
        
        return HealthResponse(
            status="healthy",
            gpu_info=gpu_info,
            models_loaded=len(MODELS),
            models_cached=memory_stats["cache_size"],
            uptime=uptime,
            services={
                "inference": "available",
                "orchestrator": "available" if orchestrator.ORCHESTRATOR_ENABLED else "disabled",
                "cache": "available", 
                "system": "available"
            }
        )
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=500, detail="Health check failed")

# ✅ MODELOS ENDPOINT (mantener aquí por ser central)
@app.get("/models")
async def list_models():
    """
    Lista todos los modelos disponibles
    Mantener en server principal por ser información central
    """
    try:
        models_list = []
        for key, name in MODELS.items():
            # Verificar si está cargado en cache
            is_loaded = model_manager.is_model_loaded(key, "optimized") or \
                       model_manager.is_model_loaded(key, "standard") or \
                       model_manager.is_model_loaded(key, "streaming")
            
            models_list.append({
                "key": key,
                "name": name,
                "available": True,
                "loaded": is_loaded
            })
        
        return models_list
    except Exception as e:
        logger.error(f"Error listing models: {e}")
        raise HTTPException(status_code=500, detail="Failed to list models")

if __name__ == "__main__":
    import sys
    
    # ✅ FIX: Manejar argumentos de línea de comandos
    reload_excludes = [
        "logs/*",
        "*.log", 
        "outputs/*", 
        "metrics/*",
        "__pycache__/*",
        "*.pyc"
    ]
    
    # Agregar excludes adicionales desde argumentos
    exclude_args = [arg for arg in sys.argv if arg.startswith("--reload-exclude=")]
    for arg in exclude_args:
        pattern = arg.split("=", 1)[1].strip('"')
        if pattern not in reload_excludes:
            reload_excludes.append(pattern)
    
    print(f"🔧 WatchFiles excludes: {reload_excludes}")
    
    uvicorn.run(
        "server:app",
        host="127.0.0.1",
        port=8001,
        reload=True,
        log_level="info",
        reload_excludes=reload_excludes,
        reload_includes=["*.py"],
        reload_dirs=["api/", "local_models/"]  # ✅ FIX: Solo monitorear carpetas necesarias
    )