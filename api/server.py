# ai-agent-lab/api/server.py
"""
Servidor principal - configuración, routers y ciclo de vida.
V3: inyecta ProviderGateway en el execution_node al iniciar.
"""

import os
import sys
import uvicorn
import logging
from datetime import datetime
from typing import Dict
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# UTF-8 robusto
os.environ['PYTHONIOENCODING'] = 'utf-8'
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.stderr.reconfigure(encoding='utf-8', errors='replace')

# Agregar path del lab para imports
LAB_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, LAB_ROOT)

# Core locales
from local_models.model_executor import ModelExecutor
from local_models.model_manager import get_model_manager, MODELS
from utils.gpu_guard import get_gpu_info

# Gateway para inyectar en el nodo de ejecución
from providers.provider_gateway import ProviderGateway
import langchain_integration.langgraph.nodes.execution_node as execution_mod

# Routers
from api.endpoints import inference, orchestrator, cache, system

# Logging básico (no duplica handlers si ya existen)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("lab-api")

# Instancias globales mínimas
executor = ModelExecutor(save_results=True)
model_manager = get_model_manager()
gateway = None  # se setea en startup

@asynccontextmanager
async def lifespan(app: FastAPI):
    # STARTUP
    global gateway
    logger.info("AI Agent Lab API v3.0 starting up...")
    logger.info(f"Available models: {list(MODELS.keys())}")
    logger.info(f"GPU Info: {get_gpu_info()}")
    logger.info("Features: Model caching, LangGraph orchestration, ProviderGateway")

    # Inyección de dependencias
    inference.init_inference_service(executor, model_manager)
    cache.init_cache_service(model_manager)
    system.init_system_service(executor, datetime.now())

    # Crear ProviderGateway con arquitectura modular
    from providers.local.local_provider import LocalProvider
    
    local_provider = LocalProvider(executor=executor)
    gateway = ProviderGateway(providers={"local": local_provider})
    
    # Gateway se inyecta vía AgentState.services en routing_agent
    # Esto hace el gateway disponible globalmente para el lab

    logger.info("All services initialized successfully")
    yield

    # SHUTDOWN
    logger.info("AI Agent Lab API shutting down...")
    logger.info("Cleaning up model cache...")
    try:
        model_manager.cleanup_all()
        logger.info("Cache cleanup completed")
    except Exception as e:
        logger.error(f"Cleanup error during shutdown: {e}")


app = FastAPI(
    title="AI Agent Lab API",
    version="3.0.0",
    description="API para modelos locales y orquestación de agentes con cache + gateway",
    lifespan=lifespan
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:8000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Routers
app.include_router(inference.router)
app.include_router(orchestrator.router)
app.include_router(cache.router)
app.include_router(system.router)

# Schemas
class HealthResponse(BaseModel):
    status: str
    gpu_info: Dict
    models_loaded: int
    models_cached: int
    uptime: str
    services: Dict[str, str]

start_time = datetime.now()

@app.get("/")
async def root():
    return {
        "service": "AI Agent Lab API",
        "version": "3.0.0",
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
            "provider_gateway"
        ]
    }

@app.get("/health", response_model=HealthResponse)
async def health():
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
                "orchestrator": "available",
                "cache": "available",
                "system": "available"
            }
        )
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=500, detail="Health check failed")

@app.get("/models")
async def list_models():
    """
    Lista modelos disponibles (usa MODELS locales por ahora).
    """
    try:
        models_list = []
        for key, name in MODELS.items():
            is_loaded = (
                model_manager.is_model_loaded(key, "optimized") or
                model_manager.is_model_loaded(key, "standard") or
                model_manager.is_model_loaded(key, "streaming")
            )
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

    reload_excludes = [
        "logs/*",
        "*.log",
        "outputs/*",
        "metrics/*",
        "__pycache__/*",
        "*.pyc"
    ]

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
        reload_dirs=["api/", "local_models/", "langchain_integration/", "providers/"]
    )
