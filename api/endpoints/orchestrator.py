#  api/endpoints/orchestrator.py
"""
Endpoint para orquestador LangGraph
NUEVO - maneja execution_type: "orchestrator"
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Optional, Dict, List, Any
from datetime import datetime
import logging
import time

logger = logging.getLogger("lab-api.orchestrator")

# Router para este módulo
router = APIRouter(prefix="/orchestrate", tags=["LangGraph Orchestrator"])

# ✅ IMPORTAR TU ORQUESTADOR (con fallback seguro)
try:
    from langchain_integration.langgraph.routing_agent import run_orchestrator
    ORCHESTRATOR_ENABLED = True
    logger.info(" Orchestrator loaded successfully")
except ImportError as e:
    ORCHESTRATOR_ENABLED = False
    logger.warning(f"⚠️ Orchestrator not available: {e}")

# Schemas para orquestador
class OrchestratorRequest(BaseModel):
    prompt: str = Field(..., description="Prompt para el orquestador")
    model: str = Field("mistral7b", description="Modelo base a usar")
    flow_type: str = Field("challenge", description="Tipo de flujo a ejecutar")
    agents: Optional[List[str]] = Field([], description="Lista de agentes")
    tools: Optional[List[str]] = Field([], description="Lista de herramientas")
    verbose: Optional[bool] = Field(False, description="Logging detallado")
    enable_history: Optional[bool] = Field(True, description="Incluir historial")
    retry_on_error: Optional[bool] = Field(True, description="Reintentar en error")
    
    # RAG Configuration (retrocompatible)
    rag_config: Optional[Dict[str, Any]] = Field({}, description="Configuración RAG específica")
    embedding_model: Optional[str] = Field(None, description="Modelo de embeddings (bge-m3, e5, etc.)")
    vector_store: Optional[str] = Field(None, description="Base vectorial (milvus, weaviate, pinecone)")
    retrieval_config: Optional[Dict[str, Any]] = Field({}, description="Config de retrieval (top_k, threshold, etc.)")

class OrchestratorResponse(BaseModel):
    id: str
    timestamp: datetime
    success: bool
    output: str
    model: str
    flow: Dict[str, Any]
    metrics: Dict[str, Any]

@router.get("/")
async def orchestrator_health():
    """
    Health check para el orchestrator
    """
    return {
        "orchestrator_available": ORCHESTRATOR_ENABLED,
        "endpoint": "/orchestrate",
        "methods": ["POST"],
        "description": "LangGraph orchestrator endpoint",
        "supported_flows": ["linear", "challenge", "rag_simple", "rag_advanced", "rag_hybrid"] if ORCHESTRATOR_ENABLED else [],
        "supported_embeddings": ["bge-m3", "e5-large", "sentence-transformers"],
        "supported_vector_stores": ["milvus", "weaviate", "pinecone", "chroma"],
        "workers": ["task_analyzer", "resource_monitor", "executor", "validator", "summarizer"] if ORCHESTRATOR_ENABLED else []
    }

@router.post("/", response_model=OrchestratorResponse)
@router.post("", response_model=OrchestratorResponse)
async def run_orchestrator_endpoint(request: OrchestratorRequest):
    """
    ✅ NUEVO ENDPOINT
    Ejecuta el orquestador LangGraph con workers especializados
    """
    if not ORCHESTRATOR_ENABLED:
        raise HTTPException(
            status_code=503,
            detail="Orchestrator not available - check LangGraph installation"
        )
    
    # Generar ID único
    execution_id = f"orch_{int(time.time())}"
    
    try:
        if not request.prompt:
            raise HTTPException(status_code=400, detail="Prompt is required")
        
        # Log del request
        logger.info(f"[{execution_id}] Orchestrator request:")
        logger.info(f"[{execution_id}]   Prompt: {request.prompt[:100]}...")
        logger.info(f"[{execution_id}]   Model: {request.model}")
        logger.info(f"[{execution_id}]   Flow Type: {request.flow_type}")
        logger.info(f"[{execution_id}]   Agents: {request.agents}")
        logger.info(f"[{execution_id}]   Tools: {request.tools}")
        
        # RAG validation and processing
        is_rag_request = request.embedding_model or request.vector_store or request.rag_config
        
        if is_rag_request:
            logger.info(f"[{execution_id}] RAG Request detected:")
            if request.embedding_model:
                logger.info(f"[{execution_id}]   Embedding: {request.embedding_model}")
            if request.vector_store:
                logger.info(f"[{execution_id}]   Vector Store: {request.vector_store}")
            if request.rag_config:
                logger.info(f"[{execution_id}]   RAG Config: {request.rag_config}")
            
            # Check if RAG flows are available
            from langchain_integration.langgraph.orchestration import list_available_flows
            available_flows = list_available_flows()
            rag_flows = [f for f in available_flows if f.startswith('rag_')]
            
            if not rag_flows and request.flow_type.startswith('rag_'):
                logger.warning(f"[{execution_id}] RAG flow '{request.flow_type}' requested but not implemented")
                logger.info(f"[{execution_id}] Using MOCK RAG with 'challenge' flow")
                
                # Use mock RAG implementation
                try:
                    from langchain_integration.tools.rag_mock import mock_rag_retrieval
                    rag_result = mock_rag_retrieval(
                        query=request.prompt,
                        embedding_model=request.embedding_model or "bge-m3",
                        vector_store=request.vector_store or "milvus", 
                        rag_config=request.rag_config or {}
                    )
                    
                    request.prompt = rag_result["enriched_prompt"]
                    request.flow_type = "challenge"  # Safe fallback
                    
                    logger.info(f"[{execution_id}] Mock RAG applied: {rag_result['metadata']['documents_found']} docs retrieved")
                    
                except Exception as e:
                    logger.error(f"[{execution_id}] Mock RAG failed: {e}")
                    # Ultimate fallback: simple context annotation
                    enriched_prompt = f"[RAG Context Request: {request.embedding_model or 'default'} + {request.vector_store or 'default'}]\n\n{request.prompt}"
                    request.prompt = enriched_prompt
                    request.flow_type = "challenge"
        
        # Prepare RAG configuration for Lab
        rag_tools = []
        if request.embedding_model:
            rag_tools.append(f"embedding_{request.embedding_model.replace('-', '_')}")
        if request.vector_store:
            rag_tools.append(f"vectorstore_{request.vector_store}")
        
        # Merge RAG tools with existing tools
        all_tools = list(request.tools) + rag_tools
        
        # ✅ LLAMAR A TU FUNCIÓN EXISTENTE con modelo y flow_type especificados
        start_time = time.time()
        orchestrator_result = run_orchestrator(
            request.prompt, 
            model=request.model,
            flow_type=request.flow_type,
            tools=all_tools if all_tools else None,
            rag_config=request.rag_config if request.rag_config else None
        )
        total_time = time.time() - start_time
        
        # Verificar que el resultado sea válido - permitir outputs vacíos para manejo de errores
        if not orchestrator_result:
            raise HTTPException(
                status_code=500,
                detail="Orchestrator returned empty result"
            )
        
        # ✅ CONVERTIR al formato esperado por el backend
        response = OrchestratorResponse(
            id=execution_id,
            timestamp=datetime.now(),
            success=True,
            output=orchestrator_result.get("output", ""),
            model=request.model,
            
            # Flow data para el backend
            flow={
                "nodes": orchestrator_result.get("flow", {}).get("nodes", []),
                "edges": orchestrator_result.get("flow", {}).get("edges", [])
            },
            
            # Métricas para el backend  
            metrics={
                "total_time": total_time,
                "total_time_sec": total_time,  # Backward compatibility
                "tokens_generated": orchestrator_result.get("metrics", {}).get("tokensGenerated", 0),
                "models_used": orchestrator_result.get("metrics", {}).get("modelsUsed", [request.model]),
                "cache_hit": orchestrator_result.get("metrics", {}).get("cacheHit", False),
                "load_time": orchestrator_result.get("metrics", {}).get("loadTime", 0) / 1000,  # Convert to seconds
                "inference_time": orchestrator_result.get("metrics", {}).get("inferenceTime", 0) / 1000,
                "workers_executed": orchestrator_result.get("metrics", {}).get("workersExecuted", 0),
                "quality_score": orchestrator_result.get("metrics", {}).get("qualityScore", 0)
            }
        )
        
        logger.info(f"[{execution_id}]   Orchestrator completed in {total_time:.2f}s")
        logger.info(f"[{execution_id}]   Workers: {response.metrics['workers_executed']}")
        logger.info(f"[{execution_id}]   Tokens: {response.metrics['tokens_generated']}")
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        error_msg = f"Orchestrator execution failed: {str(e)}"
        logger.error(f"[{execution_id}] ❌ {error_msg}")
        
        # Retornar error estructurado
        return OrchestratorResponse(
            id=execution_id,
            timestamp=datetime.now(),
            success=False,
            output=f"Error ejecutando orquestador: {str(e)}",
            model=request.model,
            flow={
                "nodes": [{
                    "id": "error",
                    "name": "Orchestrator Error", 
                    "type": "error",
                    "status": "error",
                    "output": error_msg
                }],
                "edges": []
            },
            metrics={
                "total_time": 0,
                "tokens_generated": 0,
                "models_used": [],
                "failed": True,
                "error": str(e)
            }
        )