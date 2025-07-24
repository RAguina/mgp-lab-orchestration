# langchain_integration/langgraph/nodes/execution_node.py

import sys
import os
import time
import logging
from typing import Dict, Any

# Agregar path del lab root para imports
LAB_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, LAB_ROOT)

from langchain_integration.langgraph.agent_state import AgentState
from local_models.model_executor import ModelExecutor

# Configurar logger específico para este nodo
logger = logging.getLogger("execution_worker")

# Instancia global del executor para reutilizar
_executor = None

def get_executor():
    """Obtiene la instancia global del ModelExecutor"""
    global _executor
    if _executor is None:
        _executor = ModelExecutor(save_results=True)
        logger.info("ModelExecutor instance created and cached")
    return _executor

def execution_node(state: AgentState) -> AgentState:
    """
    Worker especializado en ejecución de modelos con logging detallado y cache inteligente.
    """
    start_time = time.time()
    node_id = f"execution_worker_{int(start_time)}"
    
    logger.info(f"[{node_id}] === EXECUTION WORKER STARTED ===")
    
    # Extraer parámetros del estado
    model_key = state.get("selected_model", "mistral7b")
    strategy = state.get("strategy", "optimized")
    original_prompt = state.get("input", "")
    analysis_metadata = state.get("analysis_metadata", {})
    
    logger.info(f"[{node_id}] Model: {model_key}")
    logger.info(f"[{node_id}] Strategy: {strategy}")
    logger.info(f"[{node_id}] Prompt length: {len(original_prompt)} chars")
    logger.info(f"[{node_id}] Task type: {state.get('task_type', 'unknown')}")
    
    print(f"[EXECUTE] Ejecutando {model_key} con estrategia {strategy} [CACHE-AWARE]...")
    messages = state.get("messages", [])

    try:
        # Preparar executor
        executor_start = time.time()
        executor = get_executor()
        executor_prep_time = time.time() - executor_start
        
        logger.info(f"[{node_id}] Executor prepared in {executor_prep_time:.3f}s")
        
        # Log del prompt que va al modelo (puede ser diferente al original)
        logger.info(f"[{node_id}] Original prompt: '{original_prompt[:100]}...'")
        
        # Determinar max_tokens basado en tipo de tarea
        task_type = state.get("task_type", "chat")
        max_tokens_mapping = {
            "code": 512,       # Código puede ser más largo
            "technical": 384,  # Explicaciones técnicas necesitan espacio
            "creative": 256,   # Creatividad más concisa
            "analysis": 384,   # Análisis detallado
            "chat": 256        # Chat general más breve
        }
        
        max_tokens = max_tokens_mapping.get(task_type, 256)
        logger.info(f"[{node_id}] Max tokens for {task_type}: {max_tokens}")
        
        # Ejecutar modelo con logging detallado
        execution_start = time.time()
        
        result = executor.execute(
            model_key=model_key,
            prompt=original_prompt,
            max_tokens=max_tokens,
            strategy=strategy,
            temperature=0.7
        )
        
        execution_end = time.time()
        total_execution_time = execution_end - execution_start
        
        logger.info(f"[{node_id}] Model execution completed in {total_execution_time:.2f}s")
        
        if result["success"]:
            output = result["output"]
            metrics = result["metrics"]
            
            # Log información detallada del cache
            cache_status = "HIT" if metrics.get("cache_hit") else "MISS"
            load_time = metrics.get("load_time_sec", 0)
            infer_time = metrics.get("inference_time_sec", 0)
            tokens_generated = metrics.get("tokens_generated", 0)
            
            logger.info(f"[{node_id}] === EXECUTION SUCCESSFUL ===")
            logger.info(f"[{node_id}] Cache status: {cache_status}")
            logger.info(f"[{node_id}] Load time: {load_time:.2f}s")
            logger.info(f"[{node_id}] Inference time: {infer_time:.2f}s")
            logger.info(f"[{node_id}] Tokens generated: {tokens_generated}")
            logger.info(f"[{node_id}] Output length: {len(output)} chars")
            logger.info(f"[{node_id}] Output preview: '{output[:150]}...'")
            
            # Detectar calidad del output
            quality_score = analyze_output_quality(output, task_type)
            logger.info(f"[{node_id}] Output quality score: {quality_score}/10")
            
            messages.append(f"[EXECUTE] Generación completada: {len(output)} caracteres")
            messages.append(f"[EXECUTE] Cache: {cache_status} | Load: {load_time:.1f}s | Inference: {infer_time:.1f}s")
            messages.append(f"[EXECUTE] Calidad estimada: {quality_score}/10")
            
            return {
                **state,
                "output": output,
                "analysis_result": output,
                "messages": messages,
                # Métricas extendidas para otros nodos
                "execution_metrics": {
                    "node_id": node_id,
                    "cache_hit": metrics.get("cache_hit", False),
                    "load_time": load_time,
                    "inference_time": infer_time,
                    "total_time": metrics.get("total_time_sec", 0),
                    "tokens_generated": tokens_generated,
                    "max_tokens_requested": max_tokens,
                    "quality_score": quality_score,
                    "model_used": model_key,
                    "strategy_used": strategy,
                    "execution_worker_time": total_execution_time
                }
            }
        else:
            error_msg = result.get("error", "Unknown error")
            logger.error(f"[{node_id}] === EXECUTION FAILED ===")
            logger.error(f"[{node_id}] Error: {error_msg}")
            
            messages.append(f"[EXECUTE] ❌ Error en ejecución: {error_msg}")
            
            return {
                **state,
                "output": f"Error al ejecutar el modelo: {error_msg}",
                "messages": messages,
                "execution_metrics": {
                    "node_id": node_id,
                    "cache_hit": False, 
                    "load_time": 0, 
                    "inference_time": 0,
                    "error": error_msg,
                    "failed": True
                }
            }
            
    except Exception as e:
        end_time = time.time()
        total_time = end_time - start_time
        
        logger.error(f"[{node_id}] === EXECUTION EXCEPTION ===")
        logger.error(f"[{node_id}] Exception: {str(e)}")
        logger.error(f"[{node_id}] Total time before failure: {total_time:.2f}s")
        
        messages.append(f"[EXECUTE] 💥 Excepción en ejecución: {str(e)}")
        
        return {
            **state,
            "output": f"Excepción al ejecutar el modelo: {str(e)}",
            "messages": messages,
            "execution_metrics": {
                "node_id": node_id,
                "cache_hit": False, 
                "load_time": 0, 
                "inference_time": 0,
                "exception": str(e),
                "failed": True,
                "execution_worker_time": total_time
            }
        }

def analyze_output_quality(output: str, task_type: str) -> int:
    """
    Analiza la calidad del output basado en el tipo de tarea.
    Retorna un score de 1-10.
    """
    if not output or len(output.strip()) < 10:
        return 1
    
    score = 5  # Base score
    
    # Ajustar según tipo de tarea
    if task_type == "code":
        if any(keyword in output.lower() for keyword in ["def ", "function", "class", "import"]):
            score += 2
        if "```" in output:
            score += 1
    elif task_type == "technical":
        if len(output) > 100:
            score += 1
        if any(keyword in output.lower() for keyword in ["porque", "debido", "significa"]):
            score += 1
    elif task_type == "creative":
        if len(output) > 50:
            score += 1
        if any(keyword in output.lower() for keyword in ["historia", "personaje", "escena"]):
            score += 1
    
    # Penalizar errores obvios
    if any(error in output.lower() for error in ["error", "exception", "traceback"]):
        score -= 3
    
    # Penalizar outputs muy cortos o muy largos
    if len(output) < 20:
        score -= 2
    elif len(output) > 2000:
        score -= 1
    
    return max(1, min(10, score))

# Función de compatibilidad si otros nodos la necesitan
def execution_node_legacy(state: AgentState) -> AgentState:
    """Función legacy para compatibilidad con código viejo"""
    return execution_node(state)