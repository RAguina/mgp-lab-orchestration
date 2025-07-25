# ========================================
# 4. SUMMARY NODE - EVOLVED VERSION
# ========================================

# langchain_integration/langgraph/nodes/summary_node.py
import time
import logging
from typing import Dict, Any
from langchain_integration.langgraph.agent_state import AgentState

# Setup logger para este nodo
logger = logging.getLogger("summary_generator")

def summary_node(state: AgentState) -> AgentState:
    """
    Worker especializado en generación de resúmenes ejecutivos con métricas completas
    """
    start_time = time.time()
    node_id = f"summary_generator_{int(start_time)}"
    
    logger.info(f"[{node_id}] === SUMMARY GENERATOR WORKER STARTED ===")
    print("[SUMMARY] Generando resumen ejecutivo del proceso...")
    
    try:
        # Recopilar todas las métricas disponibles
        task_type = state.get('task_type', 'unknown')
        selected_model = state.get('selected_model', 'unknown')
        strategy = state.get('strategy', 'unknown')
        output_length = len(state.get('output', ''))
        
        # Métricas de ejecución si están disponibles
        execution_metrics = state.get('execution_metrics', {})
        analysis_metadata = state.get('analysis_metadata', {})
        history_metadata = state.get('history_metadata', {})
        validation_metadata = state.get('validation_metadata', {})
        
        # Construir summary parts con más detalle
        summary_parts = []
        
        # Información básica
        summary_parts.append(f"Tarea: {task_type}")
        summary_parts.append(f"Modelo: {selected_model}")
        summary_parts.append(f"Estrategia: {strategy}")
        summary_parts.append(f"Output: {output_length} caracteres")
        
        # Métricas de timing si disponibles
        if execution_metrics:
            total_time = execution_metrics.get('total_time', 0)
            load_time = execution_metrics.get('load_time', 0)
            inference_time = execution_metrics.get('inference_time', 0)
            cache_hit = execution_metrics.get('cache_hit', False)
            
            summary_parts.append(f"Tiempo total: {total_time:.1f}s")
            if load_time > 0:
                summary_parts.append(f"Carga: {load_time:.1f}s")
            if inference_time > 0:
                summary_parts.append(f"Inferencia: {inference_time:.1f}s")
            summary_parts.append(f"Cache: {'HIT' if cache_hit else 'MISS'}")
        
        # Métricas de calidad si disponibles
        if validation_metadata:
            quality_score = validation_metadata.get('validation_results', {}).get('overall_score', 0)
            if quality_score > 0:
                summary_parts.append(f"Calidad: {quality_score}/10")
        
        # Información de VRAM
        vram_status = state.get("vram_status", "")
        if vram_status and "VRAM Usada:" in vram_status:
            try:
                vram_used = vram_status.split("VRAM Usada:")[1].split("GB")[0].strip()
                summary_parts.append(f"VRAM: {vram_used}GB")
            except:
                pass
        
        # Mensajes del proceso
        messages = state.get("messages", [])
        summary_parts.append(f"Mensajes: {len(messages)}")
        
        # Crear resumen final
        final_summary = " | ".join(summary_parts)
        
        # Summary extendido con detalles
        extended_summary = {
            "execution_summary": final_summary,
            "task_details": {
                "type": task_type,
                "model": selected_model,
                "strategy": strategy,
                "output_chars": output_length
            },
            "performance_metrics": execution_metrics,
            "quality_metrics": validation_metadata,
            "process_messages": len(messages),
            "total_processing_time": time.time() - start_time
        }
        
        # Log del resumen
        logger.info(f"[{node_id}] Summary generated: {final_summary}")
        logger.info(f"[{node_id}] Extended metrics: {len(extended_summary)} fields")
        
    except Exception as e:
        logger.error(f"[{node_id}] Error generating summary: {str(e)}")
        final_summary = f"Error en resumen: {str(e)}"
        extended_summary = {"error": str(e)}
    
    # Resultado
    end_time = time.time()
    total_time = end_time - start_time
    
    logger.info(f"[{node_id}] === SUMMARY GENERATOR COMPLETED ===")
    logger.info(f"[{node_id}] Total processing time: {total_time:.3f}s")
    
    print(f"[SUMMARY] Completado: resumen ejecutivo en {total_time:.3f}s")
    
    return {
        **state,
        "final_summary": final_summary,
        "extended_summary": extended_summary
    }