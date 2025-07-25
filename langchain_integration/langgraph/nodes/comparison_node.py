# ========================================
# 2. COMPARISON NODE - EVOLVED VERSION  
# ========================================

# langchain_integration/langgraph/nodes/comparison_node.py
import time
import logging
from typing import Dict, Any
from langchain_integration.langgraph.agent_state import AgentState
from langchain_integration.langgraph.local_llm_node import build_local_llm_tool_node

# Setup logger para este nodo
logger = logging.getLogger("comparison_worker")

def comparison_node(state: AgentState) -> AgentState:
    """
    Worker especializado en comparación de respuestas con análisis detallado
    """
    start_time = time.time()
    node_id = f"comparison_worker_{int(start_time)}"
    
    logger.info(f"[{node_id}] === COMPARISON WORKER STARTED ===")
    print("[COMPARE] Comparando respuestas...")
    
    messages = state.get("messages", [])
    comparison_result = ""
    comparison_metadata = {}
    
    try:
        # Verificar que tenemos respuestas para comparar
        output_a = state.get("output_a", "").strip()
        output_b = state.get("output_b", "").strip()
        
        if not output_a and not output_b:
            # Usar output actual vs historial como fallback
            output_a = state.get("output", "").strip()
            output_b = state.get("last_output", "").strip()
            logger.info(f"[{node_id}] Using current output vs history for comparison")
        
        if not output_a or not output_b:
            logger.warning(f"[{node_id}] Insufficient outputs for comparison")
            messages.append("[COMPARE] No hay suficientes respuestas para comparar")
            comparison_result = "Comparación no posible: respuestas insuficientes"
        else:
            logger.info(f"[{node_id}] Comparing outputs: A={len(output_a)} chars, B={len(output_b)} chars")
            
            # Construir LLM tool
            llm_start = time.time()
            llm = build_local_llm_tool_node(
                model_key=state.get("selected_model", "mistral7b"),
                strategy=state.get("strategy", "optimized"),
                max_tokens=512
            )
            llm_build_time = time.time() - llm_start
            
            # Prompt mejorado para comparación
            prompt = f"""Analiza y compara estas dos respuestas a la pregunta dada. Evalúa objetivamente usando estos criterios:

1. CLARIDAD: ¿Cuál es más fácil de entender?
2. CORRECCIÓN: ¿Cuál es técnicamente más precisa?
3. COMPLETITUD: ¿Cuál responde mejor la pregunta?
4. UTILIDAD: ¿Cuál es más práctica/aplicable?

❓ Pregunta Original:
{state.get("input", "No especificada")}

🅰️ RESPUESTA A ({len(output_a)} caracteres):
{output_a[:1000]}{"..." if len(output_a) > 1000 else ""}

🅱️ RESPUESTA B ({len(output_b)} caracteres):
{output_b[:1000]}{"..." if len(output_b) > 1000 else ""}

Formato de respuesta:
GANADORA: [A/B/EMPATE]
RAZÓN: [explicación breve]
PUNTAJES: A=[1-10] B=[1-10]
RECOMENDACIÓN: [qué hacer con este análisis]"""

            # Ejecutar comparación
            inference_start = time.time()
            comparison_result = llm.invoke(prompt)
            inference_time = time.time() - inference_start
            
            # Metadata de la comparación
            comparison_metadata = {
                "llm_build_time": llm_build_time,
                "inference_time": inference_time,
                "input_length_a": len(output_a),
                "input_length_b": len(output_b),
                "output_length": len(comparison_result),
                "model_used": state.get("selected_model", "unknown"),
                "strategy_used": state.get("strategy", "unknown")
            }
            
            messages.append(f"[COMPARE] Comparación completada usando {state.get('selected_model', 'modelo')}")
            logger.info(f"[{node_id}] Comparison completed in {inference_time:.2f}s")
            
    except Exception as e:
        logger.error(f"[{node_id}] Error in comparison: {str(e)}")
        messages.append(f"[COMPARE] Error en comparación: {str(e)}")
        comparison_result = f"Error en comparación: {str(e)}"
        comparison_metadata = {"error": str(e)}
    
    # Resultado
    end_time = time.time()
    total_time = end_time - start_time
    
    logger.info(f"[{node_id}] === COMPARISON COMPLETED ===")
    logger.info(f"[{node_id}] Total processing time: {total_time:.3f}s")
    logger.info(f"[{node_id}] Comparison result length: {len(comparison_result)} chars")
    
    print(f"[COMPARE] Completado: {len(comparison_result)} caracteres en {total_time:.3f}s")
    
    return {
        **state,
        "comparison_result": comparison_result,
        "comparison_metadata": comparison_metadata,
        "messages": messages
    }