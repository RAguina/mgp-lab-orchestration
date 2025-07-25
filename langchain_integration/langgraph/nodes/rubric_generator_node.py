# ========================================
# 3. RUBRIC GENERATOR NODE - EVOLVED VERSION
# ========================================

# langchain_integration/langgraph/nodes/rubric_generator_node.py
import time
import logging
from typing import Dict, Any, List
from langchain_integration.langgraph.agent_state import AgentState
from langchain_integration.langgraph.local_llm_node import build_local_llm_tool_node

# Setup logger para este nodo
logger = logging.getLogger("rubric_generator")

def rubric_generator_node(state: AgentState) -> AgentState:
    """
    Worker especializado en generación de rúbricas de evaluación contextual
    """
    start_time = time.time()
    node_id = f"rubric_generator_{int(start_time)}"
    
    logger.info(f"[{node_id}] === RUBRIC GENERATOR WORKER STARTED ===")
    print("[RUBRIC] Generando rúbricas de evaluación...")
    
    messages = state.get("messages", [])
    rubric_result = ""
    rubric_metadata = {}
    
    try:
        # Obtener contexto para las rúbricas
        task_type = state.get("task_type", "unknown")
        user_input = state.get("input", "")
        model_output = state.get("output", "")
        
        if not model_output.strip():
            logger.warning(f"[{node_id}] No output available for rubric generation")
            messages.append("[RUBRIC] No hay output para evaluar")
            rubric_result = "No se puede generar rúbrica: sin output para evaluar"
        else:
            logger.info(f"[{node_id}] Generating rubrics for task_type: {task_type}")
            logger.info(f"[{node_id}] Input: {len(user_input)} chars, Output: {len(model_output)} chars")
            
            # Rúbricas específicas por tipo de tarea
            rubric_templates = {
                "code": [
                    "FUNCIONALIDAD: ¿El código ejecuta correctamente?",
                    "EFICIENCIA: ¿Es algorítmicamente eficiente?", 
                    "LEGIBILIDAD: ¿Es fácil de entender y mantener?",
                    "BUENAS PRÁCTICAS: ¿Sigue convenciones del lenguaje?",
                    "COMPLETITUD: ¿Resuelve completamente el problema?"
                ],
                "technical": [
                    "PRECISIÓN: ¿La información es técnicamente correcta?",
                    "CLARIDAD: ¿Es comprensible para la audiencia?",
                    "PROFUNDIDAD: ¿Cubre aspectos importantes del tema?",
                    "EJEMPLOS: ¿Incluye casos prácticos útiles?",
                    "ESTRUCTURA: ¿Está bien organizada la respuesta?"
                ],
                "creative": [
                    "ORIGINALIDAD: ¿Es creativa e innovadora?",
                    "COHERENCIA: ¿Mantiene consistencia narrativa?",
                    "ENGAGEMENT: ¿Es interesante y atractiva?",
                    "COMPLETITUD: ¿Cumple con lo solicitado?",
                    "ESTILO: ¿Tiene buen uso del lenguaje?"
                ]
            }
            
            # Seleccionar template base
            base_rubrics = rubric_templates.get(task_type, rubric_templates["technical"])
            
            # Construir LLM tool
            llm_start = time.time()
            llm = build_local_llm_tool_node(
                model_key=state.get("selected_model", "mistral7b"),
                strategy=state.get("strategy", "optimized"),
                max_tokens=768  # Más tokens para rúbricas detalladas
            )
            llm_build_time = time.time() - llm_start
            
            # Prompt contextual para rúbricas
            prompt = f"""Genera rúbricas de evaluación específicas y detalladas para evaluar la siguiente respuesta.

CONTEXTO:
- Tipo de tarea: {task_type}
- Pregunta original: {user_input}
- Longitud de respuesta: {len(model_output)} caracteres

RESPUESTA A EVALUAR:
{model_output[:1500]}{"..." if len(model_output) > 1500 else ""}

INSTRUCCIONES:
1. Crea 5-7 criterios de evaluación específicos para este contexto
2. Cada criterio debe tener: nombre, descripción, y escala 1-10
3. Incluye criterios base: {', '.join(base_rubrics[:3])}
4. Agrega criterios específicos para este caso particular
5. Usa formato claro y estructurado

FORMATO:
## RÚBRICA DE EVALUACIÓN

### CRITERIO 1: [NOMBRE]
- **Descripción**: [qué evalúa específicamente]
- **Escala**: 1 (deficiente) - 10 (excelente)
- **Indicadores**: [qué buscar para puntuar alto]

[continuar con otros criterios...]"""

            # Ejecutar generación
            inference_start = time.time()
            rubric_result = llm.invoke(prompt)
            inference_time = time.time() - inference_start
            
            # Metadata de la generación
            rubric_metadata = {
                "llm_build_time": llm_build_time,
                "inference_time": inference_time,
                "task_type": task_type,
                "input_length": len(user_input),
                "output_evaluated_length": len(model_output),
                "rubric_length": len(rubric_result),
                "base_template": task_type,
                "criteria_count": len(base_rubrics),
                "model_used": state.get("selected_model", "unknown")
            }
            
            messages.append(f"[RUBRIC] Rúbricas generadas para tarea tipo '{task_type}'")
            logger.info(f"[{node_id}] Rubric generation completed in {inference_time:.2f}s")
            
    except Exception as e:
        logger.error(f"[{node_id}] Error generating rubrics: {str(e)}")
        messages.append(f"[RUBRIC] Error al generar rúbricas: {str(e)}")
        rubric_result = f"Error en generación de rúbricas: {str(e)}"
        rubric_metadata = {"error": str(e)}
    
    # Resultado
    end_time = time.time()
    total_time = end_time - start_time
    
    logger.info(f"[{node_id}] === RUBRIC GENERATOR COMPLETED ===")
    logger.info(f"[{node_id}] Total processing time: {total_time:.3f}s")
    logger.info(f"[{node_id}] Rubric result length: {len(rubric_result)} chars")
    
    print(f"[RUBRIC] Completado: {len(rubric_result)} caracteres en {total_time:.3f}s")
    
    return {
        **state,
        "analysis_result": rubric_result,
        "rubric_metadata": rubric_metadata,
        "messages": messages
    }