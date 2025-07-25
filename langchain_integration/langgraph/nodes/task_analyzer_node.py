# langchain_integration/langgraph/nodes/task_analyzer_node.py - EVOLVED VERSION

import time
import logging
from typing import Dict, Any
from langchain_integration.tools.lab_tools import ModelSelectorTool

# ✅ NUEVO: Import del sistema de observabilidad (opcional)
try:
    from utils.langgraph_logger import langgraph_thinking_logger, langgraph_decision_logger
    OBSERVABILITY_AVAILABLE = True
except ImportError:
    OBSERVABILITY_AVAILABLE = False

# Configurar logger específico para este nodo (mantener existente)
logger = logging.getLogger("task_analyzer")

def task_analyzer_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Worker especializado en análisis de tareas con logging detallado y observabilidad opcional.
    """
    start_time = time.time()
    node_id = f"task_analyzer_{int(start_time)}"
    
    logger.info(f"[{node_id}] === TASK ANALYZER WORKER STARTED ===")
    logger.info(f"[{node_id}] Input prompt: '{state.get('input', '')[:100]}...'")
    
    print("[ANALYZE] Analizando tipo de tarea...")

    input_text = state.get("input", "").lower()
    original_prompt = state.get("input", "")
    messages = state.get("messages", [])
    
    logger.info(f"[{node_id}] Processing text length: {len(input_text)} chars")

    # ✅ NUEVO: Setup observabilidad opcional
    thinking_log = None
    decision_log = None
    if OBSERVABILITY_AVAILABLE and 'langgraph_logger' in state:
        thinking_log = langgraph_thinking_logger(node_id, state)
        decision_log = langgraph_decision_logger(node_id, state)
        
        thinking_log(
            step="analysis_start",
            prompt_length=len(input_text),
            prompt_preview=original_prompt[:100]
        )

    # Análisis de tipo de tarea con logging detallado (MANTENER LÓGICA EXISTENTE)
    task_keywords = {
        "code": ["código", "code", "función", "script", "programar", "python", "javascript"],
        "technical": ["explica", "qué es", "define", "cómo funciona", "technical"],
        "creative": ["historia", "cuento", "poema", "creativo", "imagina"],
        "analysis": ["analiza", "examina", "evalúa", "compara", "ventajas", "desventajas"],
    }
    
    detected_keywords = []
    task_type = "chat"  # Default
    
    # ✅ NUEVO: Log proceso de clasificación
    if thinking_log:
        thinking_log(
            step="keyword_analysis", 
            available_categories=list(task_keywords.keys()),
            scanning_text_length=len(input_text)
        )
    
    for task, keywords in task_keywords.items():
        found_keywords = [kw for kw in keywords if kw in input_text]
        if found_keywords:
            detected_keywords.extend(found_keywords)
            task_type = task
            logger.info(f"[{node_id}] Task type '{task}' detected by keywords: {found_keywords}")
            
            # ✅ NUEVO: Log cuando se encuentra un match
            if thinking_log:
                thinking_log(
                    step="keyword_match_found",
                    matched_category=task,
                    found_keywords=found_keywords,
                    confidence="high"
                )
            break
    
    if not detected_keywords:
        logger.info(f"[{node_id}] No specific keywords found, defaulting to 'chat'")
        if thinking_log:
            thinking_log(
                step="default_classification",
                reason="no_keywords_found",
                default_type="chat"
            )
    
    messages.append(f"[ANALYZER] Tipo de tarea detectada: {task_type}")
    messages.append(f"[ANALYZER] Keywords encontradas: {detected_keywords}")

    # Selección de modelo con logging (MANTENER LÓGICA EXISTENTE)
    logger.info(f"[{node_id}] Starting model selection for task type: {task_type}")
    
    # ✅ NUEVO: Log inicio de selección de modelo
    if thinking_log:
        thinking_log(
            step="model_selection_start",
            task_type=task_type,
            using_selector_tool=True
        )
    
    try:
        selector = ModelSelectorTool()
        selection_start = time.time()
        selection_result = selector.run(original_prompt)
        selection_time = time.time() - selection_start
        
        logger.info(f"[{node_id}] Model selection completed in {selection_time:.2f}s")
        logger.info(f"[{node_id}] Selection result: {selection_result[:200]}...")
        
        selected_model = "mistral7b"  # Default fallback
        
        if "Modelo seleccionado:" in selection_result:
            try:
                selected_model = selection_result.split("Modelo seleccionado:")[1].split("\n")[0].strip()
                logger.info(f"[{node_id}] Parsed selected model: {selected_model}")
                
                # ✅ NUEVO: Log selección exitosa
                if thinking_log:
                    thinking_log(
                        step="model_selection_success",
                        selected_model=selected_model,
                        selection_time_seconds=selection_time,
                        selector_confidence="parsed_successfully"
                    )
                    
            except Exception as parse_error:
                logger.warning(f"[{node_id}] Failed to parse model selection, using default: {parse_error}")
                if thinking_log:
                    thinking_log(
                        step="model_selection_parse_error",
                        error=str(parse_error),
                        fallback_model=selected_model
                    )
        
    except Exception as selector_error:
        logger.error(f"[{node_id}] Model selector failed: {selector_error}")
        selected_model = "mistral7b"
        selection_result = f"Error in selection: {str(selector_error)}"
        
        # ✅ NUEVO: Log error en selección
        if thinking_log:
            thinking_log(
                step="model_selection_error",
                error=str(selector_error),
                fallback_model=selected_model
            )

    messages.append(f"[ANALYZER] Modelo seleccionado: {selected_model}")

    # Determinar estrategia basada en tipo de tarea (MANTENER LÓGICA EXISTENTE)
    strategy_mapping = {
        "code": "optimized",      # Código necesita precisión
        "technical": "standard",  # Explicaciones técnicas necesitan más contexto
        "creative": "standard",   # Creatividad necesita más libertad
        "analysis": "optimized",  # Análisis necesita eficiencia
        "chat": "optimized"       # Chat general optimizado
    }
    
    suggested_strategy = strategy_mapping.get(task_type, "optimized")
    logger.info(f"[{node_id}] Suggested strategy for {task_type}: {suggested_strategy}")
    
    # ✅ NUEVO: Log decisión final
    if decision_log:
        decision_log(
            decision_type="task_analysis_complete",
            final_classification={
                "task_type": task_type,
                "selected_model": selected_model,
                "strategy": suggested_strategy,
                "confidence_indicators": {
                    "keywords_found": len(detected_keywords) > 0,
                    "model_selection_successful": "Error" not in selection_result,
                    "strategy_mapping_applied": True
                }
            },
            processing_pipeline={
                "keyword_analysis": "completed",
                "model_selection": "completed",
                "strategy_mapping": "completed"
            },
            reasoning=f"Classified as '{task_type}' based on {detected_keywords if detected_keywords else 'default rules'}"
        )
    
    # Preparar resultado (MANTENER ESTRUCTURA EXISTENTE)
    end_time = time.time()
    total_time = end_time - start_time
    
    # ✅ NUEVO: Agregar métricas de observabilidad si están disponibles
    analysis_metadata = {
        "node_id": node_id,
        "detected_keywords": detected_keywords,
        "processing_time": total_time,
        "model_selection_result": selection_result,
        "original_prompt": original_prompt,
        "suggested_strategy": suggested_strategy
    }
    
    # ✅ NUEVO: Agregar métricas adicionales si hay observabilidad
    if OBSERVABILITY_AVAILABLE and 'langgraph_logger' in state:
        analysis_metadata.update({
            "observability_enabled": True,
            "thinking_steps_logged": True,
            "decisions_logged": True,
            "node_execution_id": node_id
        })
    
    result_state = {
        **state,
        "task_type": task_type,
        "selected_model": selected_model,
        "strategy": suggested_strategy,
        "messages": messages,
        "analysis_metadata": analysis_metadata
    }
    
    logger.info(f"[{node_id}] === TASK ANALYZER COMPLETED ===")
    logger.info(f"[{node_id}] Total processing time: {total_time:.2f}s")
    logger.info(f"[{node_id}] Final decision: task={task_type}, model={selected_model}, strategy={suggested_strategy}")
    
    print(f"[ANALYZE] Completado: {task_type} → {selected_model} ({suggested_strategy})")
    
    return result_state

# ✅ NUEVO: Función wrapper con decorator para observabilidad completa (OPCIONAL)
def create_observable_task_analyzer():
    """
    Crea una versión del task analyzer con observabilidad completa usando decorator
    Solo usar si se quiere migrar completamente al nuevo sistema
    """
    if not OBSERVABILITY_AVAILABLE:
        return task_analyzer_node
    
    from utils.langgraph_logger import langgraph_node_logger
    
    @langgraph_node_logger("task_analyzer", enable_detailed_logging=True)
    def observable_task_analyzer(state: Dict[str, Any]) -> Dict[str, Any]:
        return task_analyzer_node(state)
    
    return observable_task_analyzer

# ✅ Para uso futuro - migración gradual
# Si quieres usar la versión con decorator completo:
# task_analyzer_node = create_observable_task_analyzer()