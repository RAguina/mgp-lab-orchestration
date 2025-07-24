# langchain_integration/langgraph/nodes/task_analyzer_node.py

import time
import logging
from typing import Dict, Any
from langchain_integration.tools.lab_tools import ModelSelectorTool

# Configurar logger específico para este nodo
logger = logging.getLogger("task_analyzer")

def task_analyzer_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Worker especializado en análisis de tareas con logging detallado.
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

    # Análisis de tipo de tarea con logging detallado
    task_keywords = {
        "code": ["código", "code", "función", "script", "programar", "python", "javascript"],
        "technical": ["explica", "qué es", "define", "cómo funciona", "technical"],
        "creative": ["historia", "cuento", "poema", "creativo", "imagina"],
        "analysis": ["analiza", "examina", "evalúa", "compara", "ventajas", "desventajas"],
    }
    
    detected_keywords = []
    task_type = "chat"  # Default
    
    for task, keywords in task_keywords.items():
        found_keywords = [kw for kw in keywords if kw in input_text]
        if found_keywords:
            detected_keywords.extend(found_keywords)
            task_type = task
            logger.info(f"[{node_id}] Task type '{task}' detected by keywords: {found_keywords}")
            break
    
    if not detected_keywords:
        logger.info(f"[{node_id}] No specific keywords found, defaulting to 'chat'")
    
    messages.append(f"[ANALYZER] Tipo de tarea detectada: {task_type}")
    messages.append(f"[ANALYZER] Keywords encontradas: {detected_keywords}")

    # Selección de modelo con logging
    logger.info(f"[{node_id}] Starting model selection for task type: {task_type}")
    
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
            except Exception as parse_error:
                logger.warning(f"[{node_id}] Failed to parse model selection, using default: {parse_error}")
        
    except Exception as selector_error:
        logger.error(f"[{node_id}] Model selector failed: {selector_error}")
        selected_model = "mistral7b"
        selection_result = f"Error in selection: {str(selector_error)}"

    messages.append(f"[ANALYZER] Modelo seleccionado: {selected_model}")

    # Determinar estrategia basada en tipo de tarea
    strategy_mapping = {
        "code": "optimized",      # Código necesita precisión
        "technical": "standard",  # Explicaciones técnicas necesitan más contexto
        "creative": "standard",   # Creatividad necesita más libertad
        "analysis": "optimized",  # Análisis necesita eficiencia
        "chat": "optimized"       # Chat general optimizado
    }
    
    suggested_strategy = strategy_mapping.get(task_type, "optimized")
    logger.info(f"[{node_id}] Suggested strategy for {task_type}: {suggested_strategy}")
    
    # Preparar resultado
    end_time = time.time()
    total_time = end_time - start_time
    
    result_state = {
        **state,
        "task_type": task_type,
        "selected_model": selected_model,
        "strategy": suggested_strategy,  # Agregar estrategia sugerida
        "messages": messages,
        # Agregar metadata de análisis
        "analysis_metadata": {
            "node_id": node_id,
            "detected_keywords": detected_keywords,
            "processing_time": total_time,
            "model_selection_result": selection_result,
            "original_prompt": original_prompt,
            "suggested_strategy": suggested_strategy
        }
    }
    
    logger.info(f"[{node_id}] === TASK ANALYZER COMPLETED ===")
    logger.info(f"[{node_id}] Total processing time: {total_time:.2f}s")
    logger.info(f"[{node_id}] Final decision: task={task_type}, model={selected_model}, strategy={suggested_strategy}")
    
    print(f"[ANALYZE] Completado: {task_type} → {selected_model} ({suggested_strategy})")
    
    return result_state