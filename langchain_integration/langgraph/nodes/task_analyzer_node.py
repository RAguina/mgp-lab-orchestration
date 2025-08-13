# langchain_integration/langgraph/nodes/task_analyzer_node.py
import time
import logging
from typing import Dict, Any
from langchain_integration.tools.lab_tools import ModelSelectorTool

logger = logging.getLogger("task_analyzer")

def task_analyzer_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Nodo de análisis de tareas.
    Clasifica el tipo de tarea, selecciona modelo y estrategia,
    y registra métricas + metadatos consistentes con execution_node.
    """
    start = time.time()
    node_id = f"task_analyzer_{int(start)}"

    messages = list(state.get("messages", []))
    prompt = state.get("input", "")
    input_text = prompt.lower()

    logger.info(f"[{node_id}] analyzer start len={len(prompt)}")

    # --- Clasificación por keywords ---
    task_keywords = {
        "code": ["código", "code", "función", "script", "programar", "python", "javascript"],
        "technical": ["explica", "qué es", "define", "cómo funciona", "technical"],
        "creative": ["historia", "cuento", "poema", "creativo", "imagina"],
        "analysis": ["analiza", "examina", "evalúa", "compara", "ventajas", "desventajas"],
    }
    detected_keywords = []
    task_type = "chat"

    for task, keywords in task_keywords.items():
        found_keywords = [kw for kw in keywords if kw in input_text]
        if found_keywords:
            detected_keywords.extend(found_keywords)
            task_type = task
            break

    messages.append(f"[ANALYZER] Tipo de tarea detectada: {task_type}")
    messages.append(f"[ANALYZER] Keywords encontradas: {detected_keywords}")

    # --- Selección de modelo ---
    selected_model = "mistral7b"
    selection_result = ""
    try:
        selector = ModelSelectorTool()
        selection_result = selector.run(prompt)
        if "Modelo seleccionado:" in selection_result:
            selected_model = selection_result.split("Modelo seleccionado:")[1].split("\n")[0].strip()
    except Exception as e:
        logger.error(f"[{node_id}] Model selector failed: {e}")
        selection_result = f"Error en selección: {str(e)}"

    messages.append(f"[ANALYZER] Modelo seleccionado: {selected_model}")

    # --- Estrategia ---
    strategy_mapping = {
        "code": "optimized",
        "technical": "standard",
        "creative": "standard",
        "analysis": "optimized",
        "chat": "optimized"
    }
    suggested_strategy = strategy_mapping.get(task_type, "optimized")

    # --- Metadata y métricas ---
    analysis_metadata = {
        "request_timestamp": start,
        "completion_timestamp": time.time(),
        "duration_seconds": time.time() - start,
        "prompt_length": len(prompt),
        "detected_keywords": detected_keywords,
        "model_selection_result": selection_result,
        "suggested_strategy": suggested_strategy,
        "node_id": node_id
    }

    analysis_metrics = {
        "keywords_found": len(detected_keywords) > 0,
        "model_selection_successful": "Error" not in selection_result,
        "strategy_mapping_applied": True
    }

    logger.info(f"[{node_id}] analyzer complete task={task_type} model={selected_model} strategy={suggested_strategy}")

    return {
        **state,
        "task_type": task_type,
        "selected_model": selected_model,
        "strategy": suggested_strategy,
        "messages": messages,
        "analysis_metrics": analysis_metrics,
        "analysis_metadata": analysis_metadata
    }
