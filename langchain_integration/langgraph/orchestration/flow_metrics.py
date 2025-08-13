"""
Flow Metrics - Construcción de métricas y flow data para el frontend
Extraído de routing_agent.py para mejor separación de responsabilidades
"""

import logging
from typing import Dict, Any, List

# Logger específico para flow metrics
flow_logger = logging.getLogger("orchestrator.flow_metrics")

def build_flow_nodes(full_state: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Construye la lista de nodos del flow basado en el estado de ejecución.
    
    Args:
        full_state: Estado completo después de la ejecución
    
    Returns:
        Lista de nodos para visualización en frontend
    """
    flow_nodes = []
    
    # Métricas de ejecución
    em = full_state.get("execution_metrics", {}) or {}
    analysis_md = full_state.get("analysis_metadata", {}) or {}
    validation_md = full_state.get("validation_metadata", {}) or {}
    model_used = full_state.get("selected_model", "mistral7b")
    
    # Task Analyzer Node
    if full_state.get("task_type"):
        flow_nodes.append({
            "id": "task_analyzer",
            "name": "Task Analyzer",
            "type": "task_analyzer",
            "status": "completed",
            "start_time": 0,
            "end_time": analysis_md.get("processing_time", 0.5),
            "output": f"Detected: {full_state.get('task_type')} → {model_used}"
        })
    
    # Resource Monitor Node
    if full_state.get("vram_status"):
        flow_nodes.append({
            "id": "resource_monitor", 
            "name": "Resource Monitor",
            "type": "resource_monitor",
            "status": "completed",
            "start_time": 0.5,
            "end_time": 1.0,
            "output": f"VRAM: {full_state.get('vram_status')}"
        })
    
    # Model Execution Node
    status = "completed" if not em.get("failed") else "error"
    flow_nodes.append({
        "id": "model_execution",
        "name": f"Model: {model_used}",
        "type": "llm_inference", 
        "status": status,
        "start_time": em.get("load_time_ms", 0) / 1000.0,
        "end_time": em.get("total_time_ms", 1) / 1000.0,
        "output": (full_state.get("output") or "Sin salida")[:100] + "..."
    })
    
    # Output Validator Node
    if validation_md:
        vstatus = "error" if validation_md.get("should_retry") else "completed"
        base_s = em.get("total_time_ms", 1) / 1000.0
        flow_nodes.append({
            "id": "output_validator",
            "name": "Output Validator", 
            "type": "validator",
            "status": vstatus,
            "start_time": base_s,
            "end_time": base_s + validation_md.get("validation_time", 0.1),
            "output": f"Score: {validation_md.get('validation_results', {}).get('overall_score', 'N/A')}/10"
        })
    
    flow_logger.info(f"[FLOW] Built {len(flow_nodes)} flow nodes")
    return flow_nodes

def build_flow_edges(flow_nodes: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    """
    Construye las conexiones entre nodos del flow.
    
    Args:
        flow_nodes: Lista de nodos del flow
    
    Returns:
        Lista de edges para visualización en frontend
    """
    edges = []
    
    if len(flow_nodes) > 1:
        for i in range(len(flow_nodes) - 1):
            edges.append({
                "source": flow_nodes[i]["id"], 
                "target": flow_nodes[i + 1]["id"]
            })
    
    flow_logger.info(f"[FLOW] Built {len(edges)} flow edges")
    return edges

def build_execution_metrics(full_state: Dict[str, Any], flow_type: str, flow_nodes: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Construye las métricas de ejecución para el frontend.
    
    Args:
        full_state: Estado completo después de la ejecución
        flow_type: Tipo de flujo ejecutado
        flow_nodes: Nodos del flow construidos
    
    Returns:
        Dict con métricas para el frontend
    """
    em = full_state.get("execution_metrics", {}) or {}
    validation_md = full_state.get("validation_metadata", {}) or {}
    model_used = full_state.get("selected_model", "mistral7b")
    
    metrics = {
        "totalTime": em.get("total_time_ms", 0),
        "tokensGenerated": em.get("tokens_generated", len((full_state.get("output") or "").split())),
        "modelsUsed": [model_used],
        "cacheHit": em.get("cache_hit", False),
        "loadTime": em.get("load_time_ms", 0),
        "inferenceTime": em.get("inference_time_ms", 0),
        "workersExecuted": len(flow_nodes),
        "qualityScore": validation_md.get("validation_results", {}).get("overall_score", 0),
        "flowType": flow_type
    }
    
    flow_logger.info(f"[METRICS] Total time: {metrics['totalTime']}ms | Workers: {metrics['workersExecuted']}")
    return metrics

def build_challenge_flow_nodes(full_state: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Construye nodos específicos para challenge flow.
    ARREGLADO: Lee desde la estructura challenge_flow en el estado.
    """
    flow_nodes = []
    
    # Método 1: Leer desde estructura anidada en el estado
    challenge_flow_data = full_state.get("challenge_flow", {})
    
    if challenge_flow_data:
        flow_logger.info(f"[CHALLENGE] Found challenge_flow data with {len(challenge_flow_data)} nodes")
        
        # Construir nodos en orden específico
        node_order = ["creator", "challenger", "refiner"]
        
        for i, node_id in enumerate(node_order):
            if node_id in challenge_flow_data:
                node_data = challenge_flow_data[node_id]
                output = node_data.get("output", "")
                
                flow_nodes.append({
                    "id": node_id,
                    "name": node_id.capitalize(),
                    "type": f"llm_{node_id}",
                    "status": "completed",
                    "start_time": float(i),
                    "end_time": float(i + 1),
                    "output": output[:200] + "..." if len(output) > 200 else output,
                    "timestamp": node_data.get("timestamp", 0),
                    "full_content_length": len(output)
                })
                
                flow_logger.info(f"[CHALLENGE] Added {node_id}: {len(output)} chars")
        
        if flow_nodes:
            flow_logger.info(f"[CHALLENGE] Built {len(flow_nodes)} nodes from challenge_flow structure")
            return flow_nodes
    
    # Método 2: Fallback - Leer desde archivo JSON si existe
    json_file = full_state.get("challenge_flow_file")
    if json_file:
        try:
            import json
            with open(json_file, "r", encoding="utf-8") as f:
                challenge_data = json.load(f)
            
            nodes_data = challenge_data.get("nodes", {})
            flow_logger.info(f"[CHALLENGE] Found JSON file with {len(nodes_data)} nodes")
            
            node_order = ["creator", "challenger", "refiner"]
            for i, node_id in enumerate(node_order):
                if node_id in nodes_data:
                    node_data = nodes_data[node_id]
                    output = node_data.get("output", "")
                    
                    flow_nodes.append({
                        "id": node_id,
                        "name": node_id.capitalize(),
                        "type": f"llm_{node_id}",
                        "status": "completed",
                        "start_time": float(i),
                        "end_time": float(i + 1),
                        "output": output[:200] + "..." if len(output) > 200 else output,
                        "source": "json_file"
                    })
            
            if flow_nodes:
                flow_logger.info(f"[CHALLENGE] Built {len(flow_nodes)} nodes from JSON file")
                return flow_nodes
                
        except Exception as e:
            flow_logger.error(f"[CHALLENGE] Failed to read JSON file {json_file}: {e}")
    
    # Método 3: Fallback final - Detectar por número de ejecuciones
    messages = full_state.get("messages", [])
    execution_count = sum(1 for msg in messages if "[EXECUTE] ✅" in msg)
    
    if execution_count >= 3:
        flow_logger.info(f"[CHALLENGE] Detected {execution_count} executions, creating placeholder nodes")
        
        node_configs = [
            {"id": "creator", "name": "Creator"},
            {"id": "challenger", "name": "Challenger"}, 
            {"id": "refiner", "name": "Refiner"}
        ]
        
        for i, config in enumerate(node_configs):
            flow_nodes.append({
                "id": config["id"],
                "name": config["name"],
                "type": f"llm_{config['id']}",
                "status": "completed",
                "start_time": float(i),
                "end_time": float(i + 1),
                "output": f"✅ {config['name']} ejecutado (ver logs para detalles)",
                "source": "execution_count"
            })
        
        # El último nodo muestra el output final
        if flow_nodes:
            final_output = full_state.get("output", "")
            flow_nodes[-1]["output"] = final_output[:200] + "..." if len(final_output) > 200 else final_output
        
        flow_logger.info(f"[CHALLENGE] Built {len(flow_nodes)} placeholder nodes")
        return flow_nodes
    
    # Si todo falla, usar flow genérico
    flow_logger.warning("[CHALLENGE] No challenge flow data found, using generic flow")
    return build_flow_nodes(full_state)

def build_api_response(full_state: Dict[str, Any], flow_type: str) -> Dict[str, Any]:
    """
    Construye la respuesta completa para la API del frontend.
    
    Args:
        full_state: Estado completo después de la ejecución
        flow_type: Tipo de flujo ejecutado
    
    Returns:
        Dict con flow, output y metrics para el frontend
    """
    flow_logger.info(f"[API] Building response for flow_type: {flow_type}")
    
    try:
        # Construir nodos según el tipo de flow
        if flow_type == "challenge":
            flow_nodes = build_challenge_flow_nodes(full_state)
        else:
            flow_nodes = build_flow_nodes(full_state)
        
        # Construir edges
        flow_edges = build_flow_edges(flow_nodes)
        
        # Construir métricas
        metrics = build_execution_metrics(full_state, flow_type, flow_nodes)
        
        # Respuesta final
        result = {
            "flow": {
                "nodes": flow_nodes,
                "edges": flow_edges
            },
            "output": full_state.get("output", "Sin salida"),
            "metrics": metrics
        }
        
        flow_logger.info(f"[API] Response built successfully | nodes={len(flow_nodes)} | edges={len(flow_edges)}")
        return result
        
    except Exception as e:
        flow_logger.error(f"[API] Failed to build response: {e}")
        return build_error_response(str(e), flow_type)

def build_error_response(error_message: str, flow_type: str) -> Dict[str, Any]:
    """
    Construye una respuesta de error para la API.
    
    Args:
        error_message: Mensaje de error
        flow_type: Tipo de flujo que falló
    
    Returns:
        Dict con respuesta de error
    """
    flow_logger.info(f"[ERROR] Building error response for: {error_message}")
    
    return {
        "flow": {
            "nodes": [{
                "id": "error",
                "name": "Orchestrator Error",
                "type": "error",
                "status": "error",
                "output": f"Error en orquestador: {error_message}"
            }],
            "edges": []
        },
        "output": f"Error ejecutando orquestador: {error_message}",
        "metrics": {
            "totalTime": 0,
            "tokensGenerated": 0,
            "modelsUsed": [],
            "cacheHit": False,
            "loadTime": 0,
            "inferenceTime": 0,
            "workersExecuted": 0,
            "failed": True,
            "flowType": flow_type
        }
    }

def get_flow_summary(full_state: Dict[str, Any], flow_type: str) -> str:
    """
    Genera un resumen del flow ejecutado.
    
    Args:
        full_state: Estado completo después de la ejecución
        flow_type: Tipo de flujo ejecutado
    
    Returns:
        String con resumen del flow
    """
    em = full_state.get("execution_metrics", {}) or {}
    model_used = full_state.get("selected_model", "unknown")
    task_type = full_state.get("task_type", "unknown")
    
    total_time = em.get("total_time_ms", 0) / 1000.0
    output_len = len(full_state.get("output", "") or "")
    
    summary = (
        f"Flow: {flow_type} | "
        f"Task: {task_type} | "
        f"Model: {model_used} | "
        f"Time: {total_time:.2f}s | "
        f"Output: {output_len} chars"
    )
    
    return summary