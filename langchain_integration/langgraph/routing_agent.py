"""
Agente LangGraph con routing condicional y múltiples nodos especializados
Versión 2: Con logging detallado y arquitectura Worker/Orchestrator
"""

import logging
import time
from typing import Literal, List, Dict, Any
from langgraph.graph import StateGraph, END
from langchain_core.runnables import RunnableLambda

# Configurar logging detallado
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    handlers=[
        logging.StreamHandler(),  # Console output
        logging.FileHandler('logs/orchestrator.log', mode='a')  # File output
    ]
)

# Logger específico para el orquestrador
orchestrator_logger = logging.getLogger("orchestrator")

# Importar AgentState desde archivo separado (evita imports circulares)
from langchain_integration.langgraph.agent_state import AgentState, create_initial_state

# Importar herramientas, nodos y lógica externa
from langchain_integration.langgraph.local_llm_node import build_local_llm_tool_node
from langchain_integration.langgraph.nodes.task_analyzer_node import task_analyzer_node
from langchain_integration.langgraph.nodes.output_validator_node import (
    output_validator_node, route_after_validation
)
from langchain_integration.tools.history_tools import (
    HistoryReaderNode, should_include_history
)
from langchain_integration.langgraph.nodes.resource_monitor_node import resource_monitor_node
from langchain_integration.langgraph.nodes.execution_node import execution_node
from langchain_integration.langgraph.nodes.summary_node import summary_node

def route_after_analysis(state: AgentState) -> Literal["monitor", "skip_monitor"]:
    """
    Ruta después del análisis de tarea con logging detallado
    """
    task_type = state.get("task_type", "unknown")
    
    orchestrator_logger.info(f"[ROUTING] Deciding path after analysis for task_type: {task_type}")
    
    if task_type in ["code", "analysis"]:
        orchestrator_logger.info(f"[ROUTING] Task '{task_type}' requires resource monitoring -> monitor")
        return "monitor"
    elif task_type == "chat":
        orchestrator_logger.info(f"[ROUTING] Task '{task_type}' is simple chat -> skip_monitor")
        return "skip_monitor"
    else:
        orchestrator_logger.info(f"[ROUTING] Unknown task '{task_type}' -> monitor (safe default)")
        return "monitor"

def build_routing_graph():
    """
    Construye el grafo de routing con todos los workers
    """
    orchestrator_logger.info("[GRAPH] Building routing graph with worker nodes")
    
    builder = StateGraph(AgentState)
    
    # Registrar workers (nodos especializados)
    workers = {
        "analyzer": task_analyzer_node,
        "monitor": resource_monitor_node,
        "executor": execution_node,
        "validator": output_validator_node,
        "history": HistoryReaderNode,
        "summarizer": summary_node
    }
    
    orchestrator_logger.info(f"[GRAPH] Registering {len(workers)} workers: {list(workers.keys())}")
    
    # Agregar nodos (workers) al grafo
    for worker_name, worker_func in workers.items():
        if worker_name == "history":
            builder.add_node(worker_name, worker_func)  # HistoryReaderNode ya es RunnableLambda
        else:
            builder.add_node(worker_name, RunnableLambda(worker_func))
        orchestrator_logger.debug(f"[GRAPH] Worker '{worker_name}' registered")

    # Configurar flujo de workers
    orchestrator_logger.info("[GRAPH] Configuring worker flow and routing logic")
    
    builder.set_entry_point("analyzer")
    
    # Routing condicional después del análisis
    builder.add_conditional_edges("analyzer", route_after_analysis, {
        "monitor": "monitor",
        "skip_monitor": "executor"
    })
    
    # Flujo lineal para el resto
    builder.add_edge("monitor", "executor")
    builder.add_edge("executor", "validator")
    
    # Validación con posible retry
    builder.add_conditional_edges("validator", route_after_validation, {
        "retry_execution": "executor",
        "continue": "history"
    })
    
    # Historia condicional
    builder.add_conditional_edges("history", should_include_history, {
        "read_history": "summarizer",
        "skip_history": "summarizer"
    })
    
    # Finalizar
    builder.add_edge("summarizer", END)
    
    orchestrator_logger.info("[GRAPH] Graph compilation completed successfully")
    
    return builder.compile()

def run_routing_agent(user_input: str, verbose: bool = True) -> Dict[str, Any]:
    """
    Ejecuta el agente de routing completo con logging detallado
    """
    execution_id = f"exec_{int(time.time())}"
    start_time = time.time()
    
    orchestrator_logger.info(f"[{execution_id}] === ORCHESTRATOR EXECUTION STARTED ===")
    orchestrator_logger.info(f"[{execution_id}] User input: '{user_input[:100]}...'")
    orchestrator_logger.info(f"[{execution_id}] Verbose mode: {verbose}")
    
    try:
        # Construir grafo
        graph_start = time.time()
        graph = build_routing_graph()
        graph_time = time.time() - graph_start
        
        orchestrator_logger.info(f"[{execution_id}] Graph built in {graph_time:.3f}s")
        
        # Crear estado inicial
        initial_state = create_initial_state(user_input)
        orchestrator_logger.info(f"[{execution_id}] Initial state created with {len(initial_state)} fields")
        
        if verbose:
            print(f"\n🤖 Ejecutando agente con routing... [ID: {execution_id}]")
            print("=" * 50)

        # Ejecutar grafo con logging de progreso
        execution_start = time.time()
        orchestrator_logger.info(f"[{execution_id}] Starting graph execution")
        
        result = graph.invoke(initial_state)
        
        execution_time = time.time() - execution_start
        total_time = time.time() - start_time
        
        orchestrator_logger.info(f"[{execution_id}] Graph execution completed in {execution_time:.2f}s")
        orchestrator_logger.info(f"[{execution_id}] Total orchestrator time: {total_time:.2f}s")
        
        # Log resultado
        output_length = len(result.get('output', ''))
        messages_count = len(result.get('messages', []))
        
        orchestrator_logger.info(f"[{execution_id}] Execution results:")
        orchestrator_logger.info(f"[{execution_id}] - Output length: {output_length} chars")
        orchestrator_logger.info(f"[{execution_id}] - Messages generated: {messages_count}")
        orchestrator_logger.info(f"[{execution_id}] - Task type: {result.get('task_type', 'unknown')}")
        orchestrator_logger.info(f"[{execution_id}] - Model used: {result.get('selected_model', 'unknown')}")
        
        # Extraer métricas de workers
        execution_metrics = result.get('execution_metrics', {})
        if execution_metrics:
            cache_hit = execution_metrics.get('cache_hit', False)
            load_time = execution_metrics.get('load_time', 0)
            inference_time = execution_metrics.get('inference_time', 0)
            
            orchestrator_logger.info(f"[{execution_id}] Worker metrics:")
            orchestrator_logger.info(f"[{execution_id}] - Cache hit: {cache_hit}")
            orchestrator_logger.info(f"[{execution_id}] - Load time: {load_time:.2f}s")
            orchestrator_logger.info(f"[{execution_id}] - Inference time: {inference_time:.2f}s")

        if verbose:
            print("\n📊 Proceso completado:")
            print(f"📍 {result.get('final_summary', 'Sin resumen')}")
            print(f"⏱️ Tiempo total: {total_time:.2f}s")
            print("\n🔍 Mensajes del proceso:")
            for i, msg in enumerate(result.get('messages', []), 1):
                print(f"  {i:2d}. {msg}")
            
            # Mostrar métricas si están disponibles
            if execution_metrics:
                print(f"\n📈 Métricas:")
                print(f"  Cache: {'HIT' if cache_hit else 'MISS'}")
                print(f"  Carga: {load_time:.1f}s | Inferencia: {inference_time:.1f}s")

        orchestrator_logger.info(f"[{execution_id}] === ORCHESTRATOR EXECUTION COMPLETED ===")
        
        return result
        
    except Exception as e:
        error_time = time.time() - start_time
        orchestrator_logger.error(f"[{execution_id}] === ORCHESTRATOR EXECUTION FAILED ===")
        orchestrator_logger.error(f"[{execution_id}] Error after {error_time:.2f}s: {str(e)}")
        orchestrator_logger.error(f"[{execution_id}] Exception type: {type(e).__name__}")
        
        if verbose:
            print(f"\n❌ Error en orquestador: {str(e)}")
        
        # Retornar estado de error
        return {
            "input": user_input,
            "output": f"Error en orquestador: {str(e)}",
            "task_type": "error",
            "selected_model": "none",
            "messages": [f"[ERROR] Orquestador falló: {str(e)}"],
            "final_summary": "Ejecución fallida",
            "execution_metrics": {
                "failed": True,
                "error": str(e),
                "total_time": error_time
            }
        }

def run_orchestrator(prompt: str) -> dict:
    """
    Wrapper público para usar el agente desde el backend.
    Ejecuta el routing_agent y extrae output, flow y métricas.
    """
    api_call_id = f"api_{int(time.time())}"
    orchestrator_logger.info(f"[{api_call_id}] API call to run_orchestrator")
    orchestrator_logger.info(f"[{api_call_id}] Prompt: '{prompt[:100]}...'")
    
    try:
        full_state = run_routing_agent(prompt, verbose=False)
        model_used = full_state.get("selected_model", "mistral7b")
        
        # Extraer métricas de ejecución si están disponibles
        execution_metrics = full_state.get("execution_metrics", {})
        analysis_metadata = full_state.get("analysis_metadata", {})
        validation_metadata = full_state.get("validation_metadata", {})
        
        orchestrator_logger.info(f"[{api_call_id}] Building flow representation")
        
        # Crear flow más detallado basado en workers ejecutados
        flow_nodes = []
        
        # Nodo de análisis (siempre se ejecuta)
        if full_state.get("task_type"):
            flow_nodes.append({
                "id": "task_analyzer",
                "name": "Task Analyzer",
                "type": "task_analyzer",
                "status": "completed",
                "start_time": 0,
                "end_time": analysis_metadata.get("processing_time", 0.5),
                "output": f"Detected: {full_state.get('task_type')} → {model_used}"
            })
        
        # Nodo de monitoreo (si se ejecutó)
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
        
        # Nodo de ejecución principal (siempre se ejecuta)
        execution_status = "completed" if execution_metrics.get("failed") != True else "error"
        flow_nodes.append({
            "id": "model_execution",
            "name": f"Model: {model_used}",
            "type": "llm_inference",
            "status": execution_status,
            "start_time": execution_metrics.get("load_time", 0),
            "end_time": execution_metrics.get("total_time", 1),
            "output": full_state.get("output", "Sin salida")[:100] + "..."
        })
        
        # Nodo de validación (si se ejecutó)
        if validation_metadata:
            validation_status = "error" if validation_metadata.get("should_retry") else "completed"
            flow_nodes.append({
                "id": "output_validator",
                "name": "Output Validator",
                "type": "validator",
                "status": validation_status,
                "start_time": execution_metrics.get("total_time", 1),
                "end_time": execution_metrics.get("total_time", 1) + validation_metadata.get("validation_time", 0.1),
                "output": f"Score: {validation_metadata.get('validation_results', {}).get('overall_score', 'N/A')}/10"
            })

        # Crear edges dinámicamente
        edges = []
        if len(flow_nodes) > 1:
            for i in range(len(flow_nodes) - 1):
                edges.append({
                    "source": flow_nodes[i]["id"],
                    "target": flow_nodes[i + 1]["id"]
                })

        result = {
            "flow": {
                "nodes": flow_nodes,
                "edges": edges
            },
            "output": full_state.get("output", "Sin salida"),
            "metrics": {
                "totalTime": execution_metrics.get("total_time", 1) * 1000,  # Convert to ms
                "tokensGenerated": execution_metrics.get("tokens_generated", 
                                                       len(full_state.get("output", "").split())),
                "modelsUsed": [model_used],
                "cacheHit": execution_metrics.get("cache_hit", False),
                "loadTime": execution_metrics.get("load_time", 0) * 1000,
                "inferenceTime": execution_metrics.get("inference_time", 0) * 1000,
                "workersExecuted": len(flow_nodes),
                "qualityScore": validation_metadata.get('validation_results', {}).get('overall_score', 0)
            }
        }
        
        orchestrator_logger.info(f"[{api_call_id}] Successfully created API response")
        orchestrator_logger.info(f"[{api_call_id}] Workers executed: {len(flow_nodes)}")
        
        return result
        
    except Exception as e:
        orchestrator_logger.error(f"[{api_call_id}] API call failed: {str(e)}")
        
        # Fallback en caso de error
        return {
            "flow": {
                "nodes": [{
                    "id": "error",
                    "name": "Orchestrator Error",
                    "type": "error",
                    "status": "error",
                    "output": f"Error en orquestador: {str(e)}"
                }],
                "edges": []
            },
            "output": f"Error ejecutando orquestador: {str(e)}",
            "metrics": {
                "totalTime": 0,
                "tokensGenerated": 0,
                "modelsUsed": [],
                "cacheHit": False,
                "loadTime": 0,
                "inferenceTime": 0,
                "workersExecuted": 0,
                "failed": True
            }
        }

if __name__ == "__main__":
    # Configurar logging adicional para modo debug
    orchestrator_logger.setLevel(logging.DEBUG)
    
    print("🥚 Demo de Agente con Routing v2.0")
    print("=" * 50)
    test_queries = [
        "Escribe una función en Python para calcular fibonacci",
        "¿Qué es la inteligencia artificial?",
        "Cuéntame una historia corta sobre un robot",
        "Analiza las ventajas y desventajas de usar IA"
    ]
    print("\nEjemplos de queries para probar:")
    for i, q in enumerate(test_queries, 1):
        print(f"{i}. {q}")
    
    user_input = input("\n📝 Tu prompt (o número de ejemplo): ").strip()
    if user_input.isdigit():
        idx = int(user_input) - 1
        if 0 <= idx < len(test_queries):
            user_input = test_queries[idx]
    
    result = run_routing_agent(user_input)
    print("\n✨ Respuesta generada:")
    print("-" * 50)
    print(result['output'])