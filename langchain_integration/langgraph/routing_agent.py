"""
Agente LangGraph con routing condicional y múltiples nodos especializados
Versión 4: Modularizado completamente - Solo ejecuta, no construye responses
"""

import logging
import time
from typing import Dict, Any

# Logger del orquestador
orchestrator_logger = logging.getLogger("orchestrator")

# Estado
from langchain_integration.langgraph.agent_state import AgentState, create_initial_state

# Gateway (inyectable)
from providers.provider_gateway import ProviderGateway

# Orquestación modularizada
from langchain_integration.langgraph.orchestration import build_routing_graph
from langchain_integration.langgraph.orchestration.flow_metrics import build_api_response, build_error_response

# Nodos - solo execution_mod para gateway injection
import langchain_integration.langgraph.nodes.execution_node as execution_mod


def run_routing_agent(
    user_input: str, 
    gateway: ProviderGateway | None = None, 
    flow_type: str = "linear",
    model: str = "mistral7b",
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Ejecuta el agente de routing con Gateway inyectado y flow type configurable.
    
    Args:
        user_input: Prompt del usuario
        gateway: Gateway de modelos (opcional, usa default si None)
        flow_type: Tipo de flujo ("linear", "challenge", etc.)
        model: Modelo específico solicitado por el usuario
        verbose: Si mostrar información detallada
    
    Returns:
        Dict con resultado de la ejecución
    """
    execution_id = f"exec_{int(time.time())}"
    start_time = time.time()

    orchestrator_logger.info(f"[{execution_id}] === ORCHESTRATOR START ===")
    orchestrator_logger.info(f"[{execution_id}] Input: '{user_input[:100]}...'")
    orchestrator_logger.info(f"[{execution_id}] Flow type: {flow_type}")
    orchestrator_logger.info(f"[{execution_id}] Verbose: {verbose}")

    try:
        # Gateway injection ahora se hace vía services en AgentState
        if gateway is None:
            gateway = ProviderGateway()

        # Construir grafo usando el builder modularizado
        t0 = time.time()
        graph = build_routing_graph(flow_type)
        build_ms = int((time.time() - t0) * 1000)
        orchestrator_logger.info(f"[{execution_id}] Graph built in {build_ms} ms")

        # Estado inicial con services injection y modelo solicitado
        services = {"gateway": gateway} if gateway else {}
        initial_state = create_initial_state(user_input, services=services, requested_model=model)
        orchestrator_logger.info(f"[{execution_id}] Initial state fields: {len(initial_state)} services: {list(services.keys())}")

        if verbose:
            print(f"\n🤖 Ejecutando agente con routing... [ID: {execution_id}]")
            print(f"📊 Flow type: {flow_type}")
            print("=" * 50)

        # Ejecutar
        t1 = time.time()
        result = graph.invoke(initial_state)
        exec_ms = int((time.time() - t1) * 1000)
        total_ms = int((time.time() - start_time) * 1000)

        orchestrator_logger.info(f"[{execution_id}] Graph exec: {exec_ms} ms | total: {total_ms} ms")

        # Log resumen
        out_len = len(result.get("output", "") or "")
        msgs = len(result.get("messages", []) or [])
        orchestrator_logger.info(f"[{execution_id}] Output len: {out_len} | Messages: {msgs}")
        orchestrator_logger.info(f"[{execution_id}] Task: {result.get('task_type', 'unknown')} | Model: {result.get('selected_model', 'unknown')}")

        # Métricas
        m = result.get("execution_metrics", {}) or {}
        if m:
            orchestrator_logger.info(
                f"[{execution_id}] Worker metrics: cache={m.get('cache_hit')} | "
                f"load={m.get('load_time_ms','?')}ms | infer={m.get('inference_time_ms','?')}ms"
            )

        if verbose:
            print("\n📊 Proceso completado:")
            print(f"⏱️ Tiempo total: {total_ms/1000:.2f}s")
            print("\n🔍 Mensajes del proceso:")
            for i, msg in enumerate(result.get('messages', []), 1):
                print(f"  {i:2d}. {msg}")

            if m:
                print(f"\n📈 Métricas:")
                print(f"  Cache: {'HIT' if m.get('cache_hit') else 'MISS'}")
                print(f"  Carga: {m.get('load_time_ms', 0)}ms | Inferencia: {m.get('inference_time_ms', 0)}ms")

        orchestrator_logger.info(f"[{execution_id}] === ORCHESTRATOR DONE ===")
        return result

    except Exception as e:
        err_ms = int((time.time() - start_time) * 1000)
        orchestrator_logger.error(f"[{execution_id}] === ORCHESTRATOR FAILED ===")
        orchestrator_logger.error(f"[{execution_id}] Error after {err_ms} ms: {e}")

        if verbose:
            print(f"\n❌ Error en orquestador: {e}")

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
                "total_time_ms": err_ms,
            }
        }


def run_orchestrator(prompt: str, flow_type: str = "linear", model: str = "mistral7b", tools: list = None, rag_config: dict = None) -> Dict[str, Any]:
    """
    Wrapper público para backend: ejecuta el routing_agent y produce flow+metrics.
    
    Args:
        prompt: Prompt del usuario
        flow_type: Tipo de flujo a ejecutar
        model: Modelo específico solicitado por el usuario
        tools: Lista de herramientas/tools a usar (incluye RAG tools)
        rag_config: Configuración específica de RAG
    
    Returns:
        Dict con flow, output y metrics para el frontend
    """
    api_call_id = f"api_{int(time.time())}"
    orchestrator_logger.info(f"[{api_call_id}] API run_orchestrator")
    orchestrator_logger.info(f"[{api_call_id}] Prompt: '{prompt[:100]}...'")
    orchestrator_logger.info(f"[{api_call_id}] Flow type: {flow_type}")
    
    # Log RAG configuration if present
    if tools:
        orchestrator_logger.info(f"[{api_call_id}] Tools: {tools}")
    if rag_config:
        orchestrator_logger.info(f"[{api_call_id}] RAG config: {rag_config}")

    try:
        # TODO: Pass tools and rag_config to routing agent once RAG flows are implemented
        # For now, fallback to standard execution
        full_state = run_routing_agent(prompt, flow_type=flow_type, model=model, verbose=False)
        
        # Construir respuesta usando flow_metrics
        result = build_api_response(full_state, flow_type)
        
        # Log de éxito
        nodes_count = len(result.get("flow", {}).get("nodes", []))
        orchestrator_logger.info(f"[{api_call_id}] API response OK | workers={nodes_count} | flow={flow_type}")
        
        return result

    except Exception as e:
        orchestrator_logger.error(f"[{api_call_id}] API failed: {e}")
        return build_error_response(str(e), flow_type)


if __name__ == "__main__":
    orchestrator_logger.setLevel(logging.DEBUG)
    print("🥚 Demo de Agente con Routing v4.0 (Completamente Modularizado)")
    print("=" * 60)
    
    # Importar lista de flows disponibles
    from langchain_integration.langgraph.orchestration import list_available_flows, get_flow_description
    
    # Mostrar flows disponibles
    available_flows = list_available_flows()
    print("📊 Flow types disponibles:")
    for flow in available_flows:
        description = get_flow_description(flow)
        print(f"  • {flow:<15} - {description}")
    
    print("\n📝 Prompts de ejemplo:")
    tests = [
        "Escribe una función en Python para calcular fibonacci",
        "¿Qué es la inteligencia artificial?", 
        "Cuéntame una historia corta sobre un robot",
        "Analiza las ventajas y desventajas de usar IA"
    ]
    
    for i, q in enumerate(tests, 1):
        print(f"  {i}. {q}")
    
    # Input del usuario
    user_input = input("\n📝 Tu prompt (o número de ejemplo): ").strip()
    if user_input.isdigit():
        idx = int(user_input) - 1
        if 0 <= idx < len(tests):
            user_input = tests[idx]
    
    # Selección de flow
    print(f"\n🔄 Flows disponibles: {', '.join(available_flows)}")
    flow_choice = input(f"🔄 Flow type [{available_flows[0]}]: ").strip() or available_flows[0]
    
    # Ejecutar
    result = run_routing_agent(user_input, flow_type=flow_choice, verbose=True)
    
    # Mostrar resultado
    print("\n✨ Respuesta generada:")
    print("-" * 60)
    print(result.get('output', ''))
    
    # Mostrar resumen del flow
    from langchain_integration.langgraph.orchestration.flow_metrics import get_flow_summary
    summary = get_flow_summary(result, flow_choice)
    print(f"\n📊 Resumen: {summary}")