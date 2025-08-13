# langchain_integration/langgraph/nodes/execution_node.py
import logging
import time
from typing import Any
from langchain_integration.langgraph.agent_state import AgentState
from providers.provider_gateway import ProviderGateway

logger = logging.getLogger("execution_worker")

def _max_tokens_for(task_type: str) -> int:
    """Asigna un límite de tokens según el tipo de tarea."""
    return {
        "code": 512,
        "technical": 384,
        "creative": 256,
        "analysis": 384,
        "chat": 256
    }.get(task_type, 256)

def _get_gateway_from_state(state: AgentState) -> ProviderGateway:
    """
    Obtiene el ProviderGateway desde state['services']['gateway'] si existe,
    con fallback seguro a una instancia por defecto (no rompe compat).
    """
    services = state.get("services", {}) or {}
    gw = services.get("gateway")
    if isinstance(gw, ProviderGateway):
        return gw
    return ProviderGateway()

def execution_node(state: AgentState) -> AgentState:
    """
    Nodo de ejecución de modelo.
    Llama al ProviderGateway y actualiza el estado con resultados, métricas y metadatos.
    """
    start = time.time()
    node_id = f"execution_worker_{int(start)}"

    messages   = list(state.get("messages", []))
    prompt     = state.get("input", "")
    task_type  = state.get("task_type", "chat")
    model_key  = state.get("selected_model", "mistral7b")
    strategy   = state.get("strategy", "optimized")
    max_tokens = _max_tokens_for(task_type)

    logger.info(f"[{node_id}] exec start model={model_key} strategy={strategy} task={task_type} len={len(prompt)}")

    req = {
        "model_key": model_key,
        "strategy": strategy,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0.7,
    }

    gw = _get_gateway_from_state(state)
    res = gw.generate(req)

    if res.get("success"):
        t = res.get("timings_ms", {}) or {}
        m = res.get("metrics", {}) or {}
        cache_hit = bool(m.get("cache_hit", False))

        messages += [
            f"[EXECUTE] ✅ cache={'HIT' if cache_hit else 'MISS'}",
            f"[EXECUTE] load={t.get('load_time_ms',0)}ms inf={t.get('inference_time_ms',0)}ms",
        ]

        return {
            **state,
            "output": res.get("output", ""),
            "messages": messages,
            "generation_request": req,
            "generation_result": res,
            "execution_metrics": {
                "cache_hit": cache_hit,
                "load_time_ms": t.get("load_time_ms", 0),
                "inference_time_ms": t.get("inference_time_ms", 0),
                "total_time_ms": t.get("total_time_ms", 0),
                "tokens_generated": m.get("tokens_generated", 0),
                "model_used": res.get("model_used", model_key),
                "strategy_used": res.get("strategy_used", strategy),
                "source": res.get("source", "local:hf"),
                "node_id": node_id,
            },
            "execution_metadata": {
                "request_timestamp": start,
                "completion_timestamp": time.time(),
                "duration_seconds": time.time() - start,
                "prompt_length": len(prompt),
                "max_tokens": max_tokens,
                "temperature": 0.7,
            }
        }

    # --- Error en ejecución ---
    err = res.get("error", "unknown")
    logger.error(f"[{node_id}] exec failed: {err}")
    messages.append(f"[EXECUTE] ❌ {err}")

    return {
        **state,
        "output": f"Error en ejecución: {err}",
        "messages": messages,
        "generation_request": req,
        "generation_result": res,
        "execution_metrics": {
            "failed": True,
            "node_id": node_id,
            "model_used": model_key,
            "strategy_used": strategy,
            "source": res.get("source", "local:hf")
        },
        "execution_metadata": {
            "request_timestamp": start,
            "completion_timestamp": time.time(),
            "duration_seconds": time.time() - start,
            "prompt_length": len(prompt),
            "max_tokens": max_tokens,
            "temperature": 0.7,
        }
    }
