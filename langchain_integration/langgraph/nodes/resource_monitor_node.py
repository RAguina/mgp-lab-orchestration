# langchain_integration/langgraph/nodes/resource_monitor_node.py
import time
import logging
from typing import Dict, Any
from langchain_integration.langgraph.agent_state import AgentState
from utils.gpu_guard import get_gpu_info

logger = logging.getLogger("resource_monitor")

def resource_monitor_node(state: AgentState) -> AgentState:
    start = time.time()
    node_id = f"resource_monitor_{int(start)}"
    logger.info(f"[{node_id}] === RESOURCE MONITOR STARTED ===")
    print("[MONITOR] Verificando recursos disponibles...")

    vram = get_gpu_info()
    status = vram.get("memory_status", "unknown")
    free_gb = float(vram.get("free_gb", 0.0))
    total_gb = float(vram.get("total_gb", 0.0))

    # política simple: si hay < 4GB libres, mantené 'optimized'
    current_strategy = state.get("strategy", "optimized")
    forced_strategy = "optimized" if free_gb < 4.0 else current_strategy

    msgs = state.get("messages", [])
    msgs.append(f"[MONITOR] VRAM: {free_gb:.2f}/{total_gb:.2f} GB ({status}) → strategy={forced_strategy}")

    logger.info(f"[{node_id}] VRAM {free_gb:.2f}/{total_gb:.2f} GB — {status} → strategy={forced_strategy}")
    logger.info(f"[{node_id}] === RESOURCE MONITOR COMPLETED in {time.time()-start:.3f}s ===")

    return {
        **state,
        "vram_status": status,
        "vram_info": vram,
        "strategy": forced_strategy,
        "messages": msgs,
        "monitor_metadata": {
            "node_id": node_id,
            "node_time": round(time.time() - start, 3),
            "free_gb": free_gb,
            "total_gb": total_gb,
        }
    }
