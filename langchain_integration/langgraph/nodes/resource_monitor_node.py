# langchain_integration/langgraph/nodes/resource_monitor_node.py - QUICK FIX VERSION
from typing import Dict, Any
from langchain_integration.langgraph.routing_agent import AgentState
from langchain_integration.tools.lab_tools import VRAMMonitorTool
import logging

# Setup logger para este nodo
logger = logging.getLogger("resource_monitor")

def resource_monitor_node(state: AgentState) -> AgentState:
    '''
    Worker de monitoreo de recursos con quick fix para tests rápidos
    FUERZA optimized strategy para evitar timeouts en testing
    '''
    import time
    start_time = time.time()
    node_id = f"resource_monitor_{int(start_time)}"
    
    logger.info(f"[{node_id}] === RESOURCE MONITOR STARTED (QUICK FIX MODE) ===")
    print("[MONITOR] Verificando recursos disponibles...")

    messages = state.get("messages", [])
    should_optimize = True
    strategy = state.get("strategy", "optimized")  # Default del task_analyzer
    vram_status = ""
    
    try:
        vram_tool = VRAMMonitorTool()
        vram_status = vram_tool.run("")
        
        logger.info(f"[{node_id}] VRAM tool result: {vram_status[:100]}...")
        
        if "VRAM Libre:" in vram_status:
            try:
                free_vram = float(vram_status.split("VRAM Libre:")[1].split("GB")[0].strip())
                logger.info(f"[{node_id}] Parsed free VRAM: {free_vram}GB")
                
                # ✅ QUICK FIX: SIEMPRE usar optimized para tests rápidos
                if free_vram > 5.0:
                    should_optimize = True  # ← CHANGED: antes era False
                    strategy = "optimized"  # ← FORCED: siempre optimized
                    messages.append(f"[MONITOR] Suficiente VRAM ({free_vram}GB), FORZANDO optimized para test rápido")
                    logger.info(f"[{node_id}] QUICK FIX: Sufficient VRAM but forcing optimized for fast testing")
                else:
                    messages.append(f"[MONITOR] VRAM limitada ({free_vram}GB), manteniendo optimized")
                    logger.info(f"[{node_id}] Limited VRAM, keeping optimized")
                    
            except Exception as parse_error:
                logger.warning(f"[{node_id}] VRAM parsing failed: {parse_error}")
                messages.append("[MONITOR] No se pudo determinar VRAM, usando optimized por defecto")
                strategy = "optimized"  # ✅ Fallback seguro
        else:
            logger.warning(f"[{node_id}] Unexpected VRAM tool format")
            messages.append("[MONITOR] Formato VRAM inesperado, usando optimized por defecto")
            strategy = "optimized"  # ✅ Fallback seguro
            
    except Exception as vram_error:
        logger.error(f"[{node_id}] VRAM tool failed: {vram_error}")
        vram_status = f"Error: {str(vram_error)}"
        messages.append(f"[MONITOR] Error en monitoreo, usando optimized: {str(vram_error)}")
        strategy = "optimized"  # ✅ Fallback seguro
    
    # ✅ QUICK FIX: Double-check que siempre sea optimized
    if strategy != "optimized":
        logger.warning(f"[{node_id}] QUICK FIX: Forcing strategy from '{strategy}' to 'optimized'")
        strategy = "optimized"
        should_optimize = True
        messages.append("[MONITOR] QUICK FIX: Forzando estrategia optimized para testing")
    
    # Resultado
    end_time = time.time()
    total_time = end_time - start_time
    
    logger.info(f"[{node_id}] === RESOURCE MONITOR COMPLETED ===")
    logger.info(f"[{node_id}] Total processing time: {total_time:.3f}s")
    logger.info(f"[{node_id}] FINAL strategy: {strategy} (QUICK FIX APPLIED)")
    
    print(f"[MONITOR] Completado: {strategy} (QUICK FIX - siempre optimized)")
    
    return {
        **state,
        "vram_status": vram_status.encode('ascii', 'ignore').decode('ascii'),  # ✅ Clean encoding
        "should_optimize": should_optimize,
        "strategy": strategy,  # ✅ Guaranteed to be "optimized"
        "messages": messages
    }