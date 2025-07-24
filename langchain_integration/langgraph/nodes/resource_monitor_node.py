# langchain_integration/langgraph/nodes/resource_monitor_node.py
from typing import Dict, Any
from langchain_integration.langgraph.routing_agent import AgentState
from langchain_integration.tools.lab_tools import VRAMMonitorTool

def resource_monitor_node(state: AgentState) -> AgentState:
    print("\ud83d\udcce Verificando recursos disponibles...")
    messages = state.get("messages", [])
    vram_tool = VRAMMonitorTool()
    vram_status = vram_tool.run("")
    should_optimize = True
    strategy = "optimized"

    if "VRAM Libre:" in vram_status:
        try:
            free_vram = float(vram_status.split("VRAM Libre:")[1].split("GB")[0].strip())
            if free_vram > 5.0:
                should_optimize = False
                strategy = "standard"
                messages.append("Suficiente VRAM disponible, usando estrategia standard")
            else:
                messages.append("VRAM limitada, usando estrategia optimizada")
        except:
            messages.append("No se pudo determinar VRAM, usando estrategia optimizada")

    return {
        **state,
        "vram_status": vram_status,
        "should_optimize": should_optimize,
        "strategy": strategy,
        "messages": messages
    }