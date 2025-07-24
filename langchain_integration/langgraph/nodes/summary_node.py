# langchain_integration/langgraph/nodes/summary_node.py
from langchain_integration.langgraph.agent_state import AgentState

def summary_node(state: AgentState) -> AgentState:
    print("\ud83d\udccb Generando resumen del proceso...")
    summary_parts = [
        f"Tarea: {state['task_type']}",
        f"Modelo usado: {state['selected_model']}",
        f"Estrategia: {state['strategy']}",
        f"Longitud respuesta: {len(state.get('output', ''))} caracteres"
    ]
    if state.get("vram_status") and "VRAM Usada:" in state["vram_status"]:
        vram_used = state["vram_status"].split("VRAM Usada:")[1].split("GB")[0].strip()
        summary_parts.append(f"VRAM usada: {vram_used}GB")
    final_summary = " | ".join(summary_parts)
    return {
        **state,
        "final_summary": final_summary
    }