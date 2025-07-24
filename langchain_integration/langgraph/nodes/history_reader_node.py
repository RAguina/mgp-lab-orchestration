# langchain_integration/langgraph/nodes/history_reader_node.py
import os
from langchain_integration.langgraph.agent_state import AgentState


def history_reader_node(state: AgentState) -> AgentState:
    print("📂 Leyendo historial de output anterior...")
    messages = state.get("messages", [])
    output_dir = "outputs"
    try:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        files = sorted(
            [f for f in os.listdir(output_dir) if f.endswith(".txt")],
            key=lambda f: os.path.getmtime(os.path.join(output_dir, f)),
            reverse=True
        )
        if files:
            last_file = os.path.join(output_dir, files[0])
            with open(last_file, "r", encoding="utf-8") as f:
                last_output = f.read()
                messages.append("Último output cargado del historial")
        else:
            last_output = ""
            messages.append("No se encontró output previo")
    except Exception as e:
        last_output = ""
        messages.append(f"Error al leer historial: {str(e)}")

    return {
        **state,
        "last_output": last_output,
        "messages": messages
    }