# langchain_integration/tools/history_tools.py
"""
Herramientas para leer historiales de outputs generados por modelos
Incluye nodo reusable para LangGraph: history_reader_node
"""

import os
from typing import Optional, Dict, Any, Literal
from datetime import datetime
from langchain_core.runnables import RunnableLambda

# Tipo esperado para el estado
from langchain_integration.langgraph.agent_state import AgentState


# -----------------------------------------------------------------------------
# Función auxiliar: buscar último archivo de output
# -----------------------------------------------------------------------------
def find_latest_output_file(model_key: str) -> Optional[str]:
    base_dir = os.path.join("outputs", model_key, "runs")
    if not os.path.exists(base_dir):
        return None

    txt_files = [
        os.path.join(base_dir, f)
        for f in os.listdir(base_dir)
        if f.endswith(".txt")
    ]
    if not txt_files:
        return None

    # Ordenar por fecha de modificación
    txt_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
    return txt_files[0]


# -----------------------------------------------------------------------------
# Nodo LangGraph para leer el último output del modelo actual
# -----------------------------------------------------------------------------
def history_reader_node(state: AgentState) -> AgentState:
    print("📖 Leyendo último output generado por el modelo...")
    model_key = state.get("selected_model", "mistral7b")
    messages = state.get("messages", [])

    last_output = ""
    try:
        latest_path = find_latest_output_file(model_key)
        if latest_path:
            # ✅ FIX: Usar 'errors="replace"' para manejar caracteres problemáticos
            with open(latest_path, "r", encoding="utf-8", errors="replace") as f:
                last_output = f.read().strip()
                
            # ✅ FIX: Limpiar caracteres surrogates si existen
            last_output = last_output.encode('utf-8', errors='ignore').decode('utf-8')
            
            messages.append(f"Último output cargado desde: {latest_path}")
        else:
            messages.append("No se encontró output previo para este modelo")
    except Exception as e:
        messages.append(f"Error leyendo historial: {str(e)}")

    return {
        **state,
        "last_output": last_output,
        "messages": messages
    }

# -----------------------------------------------------------------------------
# Función de routing condicional si se desea usar solo en ciertos casos
# -----------------------------------------------------------------------------
def should_include_history(state: AgentState) -> Literal["read_history", "skip_history"]:
    # ✅ TEMPORAL: Siempre skip hasta que limpiemos los archivos viejos
    return "skip_history"
    
    # Original (comentado temporalmente):
    # if state["task_type"] in ["analysis", "code"]:
    #     return "read_history"
    # return "skip_history"


# Exportar nodo como RunnableLambda
HistoryReaderNode = RunnableLambda(history_reader_node)
