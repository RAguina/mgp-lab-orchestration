# langchain_integration/langgraph/agent_state.py
from typing import TypedDict, Dict, Any, List, Optional

class AgentState(TypedDict, total=False):
    # Meta
    version: str               # Ej: "v3.0.0"
    request_id: str            # UUID o similar para trazar la petición

    # Input/Output básico
    input: str
    output: str

    # Análisis y selección
    task_type: str
    selected_model: str
    strategy: str
    messages: List[str]

    # V3: datos de ejecución normalizados
    generation_request: Dict[str, Any]
    generation_result: Dict[str, Any]
    execution_metrics: Dict[str, Any]

    # V4: Dependency Injection via services
    services: Dict[str, Any]   # gateway, cache, metrics, etc.

def create_initial_state(
    user_input: str, 
    request_id: Optional[str] = None,
    services: Optional[Dict[str, Any]] = None
) -> AgentState:
    """
    Crea el estado inicial del agente con valores por defecto.
    - version: versión de la estructura del estado
    - request_id: identificador único de la petición  
    - services: dependency injection (gateway, cache, metrics, etc.)
    """
    return {
        "version": "v4.0.0",  # Updated for services support
        "request_id": request_id or "",
        "input": user_input,
        "messages": [],
        "strategy": "optimized",
        "services": services or {},
    }
