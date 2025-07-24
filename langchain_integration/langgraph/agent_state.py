# langchain_integration/langgraph/agent_state.py
from typing import TypedDict, List, Dict, Any, Optional

# Estado principal del agente (compatible con LangGraph)
class AgentState(TypedDict):
    # Input/Output básico
    input: str
    output: str
    
    # Análisis de tarea
    task_type: str
    analysis_result: str
    
    # Configuración de ejecución
    selected_model: str
    strategy: str
    vram_status: str
    should_optimize: bool
    
    # Control de flujo
    retry_count: int
    retry: bool
    last_output: str
    
    # Logging y mensajes
    messages: List[str]
    final_summary: str
    
    # Métricas de ejecución (nuevo)
    execution_metrics: Optional[Dict[str, Any]]

# Tipos auxiliares para mejor organización
class TaskType:
    CODE = "code"
    CHAT = "chat" 
    ANALYSIS = "analysis"
    CREATIVE = "creative"
    TECHNICAL = "technical"

class Strategy:
    STANDARD = "standard"
    OPTIMIZED = "optimized"
    STREAMING = "streaming"

class NodeStatus:
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    ERROR = "error"

# Estado inicial por defecto
DEFAULT_AGENT_STATE: AgentState = {
    "input": "",
    "output": "",
    "task_type": "",
    "selected_model": "mistral7b",  # Default seguro
    "strategy": Strategy.OPTIMIZED,
    "vram_status": "",
    "should_optimize": True,
    "messages": [],
    "analysis_result": "",
    "final_summary": "",
    "retry_count": 0,
    "retry": False,
    "last_output": "",
    "execution_metrics": None,
}

# Funciones auxiliares para crear estados
def create_initial_state(user_input: str) -> AgentState:
    """Crea un estado inicial con el input del usuario"""
    state = DEFAULT_AGENT_STATE.copy()
    state["input"] = user_input
    state["messages"] = []
    return state

def add_message(state: AgentState, message: str) -> AgentState:
    """Helper para agregar mensajes al estado"""
    messages = state.get("messages", [])
    messages.append(message)
    return {**state, "messages": messages}

def set_execution_metrics(state: AgentState, metrics: Dict[str, Any]) -> AgentState:
    """Helper para setear métricas de ejecución"""
    return {**state, "execution_metrics": metrics}