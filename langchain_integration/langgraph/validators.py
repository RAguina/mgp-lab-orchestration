# langchain_integration/langgraph/validators.py

from typing import Literal
from langchain_core.runnables import RunnableLambda
from langchain_integration.langgraph.agent_state import AgentState

MAX_RETRIES = 1

def output_validator_node(state: AgentState) -> AgentState:
    """Valida la salida generada y decide si hacer retry"""
    print("\u2705 Validando salida del modelo...")

    output = state.get("output", "").strip()
    messages = state.get("messages", [])
    retry_count = state.get("retry_count", 0)

    # Criterios simples de validación
    too_short = len(output) < 50
    has_error = any(err in output.lower() for err in ["error", "traceback", "exception"])

    if (too_short or has_error) and retry_count < MAX_RETRIES:
        messages.append(f"⚠️ Salida inválida detectada (retry #{retry_count + 1})")
        return {
            **state,
            "retry": True,
            "retry_count": retry_count + 1,
            "messages": messages
        }

    messages.append("✅ Salida validada correctamente")
    return {
        **state,
        "retry": False,
        "messages": messages
    }

def route_after_validation(state: AgentState) -> Literal["retry_execution", "continue"]:
    return "retry_execution" if state.get("retry", False) else "continue"
