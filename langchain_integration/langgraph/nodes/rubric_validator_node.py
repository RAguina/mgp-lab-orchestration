# langchain_integration/langgraph/nodes/rubric_validator_node.py
from langchain_integration.langgraph.agent_state import AgentState
from langchain_integration.langgraph.local_llm_node import build_local_llm_tool_node

def rubric_validator_node(state: AgentState) -> AgentState:
    print("✅ Validando cumplimiento de rúbricas...")
    messages = state.get("messages", [])

    llm = build_local_llm_tool_node(
        model_key=state["selected_model"],
        strategy=state["strategy"],
        max_tokens=512
    )

    prompt = f"""Dadas las siguientes rúbricas y la respuesta de un modelo, evalúa si cumple cada criterio. Devuelve un resumen conciso.

🎯 Rúbricas:
{state["analysis_result"]}

💬 Respuesta del modelo:
{state["output"]}

Indica si se cumplen o no, y justifica brevemente.
"""

    try:
        result = llm.invoke(prompt)
        messages.append("Evaluación de rúbricas completada.")
        return {
            **state,
            "output": result,
            "messages": messages
        }
    except Exception as e:
        messages.append(f"Error al validar rúbricas: {str(e)}")
        return {
            **state,
            "output": "",
            "messages": messages
        }