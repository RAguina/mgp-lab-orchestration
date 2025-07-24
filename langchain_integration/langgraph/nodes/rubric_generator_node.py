# langchain_integration/langgraph/nodes/rubric_generator_node.py
from langchain_integration.langgraph.agent_state import AgentState
from langchain_integration.langgraph.local_llm_node import build_local_llm_tool_node

def rubric_generator_node(state: AgentState) -> AgentState:
    print("🧮 Generando rúbricas...")
    messages = state.get("messages", [])

    llm = build_local_llm_tool_node(
        model_key=state["selected_model"],
        strategy=state["strategy"],
        max_tokens=512
    )

    prompt = f"""Dada la siguiente pregunta y respuesta de un modelo, genera criterios técnicos (rúbricas) para evaluarla críticamente:

❓ Pregunta: {state["input"]}
💬 Respuesta del modelo: {state["output"]}

Devuelve al menos 3 rúbricas en forma de lista clara y objetiva.
"""

    try:
        result = llm.invoke(prompt)
        messages.append("Rúbricas generadas exitosamente.")
        return {
            **state,
            "analysis_result": result,
            "messages": messages
        }
    except Exception as e:
        messages.append(f"Error al generar rúbricas: {str(e)}")
        return {
            **state,
            "analysis_result": "",
            "messages": messages
        }