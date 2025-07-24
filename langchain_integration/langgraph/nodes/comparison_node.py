# langchain_integration/langgraph/nodes/comparison_node.py
from langchain_integration.langgraph.agent_state import AgentState
from langchain_integration.langgraph.local_llm_node import build_local_llm_tool_node

def comparison_node(state: AgentState) -> AgentState:
    print("⚖️ Comparando respuestas...")
    messages = state.get("messages", [])

    llm = build_local_llm_tool_node(
        model_key=state["selected_model"],
        strategy=state["strategy"],
        max_tokens=512
    )

    prompt = f"""Dada la siguiente pregunta y dos respuestas de distintos modelos, decide cuál es mejor y por qué. Usa criterios de claridad, corrección técnica y utilidad.

❓ Pregunta:
{state["input"]}

🅰️ Respuesta A:
{state.get("output_a", "(sin definir)")}

🅱️ Respuesta B:
{state.get("output_b", "(sin definir)")}

Indica cuál preferís y justifica la elección brevemente.
"""

    try:
        result = llm.invoke(prompt)
        messages.append("Comparación completada.")
        return {
            **state,
            "output": result,
            "messages": messages
        }
    except Exception as e:
        messages.append(f"Error en comparación: {str(e)}")
        return {
            **state,
            "output": "",
            "messages": messages
        }