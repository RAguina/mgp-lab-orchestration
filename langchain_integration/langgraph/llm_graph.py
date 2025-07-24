# langchain_integration/langgraph/llm_graph.py

from langgraph.graph import StateGraph
from langchain_core.runnables import RunnableLambda
from typing import TypedDict

from langchain_integration.langgraph.local_llm_node import build_local_llm_tool_node

# Estado del grafo
class GraphState(TypedDict):
    input: str
    output: str

# Nodo de entrada (recibe el prompt)
def process_input(state: GraphState) -> GraphState:
    print("🔹 Nodo inicial: Recibiendo input del usuario")
    return {"input": state["input"]}

# Nodo LLM
llm_node = build_local_llm_tool_node("mistral7b", strategy="optimized", max_tokens=128)

# Wrapper para adaptarlo al grafo
def run_llm(state: GraphState) -> GraphState:
    print("🔸 Ejecutando modelo local...")
    result = llm_node.invoke(state["input"])
    return {"input": state["input"], "output": result}

# Construcción del grafo
def build_graph():
    builder = StateGraph(GraphState)
    
    # Cambiar nombres de nodos para evitar conflicto con claves de estado
    builder.add_node("process_input", RunnableLambda(process_input))
    builder.add_node("llm_processor", RunnableLambda(run_llm))

    builder.set_entry_point("process_input")
    builder.add_edge("process_input", "llm_processor")
    builder.set_finish_point("llm_processor")

    return builder.compile()

# Demo mínimo
if __name__ == "__main__":
    graph = build_graph()

    print("🔧 Grafo construido. Ingresá un prompt:\n")
    user_input = input("📝 Prompt: ")

    result = graph.invoke({"input": user_input})
    print("\n✅ Resultado final:")
    print(result["output"])