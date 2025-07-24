# tests/langgraph/test_rubric_generator.py

from langchain_integration.langgraph.nodes.rubric_generator_node import rubric_generator_node

# Simular estado
state = {
    "input": "Explica qué es un modelo de lenguaje.",
    "output": "Un modelo de lenguaje es una red neuronal entrenada para predecir la siguiente palabra en una secuencia de texto.",
    "selected_model": "mistral7b",
    "strategy": "optimized",
    "messages": []
}

# Ejecutar nodo directamente
new_state = rubric_generator_node(state)

# Mostrar resultado
print("\n🔍 Resultado del nodo:")
print(new_state["analysis_result"])
print("\n🧾 Trazas:")
for msg in new_state["messages"]:
    print(f" - {msg}")
