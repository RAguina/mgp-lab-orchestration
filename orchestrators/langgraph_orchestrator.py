# orchestrators/langgraph_orchestrator.py

from langgraph.graph import StateGraph, END

# Workers simulados: podés reemplazar por tus propios LLM/funciones
def worker_token_creation(state):
    print("[LangGraph] Token creation")
    return {"token": "Token creado"}

def worker_web_dev(state):
    print("[LangGraph] Web dev")
    return {"web": "Web generada"}

def worker_whitepaper(state):
    print("[LangGraph] Whitepaper")
    return {"whitepaper": "Whitepaper generado"}

# Grafo simple: token → web → whitepaper → END
def main():
    builder = StateGraph()
    builder.add_node("token", worker_token_creation)
    builder.add_node("web", worker_web_dev)
    builder.add_node("whitepaper", worker_whitepaper)

    builder.add_edge("token", "web")
    builder.add_edge("web", "whitepaper")
    builder.add_edge("whitepaper", END)

    graph = builder.compile()
    # Estado inicial (puede ser dict vacío)
    result = graph.invoke({})
    print(result)

if __name__ == "__main__":
    main()
