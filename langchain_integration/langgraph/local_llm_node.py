# langchain_integration/langgraph/local_llm_node.py

from langchain_core.runnables import Runnable
from typing import Optional, Dict, Any

from langchain_integration.memory.local_llm_manager import LocalLLMManager

# Inicializar el manager global del laboratorio
manager = LocalLLMManager()


class LocalLLMRunnable(Runnable):
    """
    Runnable que permite usar modelos locales del laboratorio como pasos de LangGraph
    """

    def __init__(self, model_key: str, strategy: str = "optimized", **kwargs):
        self.model_key = model_key
        self.strategy = strategy
        self.wrapper_kwargs = kwargs

    def invoke(self, input: str, config: Optional[Dict[str, Any]] = None) -> str:
        # Obtener el wrapper del manager
        wrapper = manager.get_llm(
            model_key=self.model_key, 
            strategy=self.strategy, 
            **self.wrapper_kwargs
        )
        return wrapper.invoke(input)

    async def ainvoke(self, input: str, config: Optional[Dict[str, Any]] = None) -> str:
        # Por ahora, usar versión síncrona
        return self.invoke(input, config)

    def __repr__(self):
        return f"<LocalLLMRunnable model='{self.model_key}' strategy='{self.strategy}'>"


def build_local_llm_tool_node(model_key: str, strategy: str = "optimized", **kwargs) -> Runnable:
    """
    Construye un Runnable compatible con LangGraph, usando el wrapper local

    Args:
        model_key: Clave del modelo a usar
        strategy: Estrategia de carga
        **kwargs: Argumentos adicionales para el wrapper

    Returns:
        Runnable usable como nodo en el grafo LangGraph
    """
    return LocalLLMRunnable(model_key=model_key, strategy=strategy, **kwargs)


# Demo mínimo
if __name__ == "__main__":
    print("🧪 Probando LocalLLMRunnable")

    # Crear runnable directamente
    runnable = LocalLLMRunnable("mistral7b", max_tokens=64)
    response = runnable.invoke("¿Qué es LangChain?")
    print("\n🧠 Respuesta del Runnable:", response)