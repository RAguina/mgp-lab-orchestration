# examples/langchain_basic.py - Test básico de LangChain
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
from config import Config

def test_langchain_basic():
    """Test básico de LangChain con un chat simple"""
    Config.validate()
    
    # Inicializar el modelo
    llm = ChatOpenAI(
        api_key=Config.OPENAI_API_KEY,
        model=Config.DEFAULT_MODEL,
        temperature=Config.TEMPERATURE
    )
    
    # Crear mensajes
    messages = [
        SystemMessage(content="Eres un asistente de programación Python. Responde de forma concisa."),
        HumanMessage(content="Explica qué es una función lambda en Python con un ejemplo.")
    ]
    
    # Obtener respuesta
    response = llm.invoke(messages)
    print("=== LANGCHAIN TEST ===")
    print(f"Respuesta: {response.content}")
    print("=====================\n")

if __name__ == "__main__":
    test_langchain_basic()