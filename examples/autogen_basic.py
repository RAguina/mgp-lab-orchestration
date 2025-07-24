# examples/autogen_basic.py - Test básico de AutoGen
import autogen
from config import Config

def test_autogen_basic():
    """Test básico de AutoGen con dos agentes que colaboran"""
    Config.validate()
    
    # Configuración del modelo
    config_list = [
        {
            "model": Config.DEFAULT_MODEL,
            "api_key": Config.OPENAI_API_KEY,
        }
    ]
    
    llm_config = {"config_list": config_list, "temperature": Config.TEMPERATURE}
    
    # Crear agentes
    assistant = autogen.AssistantAgent(
        name="assistant",
        system_message="Eres un programador Python experto. Ayudas a crear código limpio y eficiente.",
        llm_config=llm_config,
    )
    
    user_proxy = autogen.UserProxyAgent(
        name="user_proxy",
        human_input_mode="NEVER",
        max_consecutive_auto_reply=3,
        is_termination_msg=lambda x: x.get("content", "").rstrip().endswith("TERMINATE"),
        code_execution_config={"work_dir": "coding"},
        llm_config=llm_config,
    )
    
    print("=== AUTOGEN TEST ===")
    
    # Iniciar conversación
    user_proxy.initiate_chat(
        assistant,
        message="Crea una función en Python que calcule el factorial de un número. Incluye docstring y manejo de errores. Después de mostrar el código, di TERMINATE."
    )
    
    print("====================\n")

if __name__ == "__main__":
    test_autogen_basic()