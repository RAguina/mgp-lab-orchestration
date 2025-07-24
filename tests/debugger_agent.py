# agents/debugger_agent.py
from autogen import AssistantAgent

def get_debugger_agent(llm_config):
    return AssistantAgent(
        name="debugger",
        system_message="Eres un experto en detectar bugs en código Python.",
        llm_config=llm_config,
    )
