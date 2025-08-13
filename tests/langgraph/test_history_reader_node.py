# tests/langgraph/test_history_reader_node.py
from langchain_integration.tools.history_tools import history_reader_node
from langchain_integration.langgraph.agent_state import AgentState


def test_history_reader_node_handles_missing_dir(tmp_path):
    state = AgentState(
        input="Analiza algo",
        output="",
        task_type="analysis",
        selected_model="modelo_inexistente",
        strategy="",
        vram_status="",
        should_optimize=True,
        messages=[],
        analysis_result="",
        final_summary="",
        retry_count=0,
        retry=False,
        last_output=""
    )
    result = history_reader_node(state)
    assert "No se encontró output previo" in result["messages"][-1] or "Error" in result["messages"][-1]