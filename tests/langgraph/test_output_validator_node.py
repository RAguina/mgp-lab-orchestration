# tests/langgraph/test_output_validator_node.py
from langchain_integration.langgraph.validators import output_validator_node
from langchain_integration.langgraph.agent_state import AgentState


def test_output_validator_retry_trigger():
    state = AgentState(
        input="Genera algo",
        output="Error: ocurrió un problema",
        task_type="",
        selected_model="",
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
    result = output_validator_node(state)
    assert result["retry"] is True
    assert result["retry_count"] == 1
