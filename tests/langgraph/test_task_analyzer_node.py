# tests/langgraph/test_task_analyzer_node.py
import pytest
from langchain_integration.langgraph.routing_agent import task_analyzer_node, AgentState


def test_task_analyzer_node_detects_code_task():
    state = AgentState(
        input="Escribe una función en Python para sumar dos números",
        output="",
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
    result = task_analyzer_node(state)
    assert result["task_type"] == "code"
    assert "Modelo seleccionado:" in result["messages"][-1]