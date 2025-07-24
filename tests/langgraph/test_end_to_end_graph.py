#test_end_to_end_graph
from langchain_integration.langgraph.routing_agent import build_routing_graph


def test_routing_graph_builds():
    graph = build_routing_graph()
    assert graph is not None