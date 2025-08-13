"""
Graph Builder - Construcción modular de grafos LangGraph
"""

import logging
from typing import Literal
from langgraph.graph import StateGraph, END
from langchain_core.runnables import RunnableLambda

# Logger específico para graph builder
graph_logger = logging.getLogger("orchestrator.graph_builder")

# Estado
from langchain_integration.langgraph.agent_state import AgentState

# Nodos
from langchain_integration.langgraph.nodes.task_analyzer_node import task_analyzer_node
from langchain_integration.langgraph.nodes.output_validator_node import (
    output_validator_node, route_after_validation
)
from langchain_integration.tools.history_tools import (
    HistoryReaderNode, should_include_history
)
from langchain_integration.langgraph.nodes.resource_monitor_node import resource_monitor_node
import langchain_integration.langgraph.nodes.execution_node as execution_mod
from langchain_integration.langgraph.nodes.summary_node import summary_node


def route_after_analysis(state: AgentState) -> Literal["monitor", "skip_monitor"]:
    """Ruta después del análisis de tarea."""
    task_type = state.get("task_type", "unknown")
    graph_logger.info(f"[ROUTING] After analysis task_type={task_type}")

    if task_type in ["code", "analysis"]:
        return "monitor"
    elif task_type == "chat":
        return "skip_monitor"
    else:
        return "monitor"


class GraphBuilder:
    """Builder para construir diferentes tipos de grafos de orquestación."""
    
    def __init__(self):
        self.registered_nodes = {
            "analyzer": task_analyzer_node,
            "monitor": resource_monitor_node,
            "executor": execution_mod.execution_node,
            "validator": output_validator_node,
            "history": HistoryReaderNode,
            "summarizer": summary_node
        }
        graph_logger.info(f"[BUILDER] Registered nodes: {list(self.registered_nodes.keys())}")
    
    def register_node(self, name: str, node_func):
        """Registra un nuevo nodo en el builder."""
        self.registered_nodes[name] = node_func
        graph_logger.info(f"[BUILDER] Registered custom node: {name}")
    
    def build_linear_flow_graph(self) -> StateGraph:
        """Construye el grafo lineal tradicional."""
        graph_logger.info("[BUILDER] Building linear flow graph")
        
        builder = StateGraph(AgentState)
        
        # Agregar nodos
        for name, fn in self.registered_nodes.items():
            if name == "history":
                builder.add_node(name, fn)
            else:
                builder.add_node(name, RunnableLambda(fn))
        
        # Entry point y edges
        builder.set_entry_point("analyzer")
        
        builder.add_conditional_edges("analyzer", route_after_analysis, {
            "monitor": "monitor",
            "skip_monitor": "executor",
        })
        
        builder.add_edge("monitor", "executor")
        builder.add_edge("executor", "validator")
        
        builder.add_conditional_edges("validator", route_after_validation, {
            "retry_execution": "executor",
            "continue": "history",
        })
        
        builder.add_conditional_edges("history", should_include_history, {
            "read_history": "summarizer",
            "skip_history": "summarizer",
        })
        
        builder.add_edge("summarizer", END)
        
        graph_logger.info("[BUILDER] Linear flow graph compiled")
        return builder.compile()
    
    def build_challenge_flow_graph(self) -> StateGraph:
        """Construye el grafo de challenge: Creator → Challenger → Refiner."""
        graph_logger.info("[BUILDER] Building challenge flow graph")
        
        try:
            from .graph_configs import get_challenge_flow_config
            config = get_challenge_flow_config()
            return self.build_graph_from_config(config)
        except Exception as e:
            graph_logger.error(f"[BUILDER] Challenge flow failed: {e}, using linear")
            return self.build_linear_flow_graph()
    
    def build_graph_from_config(self, config) -> StateGraph:
        """Construye un grafo basado en una configuración FlowConfig."""
        graph_logger.info(f"[BUILDER] Building {config.name} flow")
        
        builder = StateGraph(AgentState)
        
        # Agregar nodos
        for node_config in config.nodes:
            if node_config.type == "execution":
                node_func = self._create_configurable_execution_node(node_config)
            else:
                # Mapear tipos a nodos registrados
                node_type = {
                    "task_analyzer": "analyzer",
                    "resource_monitor": "monitor", 
                    "output_validator": "validator",
                    "history_reader": "history",
                    "summary": "summarizer"
                }.get(node_config.type, node_config.type)
                
                if node_type not in self.registered_nodes:
                    graph_logger.warning(f"[BUILDER] Node type '{node_type}' not found, skipping")
                    continue
                node_func = self.registered_nodes[node_type]
            
            # Agregar al grafo
            if node_config.id == "history":
                builder.add_node(node_config.id, node_func)
            else:
                builder.add_node(node_config.id, RunnableLambda(node_func))
        
        # Entry point y edges
        builder.set_entry_point(config.entry_point)
        
        for edge_config in config.edges:
            builder.add_edge(edge_config.source, edge_config.target)
        
        # Agregar END al último nodo
        terminal_nodes = self._find_terminal_nodes(config)
        if not terminal_nodes:
            terminal_nodes = [config.nodes[-1].id]  # Fallback al último nodo
            
        for node_id in terminal_nodes:
            builder.add_edge(node_id, END)
        
        graph_logger.info(f"[BUILDER] {config.name} flow compiled successfully")
        return builder.compile()
    
    def _create_configurable_execution_node(self, node_config):
            """Crea un nodo de ejecución configurable - ARREGLADO con estado estructurado."""
            def configurable_execution_node(state: AgentState) -> AgentState:
                graph_logger.info(f"[EXECUTION] Running {node_config.id}")
                
                try:
                    from .graph_configs import build_context_from_state, format_prompt_template
                    import json
                    import time
                    from pathlib import Path
                    
                    # Construir contexto y formatear prompt
                    context = build_context_from_state(state, node_config.id)
                    
                    if node_config.prompt_template:
                        try:
                            formatted_prompt = format_prompt_template(node_config.prompt_template, context)
                        except ValueError as e:
                            graph_logger.warning(f"[EXECUTION] Template error: {e}")
                            formatted_prompt = context.get("user_input", "")
                    else:
                        formatted_prompt = context.get("user_input", "")
                    
                    # Ejecutar con prompt formateado
                    modified_state = state.copy()
                    modified_state["input"] = formatted_prompt
                    modified_state[f"{node_config.id}_config"] = node_config.config or {}
                    
                    result_state = execution_mod.execution_node(modified_state)
                    
                    # 🔥 ARREGLO PRINCIPAL: Guardar en estructura de challenge_flow
                    if "challenge_flow" not in result_state:
                        result_state["challenge_flow"] = {}
                    
                    # Guardar output específico del nodo en estructura anidada
                    result_state["challenge_flow"][node_config.id] = {
                        "output": result_state.get("output", ""),
                        "timestamp": time.time(),
                        "node_type": node_config.type,
                        "config": node_config.config or {},
                        "prompt_used": formatted_prompt[:100] + "..." if len(formatted_prompt) > 100 else formatted_prompt
                    }
                    
                    # También mantener compatibilidad con el sistema anterior
                    result_state[f"{node_config.id}_output"] = result_state.get("output", "")
                    
                    # 💾 Guardar todo el challenge flow en JSON único
                    execution_id = result_state.get("request_id", f"challenge_{int(time.time())}")
                    challenge_data = {
                        "execution_id": execution_id,
                        "flow_type": "challenge",
                        "timestamp": time.time(),
                        "user_input": context.get("user_input", ""),
                        "nodes": result_state["challenge_flow"],
                        "final_output": result_state.get("output", "")
                    }
                    
                    # Guardar JSON único
                    try:
                        output_dir = Path("outputs/challenge_flows")
                        output_dir.mkdir(parents=True, exist_ok=True)
                        
                        json_path = output_dir / f"{execution_id}.json"
                        with open(json_path, "w", encoding="utf-8") as f:
                            json.dump(challenge_data, f, indent=2, ensure_ascii=False)
                        
                        graph_logger.info(f"[EXECUTION] Saved challenge flow to {json_path}")
                        result_state["challenge_flow_file"] = str(json_path)
                        
                    except Exception as e:
                        graph_logger.error(f"[EXECUTION] Failed to save challenge flow JSON: {e}")
                    
                    return result_state
                    
                except Exception as e:
                    graph_logger.error(f"[EXECUTION] Error in {node_config.id}: {e}")
                    error_state = state.copy()
                    error_state["output"] = f"Error in {node_config.id}: {str(e)}"
                    error_state[f"{node_config.id}_output"] = f"Error: {str(e)}"
                    return error_state
            
            return configurable_execution_node

    def _find_terminal_nodes(self, config) -> list:
        """Encuentra nodos terminales (sin edges salientes)."""
        if not config.edges:
            return [config.nodes[-1].id] if config.nodes else []
        
        all_sources = {edge.source for edge in config.edges}
        all_nodes = {node.id for node in config.nodes}
        
        # Nodos que no son source de ningún edge
        terminal_nodes = all_nodes - all_sources
        
        return list(terminal_nodes) if terminal_nodes else [config.nodes[-1].id]
    
    def build_graph(self, flow_type: str = "linear") -> StateGraph:
        """Método principal para construir grafos."""
        graph_logger.info(f"[BUILDER] Building {flow_type} flow")
        
        if flow_type == "linear":
            return self.build_linear_flow_graph()
        elif flow_type == "challenge":
            return self.build_challenge_flow_graph()
        else:
            graph_logger.warning(f"[BUILDER] Unknown flow type '{flow_type}', using linear")
            return self.build_linear_flow_graph()


# Singleton
_graph_builder_instance = None

def get_graph_builder() -> GraphBuilder:
    """Obtiene la instancia singleton del GraphBuilder."""
    global _graph_builder_instance
    if _graph_builder_instance is None:
        _graph_builder_instance = GraphBuilder()
    return _graph_builder_instance

def build_routing_graph(flow_type: str = "linear") -> StateGraph:
    """Función de conveniencia para compatibilidad."""
    builder = get_graph_builder()
    return builder.build_graph(flow_type)