"""
Graph Configurations - Definiciones de diferentes tipos de flujos
"""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass

@dataclass
class NodeConfig:
    """Configuración de un nodo individual."""
    id: str
    type: str
    name: str
    prompt_template: Optional[str] = None
    model_hint: Optional[str] = None  # Sugerencia de modelo a usar
    config: Optional[Dict[str, Any]] = None

@dataclass 
class EdgeConfig:
    """Configuración de una conexión entre nodos."""
    source: str
    target: str
    condition: Optional[str] = None  # Para routing condicional

@dataclass
class FlowConfig:
    """Configuración completa de un flujo."""
    name: str
    description: str
    nodes: List[NodeConfig]
    edges: List[EdgeConfig]
    entry_point: str

# ============================================================================
# FLOW CONFIGURATIONS
# ============================================================================

def get_linear_flow_config() -> FlowConfig:
    """Configuración del flujo lineal tradicional."""
    return FlowConfig(
        name="linear",
        description="Flujo tradicional: análisis → monitor → ejecución → validación",
        nodes=[
            NodeConfig("analyzer", "task_analyzer", "Task Analyzer"),
            NodeConfig("monitor", "resource_monitor", "Resource Monitor"),
            NodeConfig("executor", "execution", "Model Executor"),
            NodeConfig("validator", "output_validator", "Output Validator"),
            NodeConfig("history", "history_reader", "History Reader"),
            NodeConfig("summarizer", "summary", "Summarizer")
        ],
        edges=[
            EdgeConfig("analyzer", "monitor", "route_after_analysis"),
            EdgeConfig("analyzer", "executor", "route_after_analysis"),  # skip_monitor case
            EdgeConfig("monitor", "executor"),
            EdgeConfig("executor", "validator"),
            EdgeConfig("validator", "history", "route_after_validation"),
            EdgeConfig("validator", "executor", "route_after_validation"),  # retry case
            EdgeConfig("history", "summarizer", "should_include_history"),
        ],
        entry_point="analyzer"
    )

def get_challenge_flow_config() -> FlowConfig:
    """
    Configuración del flujo de challenge/debate entre modelos.
    Implementa el patrón: Creator → Challenger → Refiner
    """
    return FlowConfig(
        name="challenge",
        description="Flujo de debate: creador → crítico → refinador",
        nodes=[
            NodeConfig(
                id="creator",
                type="execution", 
                name="Creator",
                prompt_template="Genera una solución para: {user_input}",
                model_hint="qwen",  # Modelo creativo
                config={"role": "creator", "temperature": 0.7}
            ),
            NodeConfig(
                id="challenger", 
                type="execution",
                name="Challenger",
                prompt_template=(
                    "Analiza críticamente esta solución y encuentra problemas potenciales:\n\n"
                    "SOLUCIÓN PROPUESTA:\n{previous_output}\n\n"
                    "Pregúntate:\n"
                    "- ¿Hay problemas de seguridad?\n"
                    "- ¿Está bien estructurado?\n"
                    "- ¿Falta algo importante?\n"
                    "- ¿Hay duplicación de lógica?\n\n"
                    "Proporciona 2-3 críticas específicas y constructivas."
                ),
                model_hint="mistral",  # Modelo crítico
                config={"role": "challenger", "temperature": 0.3}
            ),
            NodeConfig(
                id="refiner",
                type="execution", 
                name="Refiner",
                prompt_template=(
                    "Mejora esta solución basándote en las críticas recibidas:\n\n"
                    "CRÍTICAS RECIBIDAS:\n{previous_output}\n\n"
                    "Tu tarea es crear una versión mejorada que resuelva los problemas identificados. "
                    "Genera una solución completa y mejorada que incorpore las sugerencias de las críticas."
                ),
                model_hint="claude",  # Modelo refinador
                config={"role": "refiner", "temperature": 0.5}
            )
        ],
        edges=[
            EdgeConfig("creator", "challenger"),
            EdgeConfig("challenger", "refiner")
        ],
        entry_point="creator"
    )
def get_multi_perspective_flow_config() -> FlowConfig:
    """
    Configuración de flujo multi-perspectiva.
    Múltiples expertos analizan en paralelo, luego un sintetizador combina.
    """
    return FlowConfig(
        name="multi_perspective", 
        description="Múltiples expertos → síntesis final",
        nodes=[
            NodeConfig("splitter", "splitter", "Task Splitter"),
            NodeConfig(
                "security_expert",
                "execution",
                "Security Expert", 
                prompt_template="Analiza desde perspectiva de SEGURIDAD: {user_input}",
                model_hint="claude"
            ),
            NodeConfig(
                "performance_expert", 
                "execution",
                "Performance Expert",
                prompt_template="Analiza desde perspectiva de PERFORMANCE: {user_input}",
                model_hint="mistral"
            ),
            NodeConfig(
                "usability_expert",
                "execution", 
                "UX Expert",
                prompt_template="Analiza desde perspectiva de USABILIDAD: {user_input}",
                model_hint="qwen"
            ),
            NodeConfig(
                "synthesizer",
                "execution",
                "Synthesizer",
                prompt_template=(
                    "Sintetiza las siguientes perspectivas en una solución integral:\n\n"
                    "SEGURIDAD: {security_output}\n\n"
                    "PERFORMANCE: {performance_output}\n\n" 
                    "USABILIDAD: {usability_output}\n\n"
                    "Crea una solución que balancee todas estas consideraciones."
                ),
                model_hint="claude"
            )
        ],
        edges=[
            EdgeConfig("splitter", "security_expert"),
            EdgeConfig("splitter", "performance_expert"), 
            EdgeConfig("splitter", "usability_expert"),
            EdgeConfig("security_expert", "synthesizer"),
            EdgeConfig("performance_expert", "synthesizer"),
            EdgeConfig("usability_expert", "synthesizer")
        ],
        entry_point="splitter"
    )

# ============================================================================
# REGISTRY DE CONFIGURACIONES
# ============================================================================

FLOW_CONFIGS = {
    "linear": get_linear_flow_config,
    "challenge": get_challenge_flow_config,
    "multi_perspective": get_multi_perspective_flow_config,
}

def get_flow_config(flow_type: str) -> FlowConfig:
    """
    Obtiene la configuración para un tipo de flujo.
    
    Args:
        flow_type: Tipo de flujo ("linear", "challenge", "multi_perspective")
    
    Returns:
        FlowConfig correspondiente
    
    Raises:
        ValueError: Si el flow_type no existe
    """
    if flow_type not in FLOW_CONFIGS:
        available = list(FLOW_CONFIGS.keys())
        raise ValueError(f"Flow type '{flow_type}' not found. Available: {available}")
    
    return FLOW_CONFIGS[flow_type]()

def list_available_flows() -> List[str]:
    """Lista todos los tipos de flujo disponibles."""
    return list(FLOW_CONFIGS.keys())

def get_flow_description(flow_type: str) -> str:
    """Obtiene la descripción de un tipo de flujo."""
    try:
        config = get_flow_config(flow_type)
        return config.description
    except ValueError:
        return f"Unknown flow type: {flow_type}"

# ============================================================================
# UTILIDADES PARA TEMPLATES
# ============================================================================

def format_prompt_template(template: str, context: Dict[str, Any]) -> str:
    """
    Formatea un template de prompt con el contexto dado.
    
    Args:
        template: Template string con placeholders {variable}
        context: Diccionario con valores para reemplazar
    
    Returns:
        Prompt formateado
    """
    try:
        return template.format(**context)
    except KeyError as e:
        missing_key = str(e).strip("'")
        raise ValueError(f"Missing context key for template: {missing_key}")

def build_context_from_state(state: Dict[str, Any], node_id: str) -> Dict[str, Any]:
    """
    Construye el contexto para un nodo basado en el estado actual.
    
    Args:
        state: Estado actual del flujo
        node_id: ID del nodo que necesita el contexto
    
    Returns:
        Diccionario con contexto para el nodo
    """
    context = {
        "user_input": state.get("input", ""),
        "task_type": state.get("task_type", "unknown"),
    }
    
    # Agregar outputs de nodos anteriores
    for key, value in state.items():
        if key.endswith("_output"):
            context[key] = value
    
    # Agregar output anterior genérico
    if "output" in state:
        context["previous_output"] = state["output"]  # ← Solo el último
    
    return context