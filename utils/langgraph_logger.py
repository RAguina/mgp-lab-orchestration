# utils/langgraph_logger.py
import functools
import time
import json
import uuid
from typing import Callable, Dict, Any, Optional
from datetime import datetime
from .logger import StructuredLogger

class LangGraphNodeLogger:
    """
    Logger especializado para nodos de LangGraph con observabilidad completa
    """
    
    def __init__(self, orchestrator_id: str, log_dir: str = "logs", enable_file_logging: bool = False):
        self.orchestrator_id = orchestrator_id
        self.enable_file_logging = enable_file_logging
        self.logger = StructuredLogger(
            f"orchestrator_{orchestrator_id}", 
            log_dir, 
            orchestrator_id,
            enable_file_logging
        )
        self.execution_flow = []
        self.start_time = time.time()
        
    def log_node_start(self, node_name: str, state: Dict[str, Any]) -> str:
        """Log inicio de ejecución de un nodo"""
        node_execution_id = str(uuid.uuid4())[:8]
        
        # Extraer información relevante del state sin logging spam
        state_summary = self._summarize_state(state)
        
        node_data = {
            "node_execution_id": node_execution_id,
            "node_name": node_name,
            "orchestrator_id": self.orchestrator_id,
            "state_summary": state_summary,
            "execution_order": len(self.execution_flow) + 1
        }
        
        self.logger.info("node_execution_start", **node_data)
        
        # Añadir al flujo de ejecución
        self.execution_flow.append({
            "node_execution_id": node_execution_id,
            "node_name": node_name,
            "start_time": time.time(),
            "state_input": state_summary
        })
        
        return node_execution_id
    
    def log_node_thinking(self, node_execution_id: str, thinking_process: Dict[str, Any]):
        """Log proceso de pensamiento del nodo"""
        self.logger.info("node_thinking", 
                        node_execution_id=node_execution_id,
                        thinking=thinking_process)
    
    def log_node_decision(self, node_execution_id: str, decision: Dict[str, Any]):
        """Log decisión tomada por el nodo"""
        self.logger.info("node_decision",
                        node_execution_id=node_execution_id,
                        decision=decision)
    
    def log_node_complete(self, node_execution_id: str, state: Dict[str, Any], 
                         success: bool = True, error: Optional[str] = None):
        """Log finalización de ejecución de un nodo"""
        end_time = time.time()
        
        # Encontrar el nodo en el flujo
        for i, node in enumerate(self.execution_flow):
            if node["node_execution_id"] == node_execution_id:
                duration = end_time - node["start_time"]
                state_summary = self._summarize_state(state)
                
                completion_data = {
                    "node_execution_id": node_execution_id,
                    "node_name": node["node_name"],
                    "duration_seconds": round(duration, 3),
                    "success": success,
                    "state_output": state_summary
                }
                
                if error:
                    completion_data["error"] = error
                    self.logger.error("node_execution_error", **completion_data)
                else:
                    self.logger.info("node_execution_complete", **completion_data)
                
                # Actualizar flujo de ejecución
                self.execution_flow[i].update({
                    "end_time": end_time,
                    "duration": duration,
                    "success": success,
                    "state_output": state_summary,
                    "error": error
                })
                break
    
    def log_orchestrator_complete(self, final_state: Dict[str, Any], success: bool = True):
        """Log finalización completa del orchestrator"""
        total_duration = time.time() - self.start_time
        
        summary = {
            "orchestrator_id": self.orchestrator_id,
            "total_duration_seconds": round(total_duration, 3),
            "nodes_executed": len(self.execution_flow),
            "successful_nodes": sum(1 for node in self.execution_flow if node.get("success", True)),
            "overall_success": success,
            "final_state": self._summarize_state(final_state)
        }
        
        self.logger.info("orchestrator_complete", **summary)
        
        return {
            "execution_flow": self.execution_flow,
            "summary": summary
        }
    
    def _summarize_state(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Crea un resumen del state para logging sin spam"""
        summary = {}
        
        for key, value in state.items():
            if isinstance(value, str):
                # Truncar strings largos
                summary[key] = value[:200] + "..." if len(value) > 200 else value
            elif isinstance(value, (int, float, bool)):
                summary[key] = value
            elif isinstance(value, dict):
                # Para dicts, solo keys y primer nivel
                summary[key] = {
                    "_type": "dict",
                    "_keys": list(value.keys()),
                    "_sample": {k: str(v)[:50] for k, v in list(value.items())[:3]}
                }
            elif isinstance(value, list):
                summary[key] = {
                    "_type": "list", 
                    "_length": len(value),
                    "_sample": [str(item)[:50] for item in value[:3]]
                }
            else:
                summary[key] = {
                    "_type": type(value).__name__,
                    "_str": str(value)[:100]
                }
        
        return summary

def langgraph_node_logger(node_name: str, enable_detailed_logging: bool = False):
    """
    Decorator para logging automático de nodos LangGraph
    Combina el decorator simple con logging estructurado
    """
    def decorator(func: Callable):
        @functools.wraps(func)
        def wrapper(state, *args, **kwargs):
            # Obtener o crear logger del orchestrator
            orchestrator_id = state.get('orchestrator_id', str(uuid.uuid4())[:8])
            
            if 'langgraph_logger' not in state:
                state['langgraph_logger'] = LangGraphNodeLogger(
                    orchestrator_id, 
                    enable_file_logging=enable_detailed_logging
                )
            
            logger = state['langgraph_logger']
            
            # Log inicio (console + structured)
            start_time = time.time()
            print(f"🌀 [{node_name}] Inicio - {str(state.get('input', ''))[:40]}...")
            
            node_execution_id = logger.log_node_start(node_name, state)
            
            try:
                # Ejecutar función del nodo
                result = func(state, *args, **kwargs)
                
                # Log finalización exitosa
                duration = round(time.time() - start_time, 3)
                print(f"✅ [{node_name}] Fin ({duration}s)")
                
                logger.log_node_complete(node_execution_id, result, success=True)
                
                return result
                
            except Exception as e:
                # Log error
                duration = round(time.time() - start_time, 3)
                print(f"❌ [{node_name}] Error ({duration}s): {str(e)}")
                
                logger.log_node_complete(node_execution_id, state, success=False, error=str(e))
                
                raise  # Re-raise la excepción
        
        return wrapper
    return decorator

def langgraph_thinking_logger(node_execution_id: str, state: Dict[str, Any]):
    """
    Helper function para logging de proceso de pensamiento
    """
    if 'langgraph_logger' in state:
        def log_thinking(**thinking_data):
            state['langgraph_logger'].log_node_thinking(node_execution_id, thinking_data)
        return log_thinking
    else:
        # Si no hay logger, return dummy function
        def dummy_log(**kwargs):
            pass
        return dummy_log

def langgraph_decision_logger(node_execution_id: str, state: Dict[str, Any]):
    """
    Helper function para logging de decisiones
    """
    if 'langgraph_logger' in state:
        def log_decision(**decision_data):
            state['langgraph_logger'].log_node_decision(node_execution_id, decision_data)
        return log_decision
    else:
        def dummy_log(**kwargs):
            pass
        return dummy_log

# Ejemplo de uso en un nodo LangGraph
@langgraph_node_logger("task_analyzer", enable_detailed_logging=True)
def task_analyzer_node(state):
    """
    Ejemplo de nodo con logging detallado del proceso de pensamiento
    """
    user_prompt = state.get('user_prompt', '')
    
    # Log pensamiento
    thinking = langgraph_thinking_logger("", state)  # El ID se maneja automáticamente
    thinking(
        step="analyzing_keywords",
        prompt_length=len(user_prompt),
        detected_keywords=["function", "Python"]  # Ejemplo
    )
    
    # Simulación de análisis
    task_type = "code_with_explanation"  # Lógica de análisis aquí
    
    # Log decisión
    decision = langgraph_decision_logger("", state)
    decision(
        decision_type="task_classification",
        chosen_type=task_type,
        confidence=0.87,
        reasoning="Detected code request with explanation requirement"
    )
    
    return {
        **state,
        "task_type": task_type,
        "complexity_level": "medium",
        "estimated_duration": 45,
        "confidence_score": 0.87
    }

# Utilities para análisis de logs
class LangGraphLogAnalyzer:
    """
    Herramientas para analizar logs de ejecución de LangGraph
    """
    
    @staticmethod
    def analyze_orchestrator_run(orchestrator_id: str, log_dir: str = "logs") -> Dict[str, Any]:
        """Analiza una ejecución completa del orchestrator"""
        from .logger import LogAnalyzer
        import glob
        import os
        
        # Buscar logs del orchestrator
        pattern = os.path.join(log_dir, f"orchestrator_{orchestrator_id}_*.log")
        log_files = glob.glob(pattern)
        
        if not log_files:
            return {"error": f"No logs found for orchestrator {orchestrator_id}"}
        
        all_entries = []
        for log_file in log_files:
            entries = LogAnalyzer.parse_log_file(log_file)
            all_entries.extend(entries)
        
        # Analizar flujo de ejecución
        nodes_executed = []
        thinking_logs = []
        decisions_made = []
        
        for entry in all_entries:
            event = entry.get("event", "")
            
            if event == "node_execution_complete":
                nodes_executed.append({
                    "node_name": entry.get("node_name"),
                    "duration": entry.get("duration_seconds"),
                    "success": entry.get("success")
                })
            
            elif event == "node_thinking":
                thinking_logs.append(entry.get("thinking", {}))
            
            elif event == "node_decision":
                decisions_made.append(entry.get("decision", {}))
        
        analysis = {
            "orchestrator_id": orchestrator_id,
            "total_entries": len(all_entries),
            "nodes_executed": len(nodes_executed),
            "successful_nodes": sum(1 for node in nodes_executed if node["success"]),
            "thinking_logs_count": len(thinking_logs),
            "decisions_made_count": len(decisions_made),
            "execution_flow": nodes_executed
        }
        
        if nodes_executed:
            total_duration = sum(node["duration"] for node in nodes_executed if node["duration"])
            analysis["total_execution_time"] = round(total_duration, 3)
            analysis["average_node_time"] = round(total_duration / len(nodes_executed), 3)
        
        return analysis
    
    @staticmethod
    def get_thinking_trace(orchestrator_id: str, log_dir: str = "logs") -> list:
        """Extrae la traza completa de pensamiento de una ejecución"""
        from .logger import LogAnalyzer
        import glob
        import os
        
        pattern = os.path.join(log_dir, f"orchestrator_{orchestrator_id}_*.log")
        log_files = glob.glob(pattern)
        
        thinking_trace = []
        
        for log_file in log_files:
            entries = LogAnalyzer.parse_log_file(log_file)
            for entry in entries:
                if entry.get("event") in ["node_thinking", "node_decision"]:
                    thinking_trace.append({
                        "timestamp": entry.get("timestamp"),
                        "node_execution_id": entry.get("node_execution_id"),
                        "event": entry.get("event"),
                        "data": entry.get("thinking") or entry.get("decision", {})
                    })
        
        # Ordenar por timestamp
        thinking_trace.sort(key=lambda x: x["timestamp"])
        
        return thinking_trace

# Test function
if __name__ == "__main__":
    # Demo del sistema de logging
    import tempfile
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Simular estado inicial
        initial_state = {
            "user_prompt": "Write a Python function to sort a list and explain the algorithm",
            "orchestrator_id": "test_orch_001"
        }
        
        # Ejecutar nodo con logging
        result = task_analyzer_node(initial_state)
        
        print("✅ Test completado")
        print(f"📊 Resultado: {result}")
        
        # Analizar logs generados
        if result.get('langgraph_logger'):
            summary = result['langgraph_logger'].log_orchestrator_complete(result)
            print(f"📋 Resumen: {summary}")