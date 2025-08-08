# orchestrator_debug_system.py
"""
Sistema completo de debugging y observabilidad para el orchestrator
Guarda logs, métricas, flows y permite análisis post-ejecución
"""

import json
import time
import os
import threading
from datetime import datetime
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict
import uuid

@dataclass
class NodeExecution:
    """Información de ejecución de un nodo"""
    node_id: str
    node_name: str
    start_time: float
    end_time: Optional[float] = None
    duration: Optional[float] = None
    success: bool = True
    input_state: Dict[str, Any] = None
    output_state: Dict[str, Any] = None
    error: Optional[str] = None
    metrics: Dict[str, Any] = None
    thinking_logs: List[Dict] = None
    decisions: List[Dict] = None

@dataclass
class OrchestratorExecution:
    """Información completa de una ejecución del orchestrator"""
    execution_id: str
    start_time: float
    end_time: Optional[float] = None
    total_duration: Optional[float] = None
    user_prompt: str = ""
    final_output: str = ""
    success: bool = True
    error: Optional[str] = None
    nodes_executed: List[NodeExecution] = None
    flow_path: List[str] = None
    final_metrics: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.nodes_executed is None:
            self.nodes_executed = []
        if self.flow_path is None:
            self.flow_path = []

class OrchestratorDebugger:
    """
    Sistema de debugging para el orchestrator con persistencia
    """
    
    def __init__(self, debug_dir: str = "debug_outputs"):
        self.debug_dir = debug_dir
        self.current_execution: Optional[OrchestratorExecution] = None
        self.executions_history: List[OrchestratorExecution] = []
        self._lock = threading.Lock()
        
        # Crear directorio de debug
        os.makedirs(debug_dir, exist_ok=True)
        os.makedirs(f"{debug_dir}/flows", exist_ok=True)
        os.makedirs(f"{debug_dir}/metrics", exist_ok=True)
        os.makedirs(f"{debug_dir}/logs", exist_ok=True)
        
        print(f"🔍 Orchestrator Debugger initialized - output dir: {debug_dir}")
    
    def start_execution(self, user_prompt: str) -> str:
        """Inicia tracking de una nueva ejecución"""
        execution_id = f"exec_{int(time.time())}_{str(uuid.uuid4())[:8]}"
        
        with self._lock:
            self.current_execution = OrchestratorExecution(
                execution_id=execution_id,
                start_time=time.time(),
                user_prompt=user_prompt
            )
        
        print(f"🚀 [DEBUG] Started execution: {execution_id}")
        print(f"📝 [DEBUG] Prompt: {user_prompt[:100]}...")
        
        return execution_id
    
    def start_node(self, node_name: str, input_state: Dict[str, Any]) -> str:
        """Inicia tracking de un nodo"""
        if not self.current_execution:
            return ""
        
        node_id = f"{node_name}_{int(time.time())}"
        
        # Limpiar input_state para logging (remover objetos grandes)
        clean_input = self._clean_state_for_logging(input_state)
        
        node_exec = NodeExecution(
            node_id=node_id,
            node_name=node_name,
            start_time=time.time(),
            input_state=clean_input,
            thinking_logs=[],
            decisions=[]
        )
        
        with self._lock:
            self.current_execution.nodes_executed.append(node_exec)
            self.current_execution.flow_path.append(node_name)
        
        print(f"🌀 [DEBUG] Node started: {node_name} ({node_id})")
        
        return node_id
    
    def log_thinking(self, node_id: str, thinking_data: Dict[str, Any]):
        """Log proceso de pensamiento de un nodo"""
        if not self.current_execution:
            return
        
        with self._lock:
            for node_exec in self.current_execution.nodes_executed:
                if node_exec.node_id == node_id:
                    node_exec.thinking_logs.append({
                        "timestamp": time.time(),
                        "data": thinking_data
                    })
                    break
    
    def log_decision(self, node_id: str, decision_data: Dict[str, Any]):
        """Log decisión de un nodo"""
        if not self.current_execution:
            return
        
        with self._lock:
            for node_exec in self.current_execution.nodes_executed:
                if node_exec.node_id == node_id:
                    node_exec.decisions.append({
                        "timestamp": time.time(),
                        "data": decision_data
                    })
                    break
    
    def complete_node(self, node_id: str, output_state: Dict[str, Any], 
                     success: bool = True, error: str = None, metrics: Dict[str, Any] = None):
        """Completa tracking de un nodo"""
        if not self.current_execution:
            return
        
        end_time = time.time()
        clean_output = self._clean_state_for_logging(output_state)
        
        with self._lock:
            for node_exec in self.current_execution.nodes_executed:
                if node_exec.node_id == node_id:
                    node_exec.end_time = end_time
                    node_exec.duration = end_time - node_exec.start_time
                    node_exec.success = success
                    node_exec.output_state = clean_output
                    node_exec.error = error
                    node_exec.metrics = metrics or {}
                    
                    status = "✅" if success else "❌"
                    print(f"{status} [DEBUG] Node completed: {node_exec.node_name} ({node_exec.duration:.3f}s)")
                    break
    
    def complete_execution(self, final_output: str = "", success: bool = True, 
                          error: str = None, final_metrics: Dict[str, Any] = None):
        """Completa tracking de la ejecución"""
        if not self.current_execution:
            return
        
        end_time = time.time()
        
        with self._lock:
            self.current_execution.end_time = end_time
            self.current_execution.total_duration = end_time - self.current_execution.start_time
            self.current_execution.final_output = final_output
            self.current_execution.success = success
            self.current_execution.error = error
            self.current_execution.final_metrics = final_metrics or {}
            
            # Guardar a historial
            self.executions_history.append(self.current_execution)
            
            # Persistir
            self._save_execution(self.current_execution)
            
            status = "✅" if success else "❌"
            print(f"{status} [DEBUG] Execution completed: {self.current_execution.execution_id} ({self.current_execution.total_duration:.3f}s)")
            
            execution = self.current_execution
            self.current_execution = None
            
            return execution
    
    def _clean_state_for_logging(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Limpia el state para logging removiendo objetos pesados"""
        clean_state = {}
        
        for key, value in state.items():
            try:
                if key in ['langgraph_logger', 'model', 'tokenizer', 'pipeline_obj']:
                    clean_state[key] = f"<{type(value).__name__}>"
                elif isinstance(value, str):
                    clean_state[key] = value[:500] + "..." if len(value) > 500 else value
                elif isinstance(value, (int, float, bool, type(None))):
                    clean_state[key] = value
                elif isinstance(value, (list, tuple)):
                    clean_state[key] = [str(item)[:100] for item in value[:10]]
                elif isinstance(value, dict):
                    clean_state[key] = {k: str(v)[:100] for k, v in list(value.items())[:10]}
                else:
                    clean_state[key] = str(value)[:100]
            except:
                clean_state[key] = f"<{type(value).__name__}>"
        
        return clean_state
    
    def _save_execution(self, execution: OrchestratorExecution):
        """Guarda una ejecución a disco"""
        try:
            # Convertir a dict serializable
            exec_dict = asdict(execution)
            
            # Guardar JSON detallado
            timestamp = datetime.fromtimestamp(execution.start_time).strftime("%Y%m%d_%H%M%S")
            filename = f"{self.debug_dir}/flows/execution_{timestamp}_{execution.execution_id}.json"
            
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(exec_dict, f, indent=2, default=str, ensure_ascii=False)
            
            # Guardar resumen
            summary = {
                "execution_id": execution.execution_id,
                "timestamp": timestamp,
                "prompt": execution.user_prompt[:200],
                "success": execution.success,
                "duration": execution.total_duration,
                "nodes_count": len(execution.nodes_executed),
                "flow_path": execution.flow_path,
                "output_length": len(execution.final_output) if execution.final_output else 0,
                "error": execution.error
            }
            
            summary_file = f"{self.debug_dir}/executions_summary.jsonl"
            with open(summary_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(summary, ensure_ascii=False) + '\n')
            
            print(f"💾 [DEBUG] Execution saved: {filename}")
            
        except Exception as e:
            print(f"❌ [DEBUG] Failed to save execution: {e}")
    
    def get_latest_execution(self) -> Optional[OrchestratorExecution]:
        """Obtiene la última ejecución"""
        return self.executions_history[-1] if self.executions_history else None
    
    def analyze_execution(self, execution_id: str = None) -> Dict[str, Any]:
        """Analiza una ejecución específica o la última"""
        if execution_id:
            execution = next((e for e in self.executions_history if e.execution_id == execution_id), None)
        else:
            execution = self.get_latest_execution()
        
        if not execution:
            return {"error": "No execution found"}
        
        # Análisis detallado
        total_nodes = len(execution.nodes_executed)
        successful_nodes = sum(1 for node in execution.nodes_executed if node.success)
        failed_nodes = total_nodes - successful_nodes
        
        # Análisis de timing
        node_timings = []
        for node in execution.nodes_executed:
            if node.duration:
                node_timings.append({
                    "node": node.node_name,
                    "duration": node.duration,
                    "percentage": (node.duration / execution.total_duration * 100) if execution.total_duration else 0
                })
        
        # Ordenar por duración
        node_timings.sort(key=lambda x: x["duration"], reverse=True)
        
        analysis = {
            "execution_id": execution.execution_id,
            "overall": {
                "success": execution.success,
                "total_duration": execution.total_duration,
                "prompt_length": len(execution.user_prompt),
                "output_length": len(execution.final_output) if execution.final_output else 0
            },
            "nodes": {
                "total": total_nodes,
                "successful": successful_nodes,
                "failed": failed_nodes,
                "success_rate": (successful_nodes / total_nodes * 100) if total_nodes > 0 else 0
            },
            "flow_path": execution.flow_path,
            "timing_analysis": {
                "slowest_nodes": node_timings[:3],
                "fastest_nodes": node_timings[-3:],
                "bottleneck": node_timings[0]["node"] if node_timings else None
            },
            "errors": [
                {"node": node.node_name, "error": node.error} 
                for node in execution.nodes_executed if node.error
            ]
        }
        
        return analysis

# Instancia global del debugger
orchestrator_debugger = OrchestratorDebugger()

# Funciones de conveniencia para usar en los nodos
def start_execution_debug(user_prompt: str) -> str:
    """Función para iniciar debug desde el orchestrator"""
    return orchestrator_debugger.start_execution(user_prompt)

def start_node_debug(node_name: str, input_state: Dict[str, Any]) -> str:
    """Función para iniciar debug de nodo"""
    return orchestrator_debugger.start_node(node_name, input_state)

def complete_node_debug(node_id: str, output_state: Dict[str, Any], 
                       success: bool = True, error: str = None, metrics: Dict[str, Any] = None):
    """Función para completar debug de nodo"""
    return orchestrator_debugger.complete_node(node_id, output_state, success, error, metrics)

def complete_execution_debug(final_output: str = "", success: bool = True, 
                           error: str = None, final_metrics: Dict[str, Any] = None):
    """Función para completar debug de ejecución"""
    return orchestrator_debugger.complete_execution(final_output, success, error, final_metrics)

def analyze_latest_execution() -> Dict[str, Any]:
    """Función para analizar la última ejecución"""
    return orchestrator_debugger.analyze_execution()

# Test function
if __name__ == "__main__":
    # Demo del sistema de debugging
    print("🔍 Testing Orchestrator Debugger...")
    
    # Simular ejecución
    exec_id = start_execution_debug("Test prompt")
    
    # Simular nodos
    node1_id = start_node_debug("task_analyzer", {"input": "test"})
    time.sleep(0.1)
    complete_node_debug(node1_id, {"task_type": "code"}, success=True, metrics={"duration": 0.1})
    
    node2_id = start_node_debug("executor", {"model": "mistral7b"})
    time.sleep(0.2)
    complete_node_debug(node2_id, {"output": "result"}, success=True, metrics={"duration": 0.2})
    
    # Completar ejecución
    complete_execution_debug("Final output", success=True, final_metrics={"total_time": 0.3})
    
    # Analizar
    analysis = analyze_latest_execution()
    print("📊 Analysis:", json.dumps(analysis, indent=2))
    
    print("✅ Debugger test completed!")