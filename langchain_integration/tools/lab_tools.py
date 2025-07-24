# langchain_integration/tools/lab_tools.py
"""
Herramientas reutilizables para el laboratorio de modelos
Semana 2: Abstracciones y Herramientas
"""

import os
import json
import torch
from typing import Dict, List, Optional, Any, ClassVar
from datetime import datetime
from pydantic import BaseModel, Field

from langchain_core.tools import BaseTool
from langchain.tools import tool

# Imports del lab
try:
    from utils.gpu_guard import get_gpu_info, clear_gpu_memory
    from local_models.llm_launcher import MODELS
except ImportError:
    # Fallback si no están disponibles
    def get_gpu_info():
        return {"cuda": False, "message": "GPU info not available"}
    MODELS = {}


class VRAMMonitorTool(BaseTool):
    """Monitorea el estado de la VRAM y GPU"""
    name: str = "vram_monitor"
    description: str = "Obtiene información detallada sobre el uso de VRAM y estado de la GPU"
    
    def _run(self, query: str = "") -> str:
        """Ejecuta el monitoreo de VRAM"""
        try:
            gpu_info = get_gpu_info()
            
            if not gpu_info.get("cuda"):
                return "No GPU CUDA disponible"
            
            # Formatear información relevante
            status = f"""
Estado GPU:
- Dispositivo: {gpu_info.get('device', 'Unknown')}
- VRAM Total: {gpu_info.get('total_gb', 0):.1f} GB
- VRAM Usada: {gpu_info.get('allocated_gb', 0):.1f} GB
- VRAM Libre: {gpu_info.get('free_gb', 0):.1f} GB
- Utilización: {gpu_info.get('utilization_pct', 0):.1f}%
- Estado: {gpu_info.get('memory_status', 'unknown')}
"""
            return status.strip()
        except Exception as e:
            return f"Error monitoreando GPU: {str(e)}"
    
    async def _arun(self, query: str = "") -> str:
        """Versión asíncrona (usa la síncrona por ahora)"""
        return self._run(query)


class FileSearchTool(BaseTool):
    """Busca archivos en las carpetas del laboratorio"""
    name: str = "file_search"
    description: str = "Busca archivos en outputs, logs o metrics por patrón o fecha"
    
    def _run(self, query: str) -> str:
        """
        Busca archivos según el query
        Ejemplos: "outputs mistral", "logs today", "metrics json"
        """
        parts = query.lower().split()
        if not parts:
            return "Por favor especifica qué buscar (ej: 'outputs mistral')"
        
        # Determinar carpeta base
        base_dirs = {
            "outputs": "outputs",
            "logs": "logs", 
            "metrics": "metrics"
        }
        
        target_dir = None
        for key, path in base_dirs.items():
            if key in parts:
                target_dir = path
                parts.remove(key)
                break
        
        if not target_dir:
            target_dir = "outputs"  # Default
        
        # Buscar archivos
        results = []
        pattern = " ".join(parts) if parts else ""
        
        try:
            for root, dirs, files in os.walk(target_dir):
                for file in files:
                    if not pattern or pattern in file.lower():
                        file_path = os.path.join(root, file)
                        size = os.path.getsize(file_path) / 1024  # KB
                        mod_time = datetime.fromtimestamp(os.path.getmtime(file_path))
                        results.append(f"- {file_path} ({size:.1f}KB, {mod_time.strftime('%Y-%m-%d %H:%M')})")
            
            if results:
                return f"Archivos encontrados en {target_dir}:\n" + "\n".join(results[:10])
            else:
                return f"No se encontraron archivos con '{pattern}' en {target_dir}"
                
        except Exception as e:
            return f"Error buscando archivos: {str(e)}"
    
    async def _arun(self, query: str) -> str:
        return self._run(query)


class ModelProfilerTool(BaseTool):
    """Analiza el rendimiento de los modelos desde las métricas"""
    name: str = "model_profiler"
    description: str = "Analiza métricas de rendimiento de los modelos (tiempo de carga, inferencia, uso de memoria)"
    
    def _run(self, model_key: str = "") -> str:
        """Analiza métricas del modelo especificado o todos"""
        try:
            metrics_dir = "metrics"
            if not os.path.exists(metrics_dir):
                return "No hay directorio de métricas disponible"
            
            # Si se especifica un modelo, buscar solo ese
            if model_key and model_key in MODELS:
                metrics_file = os.path.join(metrics_dir, model_key, f"{model_key}_metrics.csv")
                if os.path.exists(metrics_file):
                    return self._analyze_metrics_file(metrics_file, model_key)
                else:
                    return f"No hay métricas para {model_key}"
            
            # Analizar todos los modelos
            results = []
            for model in MODELS.keys():
                metrics_file = os.path.join(metrics_dir, model, f"{model}_metrics.csv")
                if os.path.exists(metrics_file):
                    results.append(self._analyze_metrics_file(metrics_file, model))
            
            return "\n\n".join(results) if results else "No hay métricas disponibles"
            
        except Exception as e:
            return f"Error analizando métricas: {str(e)}"
    
    def _analyze_metrics_file(self, file_path: str, model_key: str) -> str:
        """Analiza un archivo de métricas específico"""
        try:
            import csv
            
            load_times = []
            infer_times = []
            memory_usage = []
            
            with open(file_path, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if 'load_time_sec' in row:
                        load_times.append(float(row.get('load_time_sec', 0)))
                    if 'infer_time_sec' in row:
                        infer_times.append(float(row.get('infer_time_sec', 0)))
                    if 'gpu_memory_used_gb' in row:
                        memory_usage.append(float(row.get('gpu_memory_used_gb', 0)))
            
            # Calcular estadísticas
            def avg(lst): return sum(lst) / len(lst) if lst else 0
            
            return f"""
Perfil de {model_key}:
- Ejecuciones totales: {len(load_times)}
- Tiempo de carga promedio: {avg(load_times):.1f}s
- Tiempo de inferencia promedio: {avg(infer_times):.1f}s
- Uso de memoria promedio: {avg(memory_usage):.1f}GB
- Última ejecución: {datetime.fromtimestamp(os.path.getmtime(file_path)).strftime('%Y-%m-%d %H:%M')}
""".strip()
        except Exception as e:
            return f"Error leyendo métricas de {model_key}: {str(e)}"
    
    async def _arun(self, model_key: str = "") -> str:
        return self._run(model_key)


class ModelSelectorTool(BaseTool):
    """Selecciona el mejor modelo según el tipo de tarea"""
    name: str = "model_selector"
    description: str = "Selecciona el modelo más apropiado según el tipo de tarea (código, chat, análisis)"
    
    # FIXED: Agregar ClassVar annotation para Pydantic v2
    TASK_MAPPING: ClassVar[Dict[str, List[str]]] = {
        "code": ["deepseek-coder-6.7b", "deepseek7b"],
        "chat": ["llama3", "mistral7b"],
        "analysis": ["mistral7b", "llama3"],
        "creative": ["llama3", "mistral7b"],
        "technical": ["deepseek7b", "deepseek-coder-6.7b"]
    }
    
    def _run(self, task_description: str) -> str:
        """Selecciona modelo basado en la descripción de la tarea"""
        task_lower = task_description.lower()
        
        # Detectar tipo de tarea
        detected_type = "general"
        if any(word in task_lower for word in ["código", "code", "programar", "script", "function"]):
            detected_type = "code"
        elif any(word in task_lower for word in ["chat", "conversar", "hablar", "platicar"]):
            detected_type = "chat"
        elif any(word in task_lower for word in ["analizar", "análisis", "examinar", "estudiar"]):
            detected_type = "analysis"
        elif any(word in task_lower for word in ["creativo", "historia", "poema", "cuento"]):
            detected_type = "creative"
        elif any(word in task_lower for word in ["técnico", "technical", "explicar", "documentar"]):
            detected_type = "technical"
        
        # Obtener modelos recomendados
        recommended = self.TASK_MAPPING.get(detected_type, ["mistral7b", "llama3"])
        
        # Verificar disponibilidad
        available = [m for m in recommended if m in MODELS]
        
        if available:
            selected = available[0]
            return f"""
Modelo seleccionado: {selected}
Tipo de tarea detectada: {detected_type}
Razón: Este modelo es óptimo para {detected_type}
Alternativas: {', '.join(available[1:])}
""".strip()
        else:
            return f"No hay modelos disponibles para {detected_type}. Modelos instalados: {', '.join(MODELS.keys())}"
    
    async def _arun(self, task_description: str) -> str:
        return self._run(task_description)


# Funciones decoradoras para crear herramientas más simples

@tool
def clear_gpu_cache() -> str:
    """Limpia la caché de GPU y libera memoria VRAM"""
    try:
        if torch.cuda.is_available():
            before = get_gpu_info()
            clear_gpu_memory()
            after = get_gpu_info()
            
            freed = before.get('allocated_gb', 0) - after.get('allocated_gb', 0)
            return f"Memoria GPU liberada: {freed:.2f}GB. VRAM libre ahora: {after.get('free_gb', 0):.1f}GB"
        else:
            return "No hay GPU CUDA disponible"
    except Exception as e:
        return f"Error limpiando GPU: {str(e)}"


@tool
def list_available_models() -> str:
    """Lista todos los modelos disponibles en el laboratorio"""
    if not MODELS:
        return "No hay modelos configurados"
    
    model_list = []
    for key, name in MODELS.items():
        model_list.append(f"- {key}: {name}")
    
    return "Modelos disponibles:\n" + "\n".join(model_list)


# Registry de todas las herramientas
def get_lab_tools() -> List[BaseTool]:
    """Retorna todas las herramientas del laboratorio"""
    return [
        VRAMMonitorTool(),
        FileSearchTool(),
        ModelProfilerTool(),
        ModelSelectorTool(),
        clear_gpu_cache,
        list_available_models
    ]


# Demo
if __name__ == "__main__":
    print("🧪 Demo de herramientas del laboratorio")
    print("=" * 50)
    
    # Probar cada herramienta
    tools = get_lab_tools()
    
    for tool in tools:
        print(f"\n📦 Herramienta: {tool.name}")
        print(f"   Descripción: {tool.description}")
    
    # Ejemplo de uso
    print("\n🔧 Ejemplo: Monitor de VRAM")
    vram_tool = VRAMMonitorTool()
    print(vram_tool.run(""))
    
    print("\n✅ Herramientas listas para LangGraph")