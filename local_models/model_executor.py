# local_models/model_executor.py
"""
ModelExecutor: Ejecutor puro de inferencia sin gestión de modelos.
Responsabilidades:
- Ejecutar inferencia en modelos ya cargados
- Generar métricas de ejecución
- Guardar resultados y logs
- NO gestiona carga/descarga de modelos
"""

import os
import json
import csv
import time
import uuid
from datetime import datetime
from typing import Optional, Dict, Any

from utils.logger import get_logger
from utils.atomic_write import atomic_write
from utils.gpu_guard import get_gpu_info
from .model_manager import get_model_manager, LoadedModel

BASE_OUTPUTS = "outputs"
BASE_LOGS = "logs"
BASE_METRICS = "metrics"


class ModelExecutor:
    """Ejecutor de inferencia para modelos LLM"""
    
    def __init__(self, save_results: bool = True):
        self.save_results = save_results
        self.model_manager = get_model_manager()
        self._logger = get_logger("model_executor", "logs/model_executor", enable_file_logging=True)
    
    def _ensure_dirs(self, model_key: str):
        """Crea directorios necesarios para outputs"""
        out_dir = os.path.join(BASE_OUTPUTS, model_key, "runs")
        log_dir = os.path.join(BASE_LOGS, model_key)
        metrics_dir = os.path.join(BASE_METRICS, model_key)
        os.makedirs(out_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)
        os.makedirs(metrics_dir, exist_ok=True)
        return out_dir, log_dir, metrics_dir
    
    def _timestamp(self) -> str:
        """Genera timestamp para archivos"""
        return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    
    def _generate_run_id(self, model_key: str) -> str:
        """Genera ID único para la ejecución"""
        ts = self._timestamp()
        unique_id = str(uuid.uuid4())[:8]
        return f"{model_key}_{ts}_{unique_id}"
    
    def _clean_output_text(self, text: str) -> str:
        """
        Limpia el texto de salida para evitar errores de encoding UTF-8
        """
        try:
            # ✅ FIX: Limpiar surrogates y caracteres problemáticos
            # Primero encode/decode para limpiar surrogates
            cleaned = text.encode('utf-8', errors='ignore').decode('utf-8')
            
            # Reemplazar caracteres problemáticos comunes
            replacements = {
                ''': "'",  # Apostrophe elegante → apostrophe simple
                ''': "'",  # Apostrophe elegante inverso
                '"': '"',  # Comilla elegante izquierda
                '"': '"',  # Comilla elegante derecha
                '–': '-',  # En dash
                '—': '-',  # Em dash
                '…': '...',  # Elipsis
            }
            
            for old, new in replacements.items():
                cleaned = cleaned.replace(old, new)
            
            return cleaned
            
        except Exception as e:
            # Si todo falla, retornar texto ASCII limpio
            return text.encode('ascii', errors='ignore').decode('ascii')
    
    def _count_tokens_safe(self, text: str, loaded_model: LoadedModel) -> int:
        """Cuenta tokens de forma segura"""
        try:
            return len(loaded_model.tokenizer.encode(text))
        except:
            return len(text.split())  # Fallback: contar palabras
    
    def _log_metrics_csv(self, model_key: str, metrics_data: Dict[str, Any]):
        """Guarda métricas en CSV"""
        if not self.save_results:
            return
            
        _, _, metrics_dir = self._ensure_dirs(model_key)
        filename = os.path.join(metrics_dir, f"{model_key}_metrics.csv")
        write_header = not os.path.exists(filename)
        
        all_fields = [
            "timestamp", "run_id", "model_key", "model_name",
            "load_time_sec", "infer_time_sec", "total_time_sec",
            "prompt_length", "output_length", "tokens_generated",
            "gpu_memory_used_gb", "loading_strategy", "cache_hit"
        ]
        
        # Asegurar que todos los campos existen
        for field in all_fields:
            if field not in metrics_data:
                metrics_data[field] = 'N/A'

        try:
            with open(filename, "a", encoding="utf-8", newline='') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=all_fields)
                if write_header:
                    writer.writeheader()
                writer.writerow({k: metrics_data.get(k, 'N/A') for k in all_fields})
        except Exception as e:
            self._logger.error("csv_write_error", error=str(e))
    
    def _save_execution_results(self, run_id: str, loaded_model: LoadedModel, 
                              prompt: str, output_text: str, 
                              load_time: float, infer_time: float, 
                              cache_hit: bool):
        """Guarda todos los resultados de la ejecución"""
        if not self.save_results:
            return
            
        try:
            out_dir, log_dir, _ = self._ensure_dirs(loaded_model.model_key)
            
            # Guardar output (ya limpio)
            out_path = os.path.join(out_dir, f"{run_id}.txt")
            atomic_write(out_path, output_text.encode("utf-8", errors='ignore'))
            
            # Guardar log detallado
            log_data = {
                "run_id": run_id,
                "model_key": loaded_model.model_key,
                "model_name": loaded_model.model_name,
                "strategy": loaded_model.strategy,
                "prompt": prompt,
                "output": output_text,
                "load_time_sec": round(load_time, 2),
                "infer_time_sec": round(infer_time, 2),
                "total_time_sec": round(load_time + infer_time, 2),
                "cache_hit": cache_hit,
                "timestamp": self._timestamp(),
                "gpu_info": get_gpu_info()
            }
            
            log_path = os.path.join(log_dir, f"{run_id}.json")
            atomic_write(log_path, json.dumps(log_data, indent=2, ensure_ascii=False).encode("utf-8", errors='ignore'))
            
            # Calcular métricas de forma segura
            tokens_generated = self._count_tokens_safe(output_text, loaded_model)
            gpu_info = get_gpu_info()
            
            metrics_data = {
                "timestamp": self._timestamp(),
                "run_id": run_id,
                "model_key": loaded_model.model_key,
                "model_name": loaded_model.model_name,
                "load_time_sec": round(load_time, 2),
                "infer_time_sec": round(infer_time, 2),
                "total_time_sec": round(load_time + infer_time, 2),
                "prompt_length": len(prompt),
                "output_length": len(output_text),
                "tokens_generated": tokens_generated,
                "gpu_memory_used_gb": round(gpu_info.get("allocated_gb", 0), 2),
                "loading_strategy": loaded_model.strategy,
                "cache_hit": cache_hit
            }
            
            self._log_metrics_csv(loaded_model.model_key, metrics_data)
            
            self._logger.info("results_saved", 
                            run_id=run_id,
                            output_path=out_path,
                            metrics=metrics_data)
            
        except Exception as e:
            self._logger.error("save_results_error", run_id=run_id, error=str(e))
    
    def execute(self, model_key: str, prompt: str, 
                max_tokens: int = 128, strategy: str = "optimized",
                temperature: float = 0.7, **kwargs) -> Dict[str, Any]:
        """
        Ejecuta inferencia en un modelo.
        
        Returns:
            Dict con: output, metrics, success, run_id
        """
        run_id = self._generate_run_id(model_key)
        start_total = time.time()
        
        self._logger.info("execution_start", 
                         run_id=run_id,
                         model=f"{model_key}_{strategy}",
                         prompt_length=len(prompt))
        
        try:
            # Obtener/cargar modelo
            start_load = time.time()
            cache_hit = self.model_manager.is_model_loaded(model_key, strategy)
            
            loaded_model = self.model_manager.load_model(model_key, strategy, **kwargs)
            end_load = time.time()
            load_time = end_load - start_load
            
            # Ejecutar inferencia
            start_infer = time.time()
            
            if hasattr(loaded_model.pipeline_obj, "generate"):
                # Custom generate method
                output_text = loaded_model.pipeline_obj.generate(
                    prompt, max_tokens, self._logger
                )
            else:
                # Standard HuggingFace pipeline
                generation_params = {
                    "max_new_tokens": max_tokens,
                    "do_sample": True,
                    "temperature": temperature,
                    "top_p": 0.9,
                    "repetition_penalty": 1.1,
                    "pad_token_id": loaded_model.tokenizer.pad_token_id,
                    "eos_token_id": loaded_model.tokenizer.eos_token_id
                }
                
                output = loaded_model.pipeline_obj(prompt, **generation_params)
                output_text = output[0]["generated_text"]
                
                # Remover el prompt del output si está incluido
                if output_text.startswith(prompt):
                    output_text = output_text[len(prompt):].strip()
            
            # ✅ FIX: Limpiar output ANTES de continuar
            output_text = self._clean_output_text(output_text)
            
            end_infer = time.time()
            infer_time = end_infer - start_infer
            total_time = time.time() - start_total
            
            # Guardar resultados
            self._save_execution_results(
                run_id, loaded_model, prompt, output_text,
                load_time, infer_time, cache_hit
            )
            
            # Preparar respuesta
            result = {
                "success": True,
                "run_id": run_id,
                "output": output_text,
                "metrics": {
                    "load_time_sec": round(load_time, 2),
                    "inference_time_sec": round(infer_time, 2),
                    "total_time_sec": round(total_time, 2),
                    "cache_hit": cache_hit,
                    "model_name": loaded_model.model_name,
                    "strategy": loaded_model.strategy,
                    "gpu_info": get_gpu_info(),
                    "tokens_generated": self._count_tokens_safe(output_text, loaded_model)
                }
            }
            
            self._logger.info("execution_success",
                            run_id=run_id,
                            total_time=round(total_time, 2),
                            cache_hit=cache_hit)
            
            return result
            
        except Exception as e:
            error_msg = str(e)
            self._logger.error("execution_failed",
                             run_id=run_id,
                             error=error_msg,
                             error_type=type(e).__name__)
            
            return {
                "success": False,
                "run_id": run_id,
                "output": "",
                "error": error_msg,
                "metrics": {
                    "load_time_sec": 0,
                    "inference_time_sec": 0,
                    "total_time_sec": time.time() - start_total,
                    "cache_hit": False
                }
            }
    
    def get_execution_stats(self) -> Dict[str, Any]:
        """Retorna estadísticas del executor y model manager"""
        return {
            "executor_info": {
                "save_results": self.save_results,
                "logger_name": self._logger.name
            },
            "model_manager_stats": self.model_manager.get_memory_stats(),
            "loaded_models": self.model_manager.get_loaded_models()
        }


# Función de conveniencia
def execute_model(model_key: str, prompt: str, max_tokens: int = 128, 
                 strategy: str = "optimized", **kwargs) -> Dict[str, Any]:
    """
    Función de conveniencia para ejecutar un modelo.
    """
    executor = ModelExecutor()
    return executor.execute(model_key, prompt, max_tokens, strategy, **kwargs)