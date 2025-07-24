# utils/logger.py - Fixed version (Sin file spam)
import os
import json
import logging
import uuid
import time
from datetime import datetime
from typing import Dict, Any, Optional

class StructuredLogger:
    """
    Logger estructurado con correlación de sesiones y contexto
    """
    
    def __init__(self, name: str, log_dir: str, session_id: Optional[str] = None, enable_file_logging: bool = False):
        self.name = name
        self.log_dir = log_dir
        self.session_id = session_id or str(uuid.uuid4())[:8]
        self.enable_file_logging = enable_file_logging  # ✅ NUEVO: Control de file logging
        self.logger = self._setup_logger()
        self._last_log_times = {}  # ✅ NUEVO: Para throttling
        
    def _setup_logger(self) -> logging.Logger:
        """Configura el logger con handlers apropiados"""
        # Crear logger único por sesión
        logger_name = f"{self.name}_{self.session_id}"
        logger = logging.getLogger(logger_name)
        
        # Evitar duplicar handlers si ya existe
        if logger.handlers:
            return logger
            
        logger.setLevel(logging.INFO)
        
        # Console handler para output inmediato (siempre)
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # Formatter simple para console
        console_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%H:%M:%S'
        )
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)
        
        # ✅ NUEVO: File handler SOLO si está habilitado
        if self.enable_file_logging:
            # Crear directorio de logs si no existe
            os.makedirs(self.log_dir, exist_ok=True)
            
            log_file = os.path.join(self.log_dir, f"{self.name}_{self.session_id}.log")
            file_handler = logging.FileHandler(log_file, encoding='utf-8')
            file_handler.setLevel(logging.INFO)
            
            # Formatter personalizado para JSON estructurado
            class JSONFormatter(logging.Formatter):
                def __init__(self, session_id):
                    super().__init__()
                    self.session_id = session_id
                    
                def format(self, record):
                    # Extraer el mensaje como JSON si es posible
                    try:
                        if isinstance(record.msg, str) and record.msg.startswith('{'):
                            msg_data = json.loads(record.msg)
                        else:
                            msg_data = {"message": str(record.msg)}
                    except:
                        msg_data = {"message": str(record.msg)}
                    
                    # Crear estructura de log completa
                    log_entry = {
                        "timestamp": datetime.fromtimestamp(record.created).isoformat(),
                        "session_id": getattr(record, 'session_id', self.session_id),
                        "logger": record.name,
                        "level": record.levelname,
                        "module": record.module,
                        "function": record.funcName,
                        "line": record.lineno,
                        **msg_data
                    }
                    
                    # Agregar información de excepción si existe
                    if record.exc_info:
                        log_entry["exception"] = self.formatException(record.exc_info)
                    
                    return json.dumps(log_entry, ensure_ascii=False)
            
            file_handler.setFormatter(JSONFormatter(self.session_id))
            logger.addHandler(file_handler)
        
        return logger
    
    def log_structured(self, level: str, event: str, **kwargs):
        """Log estructurado con contexto automático y throttling"""
        # ✅ NUEVO: Throttling para evitar spam
        throttle_key = f"{level}:{event}"
        now = time.time()
        
        # Permitir el mismo evento solo cada 30 segundos
        if throttle_key in self._last_log_times:
            if now - self._last_log_times[throttle_key] < 30.0:
                return  # Skip este log para evitar spam
        
        self._last_log_times[throttle_key] = now
        
        log_data = {
            "event": event,
            "session_id": self.session_id,
            **kwargs
        }
        
        # Agregar timestamp si no está presente
        if "timestamp" not in log_data:
            log_data["timestamp"] = datetime.now().isoformat()
        
        message = json.dumps(log_data, ensure_ascii=False)
        
        # Enviar al logger según el nivel
        getattr(self.logger, level.lower())(message)
    
    def info(self, event: str, **kwargs):
        """Log de información"""
        self.log_structured("info", event, **kwargs)
    
    def warning(self, event: str, **kwargs):
        """Log de advertencia"""
        self.log_structured("warning", event, **kwargs)
    
    def error(self, event: str, **kwargs):
        """Log de error"""
        self.log_structured("error", event, **kwargs)
    
    def debug(self, event: str, **kwargs):
        """Log de debug"""
        self.log_structured("debug", event, **kwargs)
    
    def exception(self, event: str, **kwargs):
        """Log de excepción con traceback"""
        self.log_structured("error", event, **kwargs)
        self.logger.exception("Exception details:")

class ModelRunLogger(StructuredLogger):
    """
    Logger especializado para runs de modelos con métricas automáticas
    """
    
    def __init__(self, model_key: str, log_dir: str, run_id: Optional[str] = None, enable_file_logging: bool = False):
        self.model_key = model_key
        self.run_id = run_id or f"{model_key}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{str(uuid.uuid4())[:8]}"
        
        # ✅ NUEVO: Pasar enable_file_logging al padre
        super().__init__(f"model_{model_key}", log_dir, self.run_id, enable_file_logging)
        
        # ✅ NUEVO: Log inicial SOLO si file logging está habilitado
        if enable_file_logging:
            self.info("run_initialized", 
                     run_id=self.run_id, 
                     model_key=model_key)
    
    def log_model_load_start(self, model_name: str, device_map: str, **kwargs):
        """Log inicio de carga de modelo"""
        self.info("model_load_start",
                 model_name=model_name,
                 device_map=str(device_map),
                 **kwargs)
    
    def log_model_load_complete(self, load_time: float, gpu_info: Dict[str, Any], **kwargs):
        """Log finalización de carga de modelo"""
        self.info("model_load_complete",
                 load_time_seconds=round(load_time, 3),
                 gpu_info=gpu_info,
                 **kwargs)
    
    def log_inference_start(self, prompt: str, max_tokens: int, temperature: float, **kwargs):
        """Log inicio de inferencia"""
        self.info("inference_start",
                 prompt_preview=prompt[:100],
                 prompt_length=len(prompt),
                 max_tokens=max_tokens,
                 temperature=temperature,
                 **kwargs)
    
    def log_inference_complete(self, inference_time: float, output: str, 
                             tokens_generated: Optional[int] = None, **kwargs):
        """Log finalización de inferencia"""
        self.info("inference_complete",
                 inference_time_seconds=round(inference_time, 3),
                 output_preview=output[:200],
                 output_length=len(output),
                 tokens_generated=tokens_generated,
                 tokens_per_second=round(tokens_generated / inference_time, 2) if tokens_generated and inference_time > 0 else None,
                 **kwargs)
    
    def log_inference_error(self, error: str, error_type: str, partial_time: float, **kwargs):
        """Log error de inferencia"""
        self.error("inference_error",
                  error_message=str(error),
                  error_type=error_type,
                  partial_time_seconds=round(partial_time, 3),
                  **kwargs)
    
    def log_resource_cleanup(self, cleanup_result: Dict[str, Any], **kwargs):
        """Log limpieza de recursos"""
        self.info("resource_cleanup",
                 cleanup_result=cleanup_result,
                 **kwargs)
    
    def log_run_summary(self, success: bool, total_time: float, **kwargs):
        """Log resumen final del run"""
        self.info("run_summary",
                 success=success,
                 total_time_seconds=round(total_time, 3),
                 **kwargs)

# ✅ NUEVO: Funciones actualizadas con control de file logging
def get_logger(model_key: str, log_dir: str, run_id: Optional[str] = None, enable_file_logging: bool = False) -> ModelRunLogger:
    """
    Función de conveniencia para obtener un logger de modelo
    Por defecto NO escribe archivos para evitar spam
    """
    return ModelRunLogger(model_key, log_dir, run_id, enable_file_logging)

def get_session_logger(name: str, log_dir: str, session_id: Optional[str] = None, enable_file_logging: bool = False) -> StructuredLogger:
    """
    Función de conveniencia para obtener un logger de sesión
    Por defecto NO escribe archivos para evitar spam
    """
    return StructuredLogger(name, log_dir, session_id, enable_file_logging)

class LogAnalyzer:
    """
    Herramientas para analizar logs generados
    """
    
    @staticmethod
    def parse_log_file(log_file_path: str) -> list:
        """Parse un archivo de log JSON a lista de diccionarios"""
        entries = []
        try:
            with open(log_file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            entry = json.loads(line)
                            entries.append(entry)
                        except json.JSONDecodeError:
                            # Manejar líneas que no son JSON válido
                            continue
        except FileNotFoundError:
            print(f"Archivo de log no encontrado: {log_file_path}")
        except Exception as e:
            print(f"Error leyendo log: {e}")
        
        return entries
    
    @staticmethod
    def analyze_model_performance(log_dir: str, model_key: str) -> Dict[str, Any]:
        """Analiza performance de un modelo basado en logs"""
        import glob
        
        # Buscar archivos de log del modelo
        pattern = os.path.join(log_dir, f"model_{model_key}_*.log")
        log_files = glob.glob(pattern)
        
        if not log_files:
            return {"error": f"No se encontraron logs para {model_key}"}
        
        all_entries = []
        for log_file in log_files:
            entries = LogAnalyzer.parse_log_file(log_file)
            all_entries.extend(entries)
        
        # Analizar métricas
        load_times = []
        inference_times = []
        successful_runs = 0
        failed_runs = 0
        
        for entry in all_entries:
            event = entry.get("event", "")
            
            if event == "model_load_complete":
                load_time = entry.get("load_time_seconds")
                if load_time:
                    load_times.append(load_time)
            
            elif event == "inference_complete":
                inference_time = entry.get("inference_time_seconds")
                if inference_time:
                    inference_times.append(inference_time)
                successful_runs += 1
            
            elif event == "inference_error":
                failed_runs += 1
        
        # Calcular estadísticas
        analysis = {
            "model_key": model_key,
            "total_log_entries": len(all_entries),
            "log_files_analyzed": len(log_files),
            "successful_runs": successful_runs,
            "failed_runs": failed_runs,
            "success_rate": round(successful_runs / (successful_runs + failed_runs) * 100, 2) if (successful_runs + failed_runs) > 0 else 0
        }
        
        if load_times:
            analysis["load_time_stats"] = {
                "avg_seconds": round(sum(load_times) / len(load_times), 2),
                "min_seconds": round(min(load_times), 2),
                "max_seconds": round(max(load_times), 2),
                "samples": len(load_times)
            }
        
        if inference_times:
            analysis["inference_time_stats"] = {
                "avg_seconds": round(sum(inference_times) / len(inference_times), 2),
                "min_seconds": round(min(inference_times), 2),
                "max_seconds": round(max(inference_times), 2),
                "samples": len(inference_times)
            }
        
        return analysis

# Tests y utilidades
if __name__ == "__main__":
    # Demo del logger mejorado
    import tempfile
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Test logger básico (CON file logging para testing)
        logger = get_session_logger("test", temp_dir, enable_file_logging=True)
        logger.info("test_event", message="Prueba de logging", value=42)
        logger.warning("warning_event", issue="Memoria baja")
        
        # Test logger de modelo (CON file logging para testing)
        model_logger = get_logger("test_model", temp_dir, enable_file_logging=True)
        model_logger.log_model_load_start("test/model", "auto")
        model_logger.log_inference_start("Test prompt", 128, 0.7)
        model_logger.log_inference_complete(1.5, "Generated text", 25)
        
        print("✅ Tests de logging completados")
        print(f"📁 Logs generados en: {temp_dir}")