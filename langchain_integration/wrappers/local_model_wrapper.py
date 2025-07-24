# langchain_integration/wrappers/local_model_wrapper.py
"""
LocalModelWrapper - Integración LangChain con modelos locales del lab
Semana 1: Wrappers & Fundamentos
"""

import time
import json
import torch
from typing import Dict, Any, Optional, List, Iterator
from datetime import datetime
from pydantic import Field

from langchain_core.language_models.llms import LLM
from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.outputs import LLMResult, Generation

# Imports del lab existente
try:
    from local_models.llm_launcher import ModelLauncher, MODELS
    from utils.gpu_guard import get_gpu_info, clear_gpu_memory
    from utils.logger import get_logger
    # Importar las estrategias de carga
    from langchain_integration.wrappers.hf_pipeline_wrappers import (
        standard,
        optimized,
        streaming,
    )
except ImportError as e:
    print(f"⚠️ Warning: Could not import lab modules: {e}")
    print("⚠️ Using fallback mode")
    
    # Definir MODELS como fallback
    MODELS = {
        "llama3": "meta-llama/Meta-Llama-3-8B-Instruct",
        "mistral7b": "mistralai/Mistral-7B-Instruct-v0.2",
        "deepseek7b": "deepseek-ai/deepseek-llm-7b-chat",
        "deepseek-coder-6.7b": "deepseek-ai/deepseek-coder-6.7b-instruct"
    }
    ModelLauncher = None
    standard = None
    optimized = None
    streaming = None


class LocalModelWrapper(LLM):
    """
    Wrapper LangChain para modelos locales del laboratorio
    Mantiene compatibilidad total con el sistema existente
    """
    
    # Definir campos como atributos de clase con Field()
    model_key: str = Field(description="Clave del modelo a cargar")
    strategy: str = Field(default="optimized", description="Estrategia de carga")
    max_tokens: int = Field(default=128, description="Máximo de tokens a generar")
    temperature: float = Field(default=0.7, description="Temperatura de generación")
    top_p: float = Field(default=0.95, description="Top-p sampling")
    device_map: str = Field(default="auto", description="Device mapping")
    
    # Estado interno - excluir de serialización
    use_launcher: bool = Field(default=True, exclude=True)
    launcher: Optional[Any] = Field(default=None, exclude=True)
    model: Optional[Any] = Field(default=None, exclude=True)
    tokenizer: Optional[Any] = Field(default=None, exclude=True)
    pipeline: Optional[Any] = Field(default=None, exclude=True)
    device: Optional[str] = Field(default=None, exclude=True)
    session_id: str = Field(default="", exclude=True)
    metrics: Dict[str, Any] = Field(default_factory=dict, exclude=True)
    
    def __init__(self, **data: Any):
        """
        Inicializa el wrapper con la configuración especificada
        """
        # Validar modelo disponible antes de inicializar
        model_key = data.get('model_key')
        if model_key and model_key not in MODELS:
            available = ", ".join(MODELS.keys())
            raise ValueError(f"Model '{model_key}' not available. Options: {available}")
        
        # Validar estrategia
        strategy = data.get('strategy', 'optimized')
        valid_strategies = ["standard", "optimized", "streaming"]
        if strategy not in valid_strategies:
            raise ValueError(f"Strategy '{strategy}' not valid. Options: {valid_strategies}")
        
        # Llamar al constructor padre
        super().__init__(**data)
        
        # Inicializar estado interno después de la validación de Pydantic
        self.session_id = f"langchain_{self.model_key}_{int(time.time())}"
        self.metrics = {
            "model_key": self.model_key,
            "strategy": self.strategy,
            "total_calls": 0,
            "total_tokens_generated": 0,
            "total_time": 0.0,
            "created_at": datetime.now().isoformat()
        }
        
        # Decidir si usar ModelLauncher o carga directa
        self.use_launcher = ModelLauncher is not None
        
        # Si no vamos a usar launcher, cargar el modelo ahora
        if not self.use_launcher:
            self._load_model_direct()
    
    def _load_model_direct(self):
        """Carga el modelo directamente usando las estrategias de hf_pipeline_wrappers"""
        if self.model is None and not self.use_launcher:
            print(f"🔧 Cargando modelo {self.model_key} con estrategia {self.strategy}")
            
            try:
                # Seleccionar la función de carga según la estrategia
                if self.strategy == "standard" and standard:
                    load_func = standard.load_model
                elif self.strategy == "optimized" and optimized:
                    load_func = optimized.load_model
                elif self.strategy == "streaming" and streaming:
                    load_func = streaming.load_model
                else:
                    raise RuntimeError(f"Estrategia {self.strategy} no disponible")
                
                # Obtener el nombre completo del modelo
                model_name = MODELS[self.model_key]
                
                # Crear un logger dummy si no está disponible
                logger = get_logger(self.model_key, "logs") if 'get_logger' in globals() else None
                
                # Cargar el modelo
                self.model, self.tokenizer, self.pipeline = load_func(
                    model_name, 
                    logger,
                    device_map=self.device_map
                )
                
                # Determinar el dispositivo
                if hasattr(self.model, 'device'):
                    self.device = str(self.model.device)
                else:
                    self.device = "cuda" if torch.cuda.is_available() else "cpu"
                
                print(f"✅ Modelo cargado en dispositivo: {self.device}")
                
            except Exception as e:
                raise RuntimeError(f"No se pudo cargar el modelo: {e}")
    
    @property
    def _llm_type(self) -> str:
        """Identificador único para LangChain"""
        return f"local_{self.model_key}_{self.strategy}"
    
    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        """
        Método principal llamado por LangChain para generar texto
        """
        start_time = time.time()
        response = None  # Inicializar response
        
        # Obtener info GPU antes (si está disponible)
        gpu_before = get_gpu_info() if 'get_gpu_info' in globals() else {"free_gb": 0, "allocated_gb": 0}
        
        try:
            # Si usamos ModelLauncher, delegar a él
            if self.use_launcher:
                response = self._call_with_launcher(prompt, stop, run_manager, **kwargs)
            else:
                response = self._call_direct(prompt, stop, run_manager, **kwargs)
            
            return response
            
        except Exception as e:
            if run_manager:
                run_manager.on_text(f"❌ Error: {str(e)}", verbose=True)
            raise
            
        finally:
            # Métricas y logging
            elapsed_time = time.time() - start_time
            gpu_after = get_gpu_info() if 'get_gpu_info' in globals() else {"free_gb": 0, "allocated_gb": 0}
            
            # Solo actualizar métricas si tenemos una respuesta
            if response is not None:
                self._update_metrics(prompt, response, elapsed_time, gpu_before, gpu_after)
    
    def _call_with_launcher(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        """Usa ModelLauncher del laboratorio"""
        # Crear launcher si no existe
        if self.launcher is None:
            self.launcher = ModelLauncher(self.model_key, self.strategy)
        
        # Merge kwargs con configuración por defecto
        max_tokens = kwargs.get("max_tokens", self.max_tokens)
        device_map = kwargs.get("device_map", self.device_map)
        
        # Callback de progreso
        if run_manager:
            run_manager.on_text("🔄 Loading model...", verbose=True)
        
        # Ejecutar generación usando el launcher existente
        result = self.launcher.launch(
            prompt=prompt,
            max_tokens=max_tokens,
            device_map=device_map
        )
        
        if result is None:
            raise RuntimeError("Model generation returned None")
        
        # Procesar stop sequences si están especificadas
        if stop:
            for stop_seq in stop:
                if stop_seq in result:
                    result = result.split(stop_seq)[0]
                    break
        
        # Callback de resultado
        if run_manager:
            run_manager.on_text(f"✅ Generated {len(result)} characters", verbose=True)
        
        return result
    
    def _call_direct(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        """Usa el modelo cargado directamente"""
        # Asegurar que el modelo esté cargado
        if self.model is None:
            self._load_model_direct()
        
        # Si tenemos un pipeline, usarlo
        if self.pipeline is not None:
            max_tokens = kwargs.get("max_tokens", self.max_tokens)
            temperature = kwargs.get("temperature", self.temperature)
            top_p = kwargs.get("top_p", self.top_p)
            
            # Callback de progreso
            if run_manager:
                run_manager.on_text("🔄 Generating...", verbose=True)
            
            # Generar usando el pipeline
            outputs = self.pipeline(
                prompt,
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
            
            # Extraer el texto generado
            response = outputs[0]["generated_text"]
            
            # Remover el prompt del resultado si está incluido
            if response.startswith(prompt):
                response = response[len(prompt):].strip()
        
        else:
            # Fallback: usar modelo y tokenizer directamente
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=2048
            )
            
            if self.device:
                inputs = inputs.to(self.device)
            
            gen_kwargs = {
                "max_new_tokens": kwargs.get("max_tokens", self.max_tokens),
                "temperature": kwargs.get("temperature", self.temperature),
                "top_p": kwargs.get("top_p", self.top_p),
                "do_sample": True,
                "pad_token_id": self.tokenizer.eos_token_id,
            }
            
            with torch.no_grad():
                outputs = self.model.generate(**inputs, **gen_kwargs)
            
            generated_tokens = outputs[0][inputs['input_ids'].shape[1]:]
            response = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        
        # Procesar stop sequences
        if stop:
            for stop_seq in stop:
                if stop_seq in response:
                    response = response.split(stop_seq)[0]
                    break
        
        return response
    
    def _update_metrics(self, prompt: str, result: str, elapsed_time: float,
                       gpu_before: Dict, gpu_after: Dict):
        """Actualiza métricas internas compatibles con el lab"""
        
        # Estimar tokens (aproximación)
        prompt_tokens = len(prompt.split())
        result_tokens = len(result.split()) if result else 0
        
        # Actualizar métricas acumuladas
        self.metrics["total_calls"] += 1
        self.metrics["total_tokens_generated"] += result_tokens
        self.metrics["total_time"] += elapsed_time
        self.metrics["last_call"] = {
            "timestamp": datetime.now().isoformat(),
            "prompt_length": len(prompt),
            "prompt_tokens_est": prompt_tokens,
            "result_length": len(result) if result else 0,
            "result_tokens_est": result_tokens,
            "elapsed_time": round(elapsed_time, 3),
            "gpu_before": {
                "free_gb": gpu_before.get("free_gb", 0),
                "allocated_gb": gpu_before.get("allocated_gb", 0)
            },
            "gpu_after": {
                "free_gb": gpu_after.get("free_gb", 0),
                "allocated_gb": gpu_after.get("allocated_gb", 0)
            },
            "vram_delta": round(
                gpu_after.get("allocated_gb", 0) - gpu_before.get("allocated_gb", 0), 3
            )
        }
    
    def _stream(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[str]:
        """
        Genera texto en streaming (implementación básica)
        """
        # Por ahora, simplemente yield el resultado completo
        # TODO: Implementar streaming real si la estrategia lo soporta
        result = self._call(prompt, stop, run_manager, **kwargs)
        yield result
    
    def get_metrics(self) -> Dict[str, Any]:
        """Retorna métricas acumuladas"""
        return self.metrics.copy()
    
    def clear_model_cache(self) -> Dict[str, Any]:
        """Limpia modelo de memoria y cache GPU"""
        # Limpiar launcher si existe
        if self.launcher is not None:
            self.launcher = None
        
        # Limpiar modelo directo si existe
        if self.model is not None:
            del self.model
            self.model = None
        
        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None
            
        if self.pipeline is not None:
            del self.pipeline
            self.pipeline = None
        
        # Limpiar memoria GPU
        if 'clear_gpu_memory' in globals():
            clear_gpu_memory()
        else:
            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        return {"status": "cleared"}
    
    @classmethod
    def get_available_models(cls) -> Dict[str, str]:
        """Retorna modelos disponibles en el laboratorio"""
        return MODELS.copy()
    
    @classmethod
    def get_available_strategies(cls) -> List[str]:
        """Retorna estrategias disponibles"""
        return ["standard", "optimized", "streaming"]
    
    def __repr__(self) -> str:
        return f"LocalModelWrapper(model={self.model_key}, strategy={self.strategy})"
    
    def __del__(self):
        """Cleanup automático al destruir el wrapper"""
        try:
            self.clear_model_cache()
            print(f"🧹 Recursos liberados para {self.model_key}")
        except:
            pass  # Ignore cleanup errors


# Función de conveniencia para crear wrappers
def create_local_llm(model_key: str, strategy: str = "optimized", **kwargs) -> LocalModelWrapper:
    """
    Factory function para crear LocalModelWrapper
    
    Args:
        model_key: Clave del modelo (llama3, mistral7b, etc.)
        strategy: Estrategia de carga (standard/optimized/streaming)
        **kwargs: Parámetros adicionales para el wrapper
        
    Returns:
        LocalModelWrapper configurado
    """
    return LocalModelWrapper(
        model_key=model_key,
        strategy=strategy,
        **kwargs
    )


# Demo básico si se ejecuta directamente
if __name__ == "__main__":
    print("🧪 Demo LocalModelWrapper")
    print("=" * 40)
    
    # Listar modelos disponibles
    models = LocalModelWrapper.get_available_models()
    print(f"Modelos disponibles: {list(models.keys())}")
    
    print("✅ LocalModelWrapper ready para LangChain")