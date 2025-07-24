# local_models/loading_strategies.py
"""
Estrategias modulares para cargar y ejecutar modelos
"""
import torch
import time
from abc import ABC, abstractmethod
from typing import Tuple, Any, Optional
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    pipeline,
    TextStreamer,
    BitsAndBytesConfig
)

class LoadingStrategy(ABC):
    """Clase base para estrategias de carga de modelos"""
    
    @abstractmethod
    def load_model(self, model_name: str, logger: Any, **kwargs) -> Tuple[Any, Any, Any]:
        """Carga modelo, tokenizer y pipeline"""
        pass
    
    @abstractmethod
    def generate(self, pipeline_obj: Any, tokenizer: Any, prompt: str, 
                max_tokens: int, logger: Any) -> str:
        """Genera texto usando el pipeline"""
        pass

class StandardLoadingStrategy(LoadingStrategy):
    """Estrategia estándar - carga rápida sin optimizaciones"""
    
    def load_model(self, model_name: str, logger: Any, **kwargs) -> Tuple[Any, Any, Any]:
        device_map = kwargs.get('device_map', 'auto')
        
        # Cargar tokenizer
        logger.info("tokenizer_load_start")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        logger.info("tokenizer_loaded")
        
        # Cargar modelo
        logger.info("model_load_start", strategy="standard")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map=device_map,
            torch_dtype=torch.float16,  # FP16 para ahorrar algo de memoria
            low_cpu_mem_usage=True
        )
        logger.info("model_loaded")
        
        # Crear pipeline
        logger.info("pipeline_create")
        llm = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            device_map=device_map
        )
        logger.info("pipeline_ready")
        
        return model, tokenizer, llm
    
    def generate(self, pipeline_obj: Any, tokenizer: Any, prompt: str, 
                max_tokens: int, logger: Any) -> str:
        logger.info("inference_start", prompt_preview=prompt[:100])
        
        output = pipeline_obj(
            prompt,
            max_new_tokens=max_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.1
        )
        
        output_text = output[0]["generated_text"]
        logger.info("inference_complete", output_preview=output_text[:200])
        
        return output_text

class OptimizedLoadingStrategy(LoadingStrategy):
    """Estrategia optimizada con quantización 4-bit"""
    
    def load_model(self, model_name: str, logger: Any, **kwargs) -> Tuple[Any, Any, Any]:
        device_map = kwargs.get('device_map', 'auto')
        use_quantization = kwargs.get('use_quantization', True)
        
        # Cargar tokenizer
        logger.info("tokenizer_load_start")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        logger.info("tokenizer_loaded")
        
        # Configuración de quantización
        model_kwargs = {
            "device_map": device_map,
            "torch_dtype": torch.float16,
            "low_cpu_mem_usage": True,
        }
        
        if use_quantization and torch.cuda.is_available():
            logger.info("quantization_config_start")
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
            model_kwargs["quantization_config"] = quantization_config
            logger.info("quantization_enabled", bits=4)
        
        # Cargar modelo con quantización
        logger.info("model_load_start", strategy="optimized_4bit")
        model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
        logger.info("model_loaded")
        
        # Crear pipeline optimizado
        logger.info("pipeline_create")
        llm = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            device_map=device_map,
            torch_dtype=torch.float16
        )
        logger.info("pipeline_ready")
        
        return model, tokenizer, llm
    
    def generate(self, pipeline_obj: Any, tokenizer: Any, prompt: str, 
                max_tokens: int, logger: Any) -> str:
        logger.info("inference_start", prompt_preview=prompt[:100])
        
        # Limpiar caché antes de inferencia
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        output = pipeline_obj(
            prompt,
            max_new_tokens=max_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
        
        output_text = output[0]["generated_text"]
        
        # Calcular tokens generados
        tokens_generated = len(tokenizer.encode(output_text)) - len(tokenizer.encode(prompt))
        logger.info("inference_complete", 
                   output_preview=output_text[:200],
                   tokens_generated=tokens_generated)
        
        return output_text

class StreamingLoadingStrategy(LoadingStrategy):
    """Estrategia con streaming - muestra output en tiempo real"""
    
    def load_model(self, model_name: str, logger: Any, **kwargs) -> Tuple[Any, Any, Any]:
        device_map = kwargs.get('device_map', 'auto')
        
        # Cargar tokenizer
        logger.info("tokenizer_load_start")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        logger.info("tokenizer_loaded")
        
        # Cargar modelo con configuración para streaming
        logger.info("model_load_start", strategy="streaming")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map=device_map,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True
        )
        logger.info("model_loaded")
        
        # Pipeline para streaming
        logger.info("pipeline_create")
        # Nota: No usamos pipeline aquí porque necesitamos más control
        logger.info("pipeline_ready")
        
        return model, tokenizer, None  # None para pipeline, usaremos generación manual
    
    def generate(self, pipeline_obj: Any, tokenizer: Any, prompt: str, 
                max_tokens: int, logger: Any) -> str:
        # Obtener el modelo del primer argumento (ya que no usamos pipeline)
        model = pipeline_obj if pipeline_obj is not None else None
        if model is None:
            raise ValueError("Model not provided for streaming generation")
        
        logger.info("inference_start", prompt_preview=prompt[:100])
        
        # Preparar inputs
        inputs = tokenizer(prompt, return_tensors="pt")
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}
        
        # Crear streamer para ver output en tiempo real
        streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
        
        print("\n" + "="*50)
        print("📝 GENERANDO RESPUESTA:")
        print("="*50)
        
        # Generar con streaming
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                repetition_penalty=1.1,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                streamer=streamer
            )
        
        print("\n" + "="*50)
        
        # Decodificar output completo
        output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        logger.info("inference_complete", output_preview=output_text[:200])
        
        return output_text

# Estrategias adicionales pueden agregarse aquí
class FastLoadingStrategy(LoadingStrategy):
    """Estrategia rápida con caché agresivo - para desarrollo"""
    
    def load_model(self, model_name: str, logger: Any, **kwargs) -> Tuple[Any, Any, Any]:
        # Implementación similar pero con opciones de caché
        # torch.backends.cudnn.benchmark = True
        # etc...
        pass
    
    def generate(self, pipeline_obj: Any, tokenizer: Any, prompt: str, 
                max_tokens: int, logger: Any) -> str:
        pass

class CPULoadingStrategy(LoadingStrategy):
    """Estrategia para ejecutar en CPU cuando no hay GPU"""
    
    def load_model(self, model_name: str, logger: Any, **kwargs) -> Tuple[Any, Any, Any]:
        logger.info("cpu_strategy_selected")
        # Forzar device_map a CPU
        # Usar torch.float32 en lugar de float16
        pass
    
    def generate(self, pipeline_obj: Any, tokenizer: Any, prompt: str, 
                max_tokens: int, logger: Any) -> str:
        pass