# local_models/llm_launcher.py
"""
Launcher principal refactorizado con funciones modulares por estrategia (hf_pipeline_wrappers)
"""
import os
import json
import csv
import time
from datetime import datetime
from typing import Optional, Dict, Any

from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from utils.logger import get_logger
from utils.atomic_write import atomic_write
from utils.gpu_guard import get_gpu_info, clear_gpu_memory

# Importar funciones de carga de estrategias
from langchain_integration.wrappers.hf_pipeline_wrappers import (
    standard,
    optimized,
    streaming,
)

STRATEGY_LOADERS = {
    "standard": standard.load_model,
    "optimized": optimized.load_model,
    "streaming": streaming.load_model,
}

MODELS = {
    "llama3": "meta-llama/Meta-Llama-3-8B-Instruct",
    "mistral7b": "mistralai/Mistral-7B-Instruct-v0.2",
    "deepseek7b": "deepseek-ai/deepseek-llm-7b-instruct",
    "deepseek-coder": "deepseek-ai/deepseek-coder-6.7b-instruct"
}

BASE_OUTPUTS = "outputs"
BASE_LOGS = "logs"
BASE_METRICS = "metrics"

def ensure_dirs(model_key):
    out_dir = os.path.join(BASE_OUTPUTS, model_key, "runs")
    log_dir = os.path.join(BASE_LOGS, model_key)
    metrics_dir = os.path.join(BASE_METRICS, model_key)
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(metrics_dir, exist_ok=True)
    return out_dir, log_dir, metrics_dir

def timestamp():
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

def log_metrics_csv(model_key, metrics_data):
    _, _, metrics_dir = ensure_dirs(model_key)
    filename = os.path.join(metrics_dir, f"{model_key}_metrics.csv")
    write_header = not os.path.exists(filename)
    
    all_fields = [
        "timestamp", "model_key", "model_name",
        "load_time_sec", "infer_time_sec",
        "prompt_length", "output_length", "tokens_generated",
        "gpu_memory_used_gb", "loading_strategy"
    ]
    for field in all_fields:
        if field not in metrics_data:
            metrics_data[field] = 'N/A'

    with open(filename, "a", encoding="utf-8", newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=all_fields)
        if write_header:
            writer.writeheader()
        writer.writerow({k: metrics_data.get(k, 'N/A') for k in all_fields})

class ModelLauncher:
    def __init__(self, model_key: str, loading_strategy: str = "standard"):
        self.model_key = model_key
        self.model_name = MODELS.get(model_key)
        if not self.model_name:
            raise ValueError(f"Modelo no encontrado: {model_key}")

        self.load_fn = STRATEGY_LOADERS.get(loading_strategy, standard.load_model)
        self.strategy_name = loading_strategy

        self.out_dir, self.log_dir, self.metrics_dir = ensure_dirs(model_key)
        self.logger = get_logger(model_key, self.log_dir)

    def launch(self, prompt: str, max_tokens: int = 128, **kwargs) -> Optional[str]:
        ts = timestamp()
        run_id = f"{self.model_key}_{ts}"
        self.logger.info("run_start", run_id=run_id, loading_strategy=self.strategy_name, gpu_info_before=get_gpu_info())

        try:
            start_load = time.time()
            model, tokenizer, pipeline_obj = self.load_fn(self.model_name, self.logger, **kwargs)
            end_load = time.time()
            load_time = end_load - start_load

            start_infer = time.time()
            if hasattr(pipeline_obj, "generate"):
                output_text = pipeline_obj.generate(prompt, max_tokens, self.logger)
            else:
                output = pipeline_obj(prompt, max_new_tokens=max_tokens, do_sample=True, temperature=0.7, top_p=0.9, repetition_penalty=1.1, pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id)
                output_text = output[0]["generated_text"]
            end_infer = time.time()
            infer_time = end_infer - start_infer

            self._save_results(run_id, prompt, output_text, load_time, infer_time, tokenizer)
            self.logger.info("run_success", gpu_info_after=get_gpu_info(), total_time=load_time + infer_time)

            return output_text

        except Exception as e:
            self.logger.exception("run_error", error=str(e), error_type=type(e).__name__)
            return None

        finally:
            self._cleanup(model if 'model' in locals() else None, pipeline_obj if 'pipeline_obj' in locals() else None)

    def _save_results(self, run_id: str, prompt: str, output_text: str, load_time: float, infer_time: float, tokenizer):
        out_path = os.path.join(self.out_dir, f"{run_id}.txt")
        atomic_write(out_path, output_text.encode("utf-8"))
        self.logger.info("output_saved", path=out_path)

        log_data = {
            "run_id": run_id,
            "model_key": self.model_key,
            "model_name": self.model_name,
            "prompt": prompt,
            "output": output_text,
            "load_time_sec": round(load_time, 2),
            "infer_time_sec": round(infer_time, 2),
            "timestamp": timestamp(),
            "loading_strategy": self.strategy_name
        }
        log_path = os.path.join(self.log_dir, f"{run_id}.json")
        atomic_write(log_path, json.dumps(log_data, indent=2, ensure_ascii=False).encode("utf-8"))

        tokens_generated = len(tokenizer.encode(output_text)) - len(tokenizer.encode(prompt))
        gpu_info = get_gpu_info()

        metrics_data = {
            "timestamp": timestamp(),
            "model_key": self.model_key,
            "model_name": self.model_name,
            "load_time_sec": round(load_time, 2),
            "infer_time_sec": round(infer_time, 2),
            "prompt_length": len(prompt),
            "output_length": len(output_text),
            "tokens_generated": tokens_generated,
            "gpu_memory_used_gb": round(gpu_info.get("allocated_gb", 0), 2),
            "loading_strategy": self.strategy_name
        }
        log_metrics_csv(self.model_key, metrics_data)
        self.logger.info("metrics_saved", metrics=metrics_data)

    def _cleanup(self, model=None, pipeline_obj=None):
        try:
            if model is not None:
                del model
            if pipeline_obj is not None:
                del pipeline_obj
            clear_gpu_memory()
            self.logger.info("cleanup_complete", gpu_info=get_gpu_info())
        except Exception as e:
            self.logger.warning("cleanup_error", error=str(e))

# Funciones de compatibilidad

def launch_model(model_key: str, prompt: str, max_tokens: int = 128, device_map: str = "auto") -> Optional[str]:
    launcher = ModelLauncher(model_key, "standard")
    return launcher.launch(prompt, max_tokens, device_map=device_map)

def launch_model_optimized(model_key: str, prompt: str, max_tokens: int = 128, device_map: str = "auto", use_quantization: bool = True) -> Optional[str]:
    launcher = ModelLauncher(model_key, "optimized")
    return launcher.launch(prompt, max_tokens, device_map=device_map, use_quantization=use_quantization)

def launch_model_with_streaming(model_key: str, prompt: str, max_tokens: int = 128, device_map: str = "auto") -> Optional[str]:
    launcher = ModelLauncher(model_key, "streaming")
    return launcher.launch(prompt, max_tokens, device_map=device_map)

def menu():
    print("\n=== LAUNCHER DE LLMs ===")
    print("Modelos disponibles:")
    for i, key in enumerate(MODELS.keys(), 1):
        print(f"  {i}. {key}")

    model_choice = input("\n¿Qué modelo quieres usar? [número o nombre]: ").strip()
    if model_choice.isdigit():
        idx = int(model_choice) - 1
        model_keys = list(MODELS.keys())
        if 0 <= idx < len(model_keys):
            model_key = model_keys[idx]
        else:
            print("❌ Número inválido")
            return
    else:
        model_key = model_choice

    if model_key not in MODELS:
        print("❌ Modelo no encontrado")
        return

    print("\nEstrategias de carga:")
    print("  1. Standard (rápida, más memoria)")
    print("  2. Optimizada (quantización 4-bit, menos memoria)")
    print("  3. Streaming (ver output en tiempo real)")

    strategy_choice = input("\nEstrategia [1-3, default=1]: ").strip() or "1"
    strategies = {"1": "standard", "2": "optimized", "3": "streaming"}
    strategy = strategies.get(strategy_choice, "standard")

    prompt = input("\nPrompt: ").strip()
    if not prompt:
        print("❌ Prompt vacío")
        return

    try:
        max_tokens = int(input("Max tokens [default=128]: ").strip() or "128")
    except ValueError:
        max_tokens = 128

    print(f"\n🚀 Lanzando {model_key} con estrategia {strategy}...")
    launcher = ModelLauncher(model_key, strategy)
    result = launcher.launch(prompt, max_tokens)

    if result:
        print("\n✅ Generación completada")
        print(f"📝 Output guardado en: outputs/{model_key}/runs/")
    else:
        print("\n❌ Error durante la generación")

if __name__ == "__main__":
    menu()
