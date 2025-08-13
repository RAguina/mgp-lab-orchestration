# providers/local/local_provider.py
from typing import Optional, Dict, Any
from providers.base import Provider, GenerationRequest, GenerationResult
from local_models.model_executor import ModelExecutor

class LocalProvider(Provider):
    def __init__(self, executor: Optional[ModelExecutor] = None):
        self.executor = executor or ModelExecutor(save_results=True)

    def generate(self, req: GenerationRequest) -> GenerationResult:
        model_key   = req.get("model_key", "mistral7b")
        strategy    = req.get("strategy", "optimized")
        prompt      = req["prompt"]
        max_tokens  = req.get("max_tokens", 256)
        temperature = req.get("temperature", 0.7)

        res = self.executor.execute(
            model_key=model_key,
            prompt=prompt,
            max_tokens=max_tokens,
            strategy=strategy,
            temperature=temperature,
        )
        m = res.get("metrics", {})
        return {
            "success": True,
            "output": res.get("output", ""),
            "metrics": m,
            "timings_ms": {
                "load_time_ms":      int(m.get("load_time_sec", 0) * 1000),
                "inference_time_ms": int(m.get("inference_time_sec", 0) * 1000),
                "total_time_ms":     int(m.get("total_time_sec", 0) * 1000),
            },
            "source":        "local:hf",
            "model_used":    model_key,
            "strategy_used": strategy,
        }
