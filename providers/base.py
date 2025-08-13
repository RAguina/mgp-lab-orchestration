# providers/base.py
from typing import TypedDict, Protocol, Dict, Any

class GenerationRequest(TypedDict, total=False):
    prompt: str
    model_key: str
    strategy: str
    max_tokens: int
    temperature: float
    task_type: str
    context: Dict[str, Any]
    constraints: Dict[str, Any]

class GenerationResult(TypedDict, total=False):
    success: bool
    output: str
    metrics: Dict[str, Any]
    timings_ms: Dict[str, int]
    source: str
    model_used: str
    strategy_used: str
    error: str

class Provider(Protocol):
    def generate(self, req: GenerationRequest) -> GenerationResult: ...
