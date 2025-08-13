# providers/registries/model_registry.py
from typing import Dict, Optional

_REGISTRY: Dict[str, Dict] = {
    "llama3":       {"provider": "hf", "id": "meta-llama/Meta-Llama-3-8B-Instruct",      "type": "chat"},
    "mistral7b":    {"provider": "hf", "id": "mistralai/Mistral-7B-Instruct-v0.2",       "type": "chat"},
    "deepseek7b":   {"provider": "hf", "id": "deepseek-ai/deepseek-llm-7b-instruct",     "type": "chat"},
    "deepseek-coder":{"provider":"hf","id":"deepseek-ai/deepseek-coder-6.7b-instruct",   "type": "code"},
}

def get(key: str) -> Optional[Dict]:
    return _REGISTRY.get(key)

def list_models() -> Dict[str, Dict]:
    return _REGISTRY.copy()
