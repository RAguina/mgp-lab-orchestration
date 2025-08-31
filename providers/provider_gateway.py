# providers/provider_gateway.py
import time
import uuid
import logging
from typing import Dict, Any, Optional, List, Literal, TypedDict

from providers.registries.model_registry import get as reg_get
from providers.base import GenerationRequest as BaseGenerationRequest, GenerationResult as BaseGenerationResult, Provider
from providers.local.local_provider import LocalProvider

logger = logging.getLogger("provider_gateway")

# -----------------------
# Tipos formales (compat V4)
# -----------------------

FlowType = Literal["simple", "linear", "challenge", "multi_perspective"]

class GenerationRequest(TypedDict, total=False):
    model_key: str
    strategy: str
    prompt: str
    max_tokens: int
    temperature: float
    # Nuevos (opcionales, retrocompatibles)
    flow_type: FlowType
    execution_id: str

class ExecutionTraceEntry(TypedDict, total=False):
    step: int
    provider: str
    model: str
    strategy: str
    status: Literal["start", "success", "error", "blocked"]
    tokens: int
    load_time_ms: int
    inference_time_ms: int
    total_time_ms: int
    error: Optional[str]

class GenerationResult(TypedDict, total=False):
    success: bool
    output: Optional[str]
    error: Optional[str]
    metrics: Dict[str, Any]
    timings_ms: Dict[str, int]
    source: str
    model_used: str
    strategy_used: str
    execution_trace: List[ExecutionTraceEntry]
    blocked: bool
    # Nuevos (trazabilidad end-to-end)
    flow_type: FlowType
    execution_id: str

# -----------------------
# Gateway
# -----------------------

class ProviderGateway:
    """
    Fachada única de generación con:
    - Circuit breaker básico por provider+modelo
    - execution_trace compatible
    - Desacople via interfaz Provider (no conoce ModelExecutor)
    - Listo para multi-provider (local/openai/anthropic)
    """
    def __init__(self, providers: Optional[Dict[str, Provider]] = None, max_failures: int = 3):
        # providers dict: claves esperadas: "local", "openai", "anthropic", ...
        self.providers: Dict[str, Provider] = providers or {
            "local": LocalProvider()
        }
        self.fail_counts: Dict[str, int] = {}
        self.max_failures = max_failures

    # -------- Circuit breaker helpers --------
    def _provider_key_from_meta(self, model_key: str) -> str:
        meta = reg_get(model_key) or {}
        # meta puede traer "provider": "local" | "openai" | "anthropic" | ...
        return (meta.get("provider") or "local").lower()

    def _make_provider_id(self, provider_key: str, model_key: str) -> str:
        # Mantenemos el patrón "local:mistral7b" para métricas/trace
        return f"{provider_key}:{model_key}"

    def should_block(self, provider_id: str) -> bool:
        return self.fail_counts.get(provider_id, 0) >= self.max_failures

    def _record_failure(self, provider_id: str) -> None:
        self.fail_counts[provider_id] = self.fail_counts.get(provider_id, 0) + 1

    def _reset_failures(self, provider_id: str) -> None:
        self.fail_counts[provider_id] = 0

    # -------- Selección de provider (mínima) --------
    def _pick_provider(self, req: GenerationRequest) -> str:
        model_key = req.get("model_key", "mistral7b")
        provider_key = self._provider_key_from_meta(model_key)
        if provider_key not in self.providers:
            logger.warning(f"[gateway] provider '{provider_key}' no disponible, usando 'local'")
            provider_key = "local"
        return provider_key

    # -------- Normalización segura --------
    @staticmethod
    def _ensure_result_shape(
        provider_key: str,
        model_key: str,
        strategy: str,
        res: BaseGenerationResult,
        flow_type: Optional[FlowType],
        execution_id: Optional[str],
    ) -> GenerationResult:
        """
        Asegura que el resultado tenga todos los campos del contrato V4,
        aun si el provider omitió alguno.
        """
        success = bool(res.get("success"))
        m = res.get("metrics", {}) or {}
        t_ms = res.get("timings_ms", {}) or {}
        output = res.get("output")
        error = res.get("error")

        out: GenerationResult = {
            "success": success,
            "output": output,
            "error": error,
            "metrics": m,
            "timings_ms": {
                "load_time_ms": int(t_ms.get("load_time_ms", 0)),
                "inference_time_ms": int(t_ms.get("inference_time_ms", 0)),
                "total_time_ms": int(t_ms.get("total_time_ms", 0)),
            },
            "source": res.get("source", f"{provider_key}:{model_key}"),
            "model_used": res.get("model_used", model_key),
            "strategy_used": res.get("strategy_used", strategy),
            "execution_trace": [],  # el Gateway lo rellena abajo
            "blocked": False,
        }
        # Trazabilidad: mantener aunque el provider no lo incluya
        if "flow_type" in res:
            out["flow_type"] = res["flow_type"]  # type: ignore
        elif flow_type is not None:
            out["flow_type"] = flow_type  # type: ignore
        if "execution_id" in res:
            out["execution_id"] = res["execution_id"]  # type: ignore
        elif execution_id is not None:
            out["execution_id"] = execution_id  # type: ignore
        return out

    # -------- Operación principal --------
    def generate(self, req: GenerationRequest) -> GenerationResult:
        model_key   = req.get("model_key", "mistral7b")
        strategy    = req.get("strategy", "optimized")
        prompt      = req.get("prompt", "")
        max_tokens  = req.get("max_tokens", 256)
        temperature = req.get("temperature", 0.7)
        flow_type   = req.get("flow_type") or "simple"  # por compatibilidad
        execution_id = req.get("execution_id") or str(uuid.uuid4())

        # Validación mínima
        if not prompt:
            return {
                "success": False,
                "error": "prompt vacío",
                "metrics": {},
                "timings_ms": {},
                "source": "unknown",
                "model_used": model_key,
                "strategy_used": strategy,
                "execution_trace": [],
                "blocked": False,
                "flow_type": flow_type,         # mantener trazabilidad incluso en error
                "execution_id": execution_id,
            }

        provider_key = self._pick_provider(req)
        provider_id = self._make_provider_id(provider_key, model_key)
        trace: List[ExecutionTraceEntry] = [{
            "step": 1,
            "provider": provider_key,
            "model": model_key,
            "strategy": strategy,
            "status": "start",
            "tokens": 0,
            "load_time_ms": 0,
            "inference_time_ms": 0,
            "total_time_ms": 0
        }]

        if self.should_block(provider_id):
            trace[-1]["status"] = "blocked"
            return {
                "success": False,
                "error": f"Provider '{provider_id}' bloqueado por fallos repetidos",
                "metrics": {},
                "timings_ms": {},
                "source": provider_id,
                "model_used": model_key,
                "strategy_used": strategy,
                "execution_trace": trace,
                "blocked": True,
                "flow_type": flow_type,
                "execution_id": execution_id,
            }

        provider = self.providers.get(provider_key)
        if provider is None:
            trace[-1]["status"] = "error"
            err = f"Provider '{provider_key}' no disponible"
            return {
                "success": False,
                "error": err,
                "metrics": {},
                "timings_ms": {},
                "source": provider_id,
                "model_used": model_key,
                "strategy_used": strategy,
                "execution_trace": trace,
                "blocked": False,
                "flow_type": flow_type,
                "execution_id": execution_id,
            }

        # Ejecutar
        try:
            t0 = time.time()
            # Pasamos el req tal cual; providers implementan la interfaz base
            provider_req: BaseGenerationRequest = {
                "model_key": model_key,
                "strategy": strategy,
                "prompt": prompt,
                "max_tokens": max_tokens,
                "temperature": temperature,
                # pista para logging/observabilidad del provider
                "flow_type": flow_type,          # type: ignore
                "execution_id": execution_id,    # type: ignore
            }
            pres: BaseGenerationResult = provider.generate(provider_req)
            elapsed_s = time.time() - t0

        except Exception as e:
            self._record_failure(provider_id)
            trace[-1]["status"] = "error"
            trace[-1]["error"] = str(e)
            return {
                "success": False,
                "error": str(e),
                "metrics": {},
                "timings_ms": {},
                "source": provider_id,
                "model_used": model_key,
                "strategy_used": strategy,
                "execution_trace": trace,
                "blocked": False,
                "flow_type": flow_type,
                "execution_id": execution_id,
            }

        # Éxito: resetear contador de fallos
        self._reset_failures(provider_id)

        # Normalizar al contrato V4
        out: GenerationResult = self._ensure_result_shape(
            provider_key, model_key, strategy, pres, flow_type, execution_id
        )

        # Asegurar total_time_ms razonable aunque el provider no lo haya dado
        t_ms = out["timings_ms"]
        if t_ms.get("total_time_ms", 0) == 0:
            t_ms["total_time_ms"] = int(elapsed_s * 1000)

        # Completar trace con métricas reales
        tokens = out["metrics"].get("tokens_generated", 0)
        trace[-1].update({
            "status": "success" if out["success"] else "error",
            "tokens": tokens,
            "load_time_ms": t_ms.get("load_time_ms", 0),
            "inference_time_ms": t_ms.get("inference_time_ms", 0),
            "total_time_ms": t_ms.get("total_time_ms", 0),
            "error": out.get("error")
        })

        out["execution_trace"] = trace
        # Reafirmar trazabilidad (por si un provider sobrescribió)
        out["flow_type"] = out.get("flow_type", flow_type)  # type: ignore
        out["execution_id"] = out.get("execution_id", execution_id)  # type: ignore
        return out
