LLM deliberation workflows

## 0) Resumen ejecutivo

Sistema AI-Agent-Lab V3: orquestador de LLMs (locales y API) con arquitectura modular, escalable y resiliente.

**Características principales:**
- Provider Gateway unificado para modelos locales (HuggingFace) y APIs (OpenAI/Anthropic)
- Grafo de orquestación con LangGraph para routing dinámico
- Circuit Breaker para resiliencia ante fallos
- Sistema de registries declarativo para configuración
- Logging estructurado y métricas detalladas
- Compatible con migración progresiva desde V2

**Stack**: Python, FastAPI, LangChain/LangGraph, HuggingFace Transformers

---

## 1) Objetivos de V3

* Migrar a **Provider Gateway pattern** para unificar acceso a LLMs locales y remotos.
* Declarar **registries** de modelos y estrategias para eliminar hardcode.
* Aislar el **orchestrator** de detalles de ejecución.
* Mejorar **logging, métricas y manejo de errores**.
* Preparar infraestructura para **OpenAI/Anthropic adapters**.
* Mantener **compatibilidad progresiva** con Direct Mode.

---

## 2) Entrypoints

* `api/server.py` → API REST principal (FastAPI)
* `api/endpoints/orchestrator.py` → Endpoint `/api/orchestrate`
* `scripts/direct_inference.py` → Modo directo para debug

---

## 3) Mapa de carpetas

```
ai-agent-lab/
├─ api/
│  ├─ server.py                 # Entrypoint servicio FastAPI
│  └─ endpoints/
│     ├─ inference.py          # Direct Mode endpoint
│     ├─ orchestrator.py       # Orchestrated Mode endpoint
│     ├─ cache.py              # Cache management
│     └─ system.py             # Health checks, metrics
│
├─ langchain_integration/
│  └─ langgraph/
│     ├─ routing_agent.py      # Orquestador principal (simplificado)
│     ├─ agent_state.py        # Estado compartido del grafo
│     ├─ orchestration/        # **NUEVO** Sistema modular
│     │  ├─ __init__.py
│     │  ├─ graph_builder.py   # Construcción de grafos
│     │  ├─ graph_configs.py   # Configuraciones declarativas
│     │  └─ flow_metrics.py    # Métricas para frontend
│     └─ nodes/                # Nodos del grafo
│        ├─ task_analyzer_node.py
│        ├─ resource_monitor_node.py
│        ├─ execution_node.py
│        ├─ output_validator_node.py
│        ├─ history_reader_node.py
│        └─ summary_node.py
│
├─ providers/
│  ├─ provider_gateway.py      # Fachada unificada
│  ├─ registries/
│  │  ├─ model_registry.py     # Catálogo de modelos
│  │  └─ strategy_registry.py  # Estrategias de sampling
│  ├─ local/
│  │  ├─ local_model_wrapper.py
│  │  └─ manager/
│  │     ├─ model_manager.py
│  │     └─ model_executor.py
│  └─ remote/
│     ├─ openai_provider.py    # (pendiente)
│     └─ anthropic_provider.py # (pendiente)
│
├─ utils/
│  ├─ logger.py
│  ├─ gpu_guard.py
│  └─ analysis/
│     └─ metrics_reader.py
│
├─ config.py                    # Configuración central
├─ .env                        # Variables de entorno
└─ tests/
   ├─ test_e2e_graph.py
   └─ test_provider_gateway.py
```

---

## 4) Límites y responsabilidades

### 4.1 Orchestrator

* **Hace**: decide ruta, compone nodos, maneja reintentos/validaciones, agrega contexto (history, recursos), agrega métricas.
* **No hace**: escoger proveedor específico, ni hablar con HF/OpenAI/Anthropic, ni cargar modelos locales.
* **API esperada hacia Providers**: `ProviderGateway.generate(request: GenerationRequest) -> GenerationResult`.
* **Estado**: `AgentState` ahora incluye `version` y `request_id` para trazabilidad.

### 4.2 Provider Gateway

* **Responsabilidad**: traducir una solicitud abstracta en una invocación concreta (local vs API) según `model_key`, `strategy`, `task_type`, `constraints` (**VRAM**, **tokens**, **latencia**, **costo**).
* **Selección**: usa `model_registry` (metadatos) + `strategy_registry` (políticas de sampling/límites) + `gpu_guard` (capacidad actual).
* **Resultados**: normaliza salida (texto, tokens, timings, usage) + etiquetas (`cache_hit`, `source=local|openai|anthropic`).
* **Circuit Breaker básico**: bloqueo temporal por proveedor tras N fallos consecutivos (`max_failures` configurable).

### 4.3 Local models manager

* Se conserva `ModelManager` + `ModelExecutor`, pero **exclusivamente detrás del Gateway**.
* Evitar que nodos llamen directamente a `local_models/*`.

---

## 5) Flujos

### 5.1 Direct Mode

`/api/inference` → ProviderGateway.generate()

### 5.2 Orchestrated Execution - Linear Flow

`/api/orchestrate` → `routing_agent.run(state, flow_type="linear")`

```
  ├─ TaskAnalyzer → set(task_type, requirements)
  ├─ ResourceMonitor → set(vram_status)
  ├─ ExecutionNode → ProviderGateway.generate()
  ├─ OutputValidator → retry/backoff si falla score
  ├─ HistoryReader → contexto relevante
  └─ SummaryNode → final_summary + metrics
```

### 5.3 Challenge Flow - LLM Deliberation Workflows **NUEVO**

`/api/orchestrate` → `routing_agent.run(state, flow_type="challenge")`

```
  ├─ Creator → genera solución inicial
  ├─ Challenger → critica y encuentra problemas
  └─ Refiner → mejora basándose en críticas
```

**Casos de uso probados:**
- Validación de contraseñas con crítica automática
- Ping-pong efectivo entre modelos (3 ejecuciones secuenciales)
- Cache inteligente (51s carga inicial → 19s ejecuciones subsecuentes)

### 5.4 Multi-perspective Flow **CONFIGURADO**

`/api/orchestrate` → `routing_agent.run(state, flow_type="multi_perspective")`

```
         ├─ Security Expert
  Input ├─ Performance Expert → Synthesizer → Output
         └─ UX Expert
```

**Tipos de flow soportados:**
- `linear` - Pipeline tradicional
- `challenge` - Debate Creator→Challenger→Refiner  
- `multi_perspective` - Expertos paralelos + síntesis


## 6) Contratos

### 6.1 GenerationRequest

```python
class GenerationRequest(TypedDict):
    prompt: str
    model_key: str          # ej. "mistral7b"
    strategy: str           # ej. "optimized"
    max_tokens: int
    temperature: float
    task_type: str          # del Analyzer
    context: dict           # history, system, tools, etc.
    constraints: dict       # vram, latency_budget, cost_budget
```

### 6.2 GenerationResult

```python
class GenerationResult(TypedDict):
    text: str
    tokens_generated: int
    timings: dict           # load_time, inference_time, total
    usage: dict             # prompt_tokens, completion_tokens, cost (si aplica)
    source: str              # "local" | "openai" | "anthropic" | ...
    cache_hit: bool
    model_resolved: str     # id real (p.ej. hf repo o gpt-4o-mini)
```

---

## 7) Configuración y registro

* `config.py`: lectura de `.env` (claves API, límites por entorno, flags de compatibilidad).
* `providers/registries/model_registry.py`:

  * **Campos**: `key`, `family`, `provider`, `context_window`, `default_strategy`, `capabilities` (tool-calling, function-schemas, vision), `cost_profile` (opc), `min_vram_gb`.
* `providers/registries/strategy_registry.py`:

  * **Campos**: `name`, `sampling` (temp, top\_p, top\_k), `limits` (max\_tokens, timeout), `fallbacks`.

---

## 8) Logging, métricas, errores

* **Logging**: `utils/logger.py`, `utils/langgraph_logger.py` por nodo y por provider.
* **Métricas mínimas**: tiempos, cache, source, reintentos, score de validador.
* **Errores**: clasificar por `ProviderError`, `ValidationError`, `CapacityError`; política de backoff exponencial en Orchestrator.
* **Circuit Breaker**: registrar bloqueos y recuperaciones por proveedor.

---

## 9) Testing

* **Smoke** por provider (local, openai, anthropic) con mocks cuando no hay claves.
* **E2E** del grafo: `tests/langgraph/test_end_to_end_graph.py` con fixtures de `AgentState`.
* **Determinismo**: seeds donde aplique; snapshot tests del Summary.
* **Circuit Breaker tests**: verificar bloqueo y recuperación automática.

---

## 10) Plan de migración desde V2

1. Crear `providers/` y mover **wrappers locales** allí.
2. Implementar `provider_gateway.py` con adapter para **local** (primero).
3. Cambiar `execution_node.py` para usar `ProviderGateway`.
4. Mover `routing_agent.py` + `agent_state.py` a `orchestrator/` (o crear alias temporal de import).
5. Sustituir `constants/models.py` → `registries/model_registry.py`; `constants/strategies.py` → `registries/strategy_registry.py`.
6. Mantener `Direct Mode` en `/inference` apuntando al Gateway.
7. Agregar adapter remoto mínimo (OpenAI) detrás del Gateway.
8. Actualizar docs por archivo (sección 11).

---

## 11) Documentación por archivo (RAG-ready)

### api/server.py

**Rol:** Entrypoint FastAPI, registra routers, DI de config, logging global.
**Funciones/Clases:**
* `create_app() -> FastAPI`
* Hooks startup/shutdown (warmup, cleanup)
**Entradas/Salidas:** HTTP; no lógica de dominio.
**Depende de:** `config.py`, endpoints, logger.

### langchain_integration/langgraph/routing_agent.py

**Rol:** Orquestador principal SIMPLIFICADO - solo ejecuta grafos, no construye responses.
**Funciones principales:**
* `run_routing_agent(user_input, gateway, flow_type, verbose)` - Ejecuta flujos
* `run_orchestrator(prompt, flow_type)` - Wrapper para API con flow+metrics
**Líneas de código:** ~140 líneas (reducido de 250+ líneas)
**No hace:** construcción de flow para frontend, lógica compleja de métricas
**Depende de:** `orchestration/`, `agent_state.py`, `ProviderGateway`

### langchain_integration/langgraph/orchestration/graph_builder.py

**Rol:** Construcción modular de grafos LangGraph.
**Funciones principales:**
* `GraphBuilder.build_linear_flow_graph()` - Grafo tradicional
* `GraphBuilder.build_challenge_flow_graph()` - Flujo de debate LLM
* `GraphBuilder.build_graph_from_config(config)` - Constructor dinámico
* `get_graph_builder()` - Singleton pattern
**Características:** Nodos configurables, templates dinámicos, terminal node detection
**Depende de:** `graph_configs.py`, nodos existentes, `AgentState`

{
register_node(name, node_func) - Permite registrar nodos custom dinámicamente
_create_configurable_execution_node() - Crea nodos de ejecución con configuración específica
_find_terminal_nodes(config) - Detecta nodos sin edges salientes para conectar a END
build_routing_graph(flow_type) - Función de conveniencia para compatibilidad legacy

Gestión del estado Challenge Flow:

Guarda outputs en estructura state["challenge_flow"][node_id]
Persiste resultados en JSON: outputs/challenge_flows/{execution_id}.json
Mantiene compatibilidad con {node_id}_output keys

Routing condicional:

route_after_analysis() - Decide si usar monitor basado en task_type
Integración con route_after_validation() del validator node
Soporte para should_include_history() del history node
}

### langchain_integration/langgraph/orchestration/graph_configs.py

**Rol:** Configuraciones declarativas de flujos - sistema n8n-like.
**Estructuras principales:**
* `FlowConfig` - Definición completa de flujo
* `NodeConfig` - Configuración de nodo individual
* `EdgeConfig` - Conexiones entre nodos
**Funciones:**
* `get_challenge_flow_config()` - Config del flujo de debate
* `get_multi_perspective_flow_config()` - Config paralelo + síntesis
* `format_prompt_template()` - Templates con contexto dinámico
**Escalabilidad:** Preparado para JSON configs y base de datos

**Flujos soportados:**
- `linear` - Flujo tradicional secuencial
- `challenge` - Debate creator→challenger→refiner
- `multi_perspective` - Análisis paralelo con síntesis (próximamente)

**Persistencia:**
- Challenge flows se guardan en `outputs/challenge_flows/`
- Formato: `{execution_id}.json` con estructura completa

**Extensibilidad:**
- Nuevos flujos se agregan en `FLOW_CONFIGS`
- Nodos custom via `GraphBuilder.register_node()`
- Templates soportan cualquier variable del estado

### langchain_integration/langgraph/orchestration/flow_metrics.py

**Rol:** Construcción de métricas y flow data para frontend.
**Funciones principales:**
* `build_api_response(full_state, flow_type)` - Response completa para API
* `build_challenge_flow_nodes()` - Nodos específicos para challenge flow
* `build_execution_metrics()` - Métricas de performance
* `build_error_response()` - Manejo de errores
**Separación:** Extraído de routing_agent.py para modularidad
**Depende de:** Estado de ejecución, configuraciones de flow

### langchain_integration/langgraph/orchestration/__init__.py

**Rol:** Exports públicos del módulo de orquestación.
**Exports:**
* `GraphBuilder`, `get_graph_builder`, `build_routing_graph`
* `get_flow_config`, `list_available_flows`  
* `build_api_response`, `get_flow_summary`
**Propósito:** API limpia para importar funcionalidad de orquestación

### providers/provider_gateway.py

**Rol:** Orquesta proveedores locales/remotos.
**Funciones:** `generate(req)`, `resolve_model(req)`, `select_strategy(req)`
**Depende de:** registries, local manager, adapters remotos
**Estado:** Funcional, injection opcional via `hasattr(execution_mod, 'set_gateway')`

### providers/registries/model_registry.py

**Rol:** Catálogo de modelos disponibles y sus características.
**Funciones:** `get_model(key)`, `list_models()`, `supports_task(model_key, task_type)`
**Depende de:** ninguno (es data declarativa).

### langchain_integration/langgraph/nodes/execution_node.py

**Rol:** Nodo que ejecuta generación vía Provider Gateway.
**Funciones:** `execution_node(state)` 
**Integración:** Preparado para Gateway injection, fallback a modo directo
**Configurabilidad:** Soporta nodos configurables con templates dinámicos
**NO hace:** NO selecciona modelo, solo invoca al Gateway.
**Depende de:** `ProviderGateway`, `AgentState`.

### utils/gpu_guard.py

**Rol:** Monitor de recursos GPU (VRAM disponible).
**Funciones:** `get_available_vram()`, `can_load_model(size_gb)`
**Depende de:** `torch`, `pynvml`.

12) Roadmap V3
Fase 1 - Core ✅ COMPLETADO

 Estructura de carpetas V3
 Sistema de orquestación modular (orchestration/)
 Challenge Flow - LLM Deliberation Workflows funcionando
 Graph Builder con configuraciones declarativas
 Flow metrics separados de routing logic
 Linear flow refactorizado y simplificado
 Provider Gateway básico con injection opcional

Fase 1.5 - Refinamiento 🔧 EN PROGRESO

 Template context fix - Arreglar creator_output missing en Refiner
 Gateway injection consistency - Implementar set_gateway en execution_node
 JSON flow configs - Migrar de Python configs a JSON declarativo
 Multi-perspective flow testing - Probar flujo paralelo + síntesis

Fase 2 - Escalabilidad PRÓXIMO

 Frontend visual builder (n8n-like interface)
 Base de datos para flows dinámicos
 API endpoints para crear/editar flows
 Flow validation y testing framework

Fase 3 - Providers Remotos

 OpenAI adapter completo
 Anthropic adapter completo
 Tests de fallback automático entre providers
 Rate limiting unificado por provider

Fase 4 - Observabilidad Avanzada

 Métricas Prometheus para challenge flows
 Dashboard específico para debate flows
 Tracing distribuido cross-model

Estado actual: Challenge Flow probado exitosamente con 3 LLMs en secuencia, debate real entre modelos funcionando, arquitectura modular estable.

## 13) Variables de entorno (consolidadas)

```env
# Servicio
APP_ENV=dev
LOG_LEVEL=INFO
PORT=8001

# Providers
OPENAI_API_KEY=
ANTHROPIC_API_KEY=
DEFAULT_PROVIDER=local
DEFAULT_STRATEGY=optimized
MAX_TOKENS_DEFAULT=1024

# Local execution
MAX_VRAM_USAGE_GB=6.0
MODEL_CACHE_DIR=.cache/models

# Circuit Breaker
CIRCUIT_BREAKER_MAX_FAILURES=3
CIRCUIT_BREAKER_TIMEOUT=60
```

---

## 14) Decisiones de diseño

* **Gateway pattern** para aislar cambios de proveedor y facilitar AB-testing.
* **Registry declarativo** para evitar hardcode en varios archivos.
* **Orchestrator minimalista**: separar routing de ejecución.
* **Compatibilidad progresiva**: `Direct Mode` sigue operativo durante la migración.
* **Circuit Breaker pattern**: resiliencia ante fallos de proveedores externos.
* **Versionado de estado**: trazabilidad completa de requests a través del grafo.

---

## 15) Apéndice: Estado legado y acciones

* `constants/models.py`, `constants/strategies.py` → **DEPRECATE** en favor de registries.
* `local_models/*` y `langchain_integration/wrappers/*` → mover a `providers/local/`.
* `examples/`, `reference/`, `fix_hanging_model(Deprecado).py` → `archive/`.

## 16) Ejemplos de uso

### Orchestrated Mode
```bash
curl -X POST http://localhost:8001/api/orchestrate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Explain quantum computing",
    "strategy": "optimized",
    "max_tokens": 500
  }'


  ## 💡 **Sugerencias adicionales:**

1. **Agregar diagrama de arquitectura** (usando Mermaid):
```mermaid
graph TB
    Client[Cliente] --> API[FastAPI Server]
    API --> GW[Provider Gateway]
    API --> ORC[Orchestrator/LangGraph]
    ORC --> GW
    GW --> Local[Local Models]
    GW --> OAI[OpenAI API]
    GW --> ANT[Anthropic API]
    
    subgraph "Circuit Breaker"
        GW --> CB[Circuit Breaker]
        CB --> Fallback[Fallback Logic]
    end

## 17) LLM Deliberation Workflows **NUEVO**

### 17.1 Concepto

Sistema de **debate inteligente entre modelos** donde múltiples LLMs colaboran y se critican mutuamente para mejorar la calidad de las soluciones.

**Inspiración:** Replicar el proceso humano de revisión por pares, pero automatizado entre modelos IA.

### 17.2 Challenge Flow - Implementado

**Flujo:** Creator → Challenger → Refiner

```python
# Ejemplo de ejecución real:
Creator (Mistral): Genera función de validación de contraseñas
↓
Challenger (Mistral): "La función no verifica repeticiones consecutivas..."
↓  
Refiner (Mistral): Mejora función basándose en críticas
```

**Prompts templates utilizados:**
- **Creator**: `"Genera una solución para: {user_input}"`
- **Challenger**: `"Analiza críticamente esta solución... ¿Hay problemas de seguridad?"`
- **Refiner**: `"Mejora la solución basándote en las críticas recibidas..."`

### 17.3 Casos de Uso Probados

**Validación de contraseñas (Ejecutado exitosamente):**
1. Creator propuso función básica con regex
2. Challenger identificó: repeticiones no validadas, hardcoded word lists
3. Refiner (pendiente contexto fix) mejoraría implementación

**Métricas de ejecución:**
- Tiempo total: ~90 segundos (3 modelos secuenciales)
- Cache hit rate: 67% (modelo ya cargado)
- Críticas específicas y técnicamente válidas generadas

### 17.4 Patrones de Debate Configurables

**Multi-perspective Flow:**
```
Input → [Security Expert, Performance Expert, UX Expert] → Synthesizer
```

**Adversarial Flow (futuro):**
```
Proposal → Red Team → Blue Team → Arbitrator
```

**Consensus Flow (futuro):**
```
Input → [Model A, Model B, Model C] → Voting Mechanism → Best Answer
```

### 17.5 Issues Técnicos Identificados

1. **Context sharing**: `creator_output` no disponible para Refiner
2. **Template expansion**: Necesita `{previous_output}`, `{challenger_output}` 
3. **Model routing**: Un solo modelo (Mistral) para todos los roles

### 17.6 Ventajas Observadas

✅ **Críticas reales**: Challenger encontró problemas técnicos válidos
✅ **Especificidad**: Sugerencias concretas de mejora  
✅ **Escalabilidad**: Fácil agregar nuevos tipos de debate
✅ **Modularidad**: Nodos independientes y reutilizables

### 17.7 Próximos Experimentos

- **Multi-model**: Usar diferentes modelos por rol (Claude, GPT, Mistral)
- **Especialización**: Crear expertos por dominio (security, performance, UX)
- **Iteración**: Múltiples rondas de crítica/refinamiento
- **Evaluación**: Métricas automáticas de mejora de calidad