# CODE-INVENTORY-V3 — Descripción por archivo (primera tanda)

> Objetivo: documentar **qué hace** cada archivo, sus **contratos públicos**, dependencias y **refactors sugeridos** para V3. Esta guía es “RAG‑ready”.

---

## 1) `api/server.py`

**Rol:** Entrypoint FastAPI. Configura app, CORS, routers, logging y ciclo de vida (lifespan). Expone `/`, `/health`, `/models`.

**Funciones/Clases clave:**

* `lifespan(app)`: inicializa servicios (`inference`, `cache`, `system`) y realiza cleanup (`model_manager.cleanup_all()`).
* Endpoints en server: `GET /`, `GET /health`, `GET /models`.
* Inicializa singletons: `executor = ModelExecutor(...)`, `model_manager = get_model_manager()`.

**Entradas/Salidas:**

* **In:** HTTP requests (REST).
* **Out:** JSON (estado, modelos, health), inicialización de servicios.

**Dependencias internas relevantes:**

* `local_models.model_executor`, `local_models.model_manager.MODELS` y `get_model_manager()`.
* `api.endpoints.{inference, orchestrator, cache, system}`.
* `utils.gpu_guard.get_gpu_info`.

**Contratos públicos:**

* API estable: `/`, `/health`, `/models` (y routers montados).
* `HealthResponse` (pydantic) como esquema de `/health`.

**Olores y riesgos:**

* **Acoplamiento a MODELS locales:** usa `local_models.model_manager.MODELS` directo.
* **Path hacking:** `sys.path.insert(0, LAB_ROOT)`.
* **Conocimiento del orquestador:** consulta `orchestrator.ORCHESTRATOR_ENABLED` (bandera cruzada).
* **Responsabilidades múltiples:** inicializa servicios + expone endpoints + conoce cache/VRAM.

**Acciones sugeridas (V3):**

* Inyectar **ProviderGateway** y **ModelRegistry** vía DI en startup (no importar MODELS directo).
* `GET /models` debe leer de `providers/registries/model_registry.py` + estado de `provider_gateway`.
* Evitar `sys.path` hack con paquetes instalables o `src/` layout.
* Mover endpoints `/health` richer a `api/endpoints/system.py` y dejar en server un health básico (o mantener schema pero consultando servicios vía DI).

**Notas de migración:**

* Mantener compatibilidad de rutas; reimplementar la lógica interna usando Gateways/Registries.

---

## 2) `langchain_integration/langgraph/routing_agent.py`

**Rol:** Construye y ejecuta el grafo de LangGraph. Hace routing condicional entre nodos “workers” y arma una vista del flujo para el frontend.

**Funciones/Clases clave:**

* `build_routing_graph() -> CompiledGraph`: registra nodos (`analyzer`, `monitor`, `executor`, `validator`, `history`, `summarizer`) y sus edges/condiciones.
* `route_after_analysis(state) -> Literal["monitor" | "skip_monitor"]`: rama según `task_type`.
* `run_routing_agent(user_input, verbose) -> AgentState`: crea estado inicial, invoca grafo y agrega métricas/logging.
* `run_orchestrator(prompt) -> dict`: envoltorio para API; empaqueta `flow.nodes/edges`, `metrics` y `output`.

**Entradas/Salidas:**

* **In:** `user_input`/`prompt` (texto), `AgentState` inicial.
* **Out:** estado final con `output`, `task_type`, `selected_model`, `execution_metrics`, `flow`.

**Dependencias internas:**

* `langchain_integration.langgraph.agent_state` (`AgentState`, `create_initial_state`).
* Nodos: `task_analyzer_node`, `resource_monitor_node`, `execution_node`, `output_validator_node`, `HistoryReaderNode`, `summary_node`.
* `RunnableLambda` (LangChain/LangGraph) para envolver nodos.

**Contratos públicos:**

* API de alto nivel: `run_orchestrator(prompt)` para backend.
* Estructura `flow { nodes[], edges[] }` para UI.

**Olores y riesgos:**

* **Logging básico global** dentro del archivo (`logging.basicConfig` + `FileHandler`), puede duplicar configuración general.
* **Demasiadas responsabilidades:** además de routing, forma respuesta HTTP‑friendly, accede a métricas, compone “flow”.
* **Acoplamiento a selección de modelo:** lee `selected_model` del estado; `execution_node` probablemente elige modelo.

**Acciones sugeridas (V3):**

* Extraer `run_orchestrator()` a un módulo API/service (p. ej. `orchestrator/service.py`) que consuma el grafo y construya la respuesta.
* `execution_node` debe usar **ProviderGateway** y nunca llamar modelos directos.
* Centralizar logging en `utils/logger.py`; aquí solo `logger = get_logger("orchestrator", ...)`.
* Dejar este archivo con: construcción del grafo + funciones de routing + `create_graph()`.

**Notas de migración:**

* Cuando exista `providers/provider_gateway.py`, reemplazar dependencia de ejecución.

---

## 3) `local_models/model_manager.py`

**Rol:** Singleton que gestiona cache de modelos locales (HF), carga bajo demanda por estrategia, aplica política LRU bajo presión de VRAM.

**Funciones/Clases clave:**

* `load_model(model_key, strategy, **kwargs) -> LoadedModel`: carga o reutiliza; mide tiempos y VRAM.
* `unload_model(model_key, strategy)`, `cleanup_all()`.
* `is_model_loaded()`, `get_model()`, `get_loaded_models()`, `get_memory_stats()`.
* `set_max_vram_usage(max_gb)`.

**Entradas/Salidas:**

* **In:** `model_key`, `strategy`, kwargs de carga (device\_map, etc.).
* **Out:** `LoadedModel` (model, tokenizer, pipeline, metadata).
* **Efectos:** consumo/liberación de VRAM (CUDA), logging, GC.

**Dependencias internas:**

* `wrappers.hf_pipeline_wrappers.{standard, optimized, streaming}`.
* `utils.gpu_guard.{get_gpu_info, clear_gpu_memory}`.

**Contratos públicos:**

* API de gestión local de modelos para ser usada por capas superiores (hoy: endpoints; mañana: ProviderGateway).

**Olores y riesgos:**

* **Duplicación de MODELS** (hardcode en este archivo). Ya existe `constants/models.py` y otros lugares.
* **Medición de presión de memoria** dependiente de `get_gpu_info()` (heurística).
* **Singleton con locking** correcto pero acopla a FS logs y utilidades.

**Acciones sugeridas (V3):**

* Mover a `providers/local/manager/` y ocultarlo detrás del **ProviderGateway**.
* Eliminar MODELS local → usar `providers/registries/model_registry.py`.
* Añadir señalización de `source="local"` y normalización de resultados (timings/usage) para Gateway.

**Notas de migración:**

* Mantener firma pública hasta completar Gateway.

---

## 3.b) `langchain_integration/wrappers/local_model_wrapper.py`

**Rol:** Implementación `LLM` de LangChain para usar modelos locales del lab (opcional). Soporta dos modos: `ModelLauncher` o carga directa vía `hf_pipeline_wrappers`.

**Funciones/Clases clave:**

* Clase `LocalModelWrapper(LLM)`: `_call()`, `_stream()`, `_load_model_direct()`, `_call_with_launcher()`, `get_metrics()`.
* Factory `create_local_llm(model_key, strategy, **kwargs)`.

**Entradas/Salidas:**

* **In:** `prompt`, config de generación (max\_tokens, temp, top\_p).
* **Out:** `str` generado (y métricas internas consultables).

**Dependencias internas:**

* `local_models.llm_launcher.ModelLauncher` (si está), `hf_pipeline_wrappers`, `utils.gpu_guard`, `utils.logger`.

**Contratos públicos:**

* Interfaz LLM estándar de LangChain; útil si algún nodo/flow quiere LLMs LC‑native.

**Olores y riesgos:**

* **Otra copia de MODELS** (fallback).
* **Doble vía de carga** (launcher vs directa) puede diverger del `ModelManager`.
* **Bypass del pool/cache** si se usa carga directa.

**Acciones sugeridas (V3):**

* Convertirlo en **adapter** que delegue siempre en `ProviderGateway` (o eliminar si no se usa en el grafo).
* Quitar tabla MODELS local → `model_registry`.
* Si se mantiene, forzar a usar `ModelManager` para no duplicar cache.

---

# Próximos pasos

1. Confirmar estos análisis.
2. Documentar **`api/endpoints/inference.py`** y **`langchain_integration/langgraph/nodes/execution_node.py`** (críticos para Gateway).
3. Crear `providers/` + `provider_gateway.py` (skeleton) y `registries/` (model/strategy). Mover `ModelManager` a `providers/local/manager/`.
4. Actualizar `api/server.py` y `routing_agent.py` para consumir Gateway por DI.



Documentaste a proposito los duplicados?
He creado un nuevo bloque en **CODE-INVENTORY-V3** para documentar estos dos archivos:

---

## `api/endpoints/inference.py`

**Rol:** Provee endpoints REST para inferencia simple y listado de modelos disponibles.

**Funciones/Clases clave:**

* `init_inference_service`: inicializa `executor` y `model_manager` globales.
* `InferenceRequest`: schema Pydantic para input.
* `InferenceResponse`: schema para output con métricas.
* `run_inference`: POST `/inference` que ejecuta un modelo local vía `ModelExecutor`.
* `list_models`: GET `/inference/models` que devuelve metadatos de modelos cargados/disponibles.

**Entradas/Salidas:**

* Entrada: JSON con prompt, modelo, estrategia, etc.
* Salida: JSON con output, métricas y metadata de ejecución.

**Dependencias internas:** `local_models.model_executor`, `local_models.model_manager`, `utils.gpu_guard`.

**Olores:** Uso de variables globales para `executor` y `model_manager`, acoplamiento fuerte con `MODELS`.

**Acciones sugeridas:**

* Inyectar dependencias vía DI en vez de globales.
* Extraer validación de modelos y ejecución a capa de servicio.

**Contrato público:** `/inference` y `/inference/models`.

---

## `langchain_integration/langgraph/nodes/execution_node.py`

**Rol:** Nodo LangGraph especializado en ejecutar modelos con logging y cache inteligente.

**Funciones/Clases clave:**

* `get_executor`: inicializa y cachea `ModelExecutor`.
* `execution_node`: ejecuta un modelo según parámetros en `AgentState`, con métricas y calidad estimada.
* `analyze_output_quality`: evalúa heurísticamente la calidad del output.
* `execution_node_legacy`: alias para compatibilidad.

**Entradas/Salidas:**

* Entrada: `AgentState` con prompt, modelo, estrategia, tipo de tarea, etc.
* Salida: `AgentState` actualizado con output, métricas y mensajes.

**Dependencias internas:** `local_models.model_executor`, `langchain_integration.langgraph.agent_state`.

**Olores:** Inserción manual del LAB\_ROOT en `sys.path`, global `_executor`, acoplamiento directo a `ModelExecutor`.

**Acciones sugeridas:**

* Inyectar `executor` desde orquestador/gateway.
* Eliminar `sys.path` y usar imports relativos o package root.
* Extraer mapeo `max_tokens` y lógica de calidad a servicios utilitarios.

**Contrato público:** Función `execution_node(state)` como RunnableNode en LangGraph.

---

Si quieres, puedo continuar con este formato para el resto de los archivos críticos y generar un **ARCHITECTURE-CODE-DOC.md** centralizado para tu RAG.

---

## 4) `api/endpoints/inference.py`

**Rol:** Endpoint REST para **inferencia directa** usando `ModelExecutor` + `ModelManager`. Centraliza validación de request y normaliza la respuesta.

**Funciones/Clases clave:**

* `init_inference_service(executor, model_manager)`: inyección de dependencias (DI) desde `server.py`.
* `run_inference(request: InferenceRequest) -> InferenceResponse`: ejecuta el flujo directo y devuelve métricas.
* `list_models() -> List[ModelInfo]`: expone catálogo + estado de carga (cache) por estrategia.
* Schemas: `InferenceRequest`, `InferenceResponse`, `ModelInfo`.

**Entradas/Salidas:**

* **In:** JSON con `prompt`, `model`, `strategy`, `max_tokens`, `temperature`.
* **Out:** `InferenceResponse` con `output`, `metrics` (incluye tiempos, cache), `success`.

**Dependencias internas relevantes:**

* `local_models.model_executor.ModelExecutor` (inyección), `local_models.model_manager.get_model_manager()` y `MODELS`.
* `utils.gpu_guard.get_gpu_info` (solo para logging indirecto/consistencia de métricas).

**Contratos públicos:**

* `POST /inference/` y `GET /inference/models`.
* **Estructura** de `metrics` tomada de `ModelExecutor` (usa claves como `load_time_sec`, `inference_time_sec`, `total_time_sec`, `cache_hit`, `tokens_generated`).

**Olores y riesgos:**

* **Acoplamiento a MODELS locales** (usa tabla global); deberá pasar a `model_registry`.
* **Lógica de negocio** (validación de modelo/estrategia) mezclada en el endpoint.
* **Naming de métricas** distinto al usado por nodos del grafo (ej. `load_time` vs `load_time_sec`).

**Acciones sugeridas (V3):**

* Reemplazar `MODELS` por `providers/registries/model_registry.list()`.
* `run_inference` debería delegar en **ProviderGateway.generate()** (modo directo) para unificar métrica y resolución de proveedor.
* Normalizar nombres de métricas a un contrato único (`load_time_ms`, `inference_time_ms`, `total_time_ms`) y dejar la conversión a segundos a la capa de presentación si hace falta.

**Contrato público (post‑V3):**

* Request: `GenerationRequest` (subset) + `mode="direct"`.
* Response: `GenerationResult` normalizado + `success`.

---

## 5) `langchain_integration/langgraph/nodes/execution_node.py`

**Rol:** Nodo del grafo que **ejecuta la generación**. Hoy delega en `ModelExecutor` local; registra métricas y sintetiza mensajes.

**Funciones/Clases clave:**

* `get_executor()` (singleton liviano del `ModelExecutor`).
* `execution_node(state: AgentState) -> AgentState`: prepara max\_tokens según `task_type`, llama a `executor.execute()`, empaqueta `execution_metrics`.
* `analyze_output_quality(output, task_type) -> int`: heurística simple de calidad.
* `execution_node_legacy(state)` alias.

**Entradas/Salidas:**

* **In (desde AgentState):** `selected_model`, `strategy`, `input` (prompt), `task_type`.
* **Out (AgentState actualizado):** `output`, `messages[]`, `execution_metrics{ cache_hit, load_time, inference_time, total_time, tokens_generated, model_used, strategy_used, quality_score }`.

**Dependencias internas relevantes:**

* `local_models.model_executor.ModelExecutor` (acoplamiento directo a local).
* Inserta `sys.path` al LAB\_ROOT (hack de import).

**Olores y riesgos:**

* **Acoplamiento a implementación local**; ignora proveedores remotos.
* **Doble semántica de métricas** (nombres `*_sec` vs sin sufijo); inconsistencia con `/inference`.
* **Heurística de calidad** dentro del nodo; debería vivir en `output_validator_node`.
* **Hacks de path** (`sys.path.insert(0, LAB_ROOT)`), frágil en empaquetado.

**Acciones sugeridas (V3):**

* Hacer `execution_node` un **cliente del ProviderGateway**:

  * Input mínimo requerido: `GenerationRequest` derivado del `AgentState`.
  * Output: `GenerationResult` normalizado (misma forma que `/inference`).
* Mover `analyze_output_quality` a `output_validator_node` y conservar aquí solo el paso de ejecución + timing.
* Quitar `sys.path` hack adoptando layout `src/` o instalando el paquete del lab (`pip install -e .`).
* Unificar claves de métricas en **ms**: `load_time_ms`, `inference_time_ms`, `total_time_ms`.

**Contrato público (post‑V3):**

* **Input (desde state):**

  * `task_type: str`, `input: str`, `selected_model: str`, `strategy: str`, `context?: dict`, `constraints?: dict`.
* **Output (state actualizado):**

  * `output: str`, `execution_metrics: GenerationResult.timings/usage + {source}`; no incluye scoring.

**Hooks/Errores:**

* Captura excepciones y marca `execution_metrics.failed=True`. El Orchestrator debe decidir **retry/backoff**.

---
