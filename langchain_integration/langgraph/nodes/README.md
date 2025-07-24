# Nodos del Agente LangGraph

Esta carpeta contiene los **nodos especializados** que forman parte del grafo de ejecución del agente LangGraph. Cada nodo encapsula una función específica dentro del pipeline de orquestación de modelos LLM.

---

## 🧠 1. `task_analyzer_node.py`

**Responsabilidad:** Detectar el tipo de tarea (code, chat, technical, analysis, creative) y seleccionar un modelo adecuado usando `ModelSelectorTool`.

**Entradas:**

* `input` (texto del usuario)
* `messages` (historial acumulado)

**Salidas:**

* `task_type` asignado
* `selected_model`

**Notas:**

* Clasifica tareas usando palabras clave.
* Agrega logs de selección al historial.

---

## 🧪 2. `resource_monitor_node.py`

**Responsabilidad:** Evaluar la disponibilidad de VRAM y sugerir una estrategia de ejecución (`standard` u `optimized`).

**Entradas:**

* Sin dependencias externas al estado

**Salidas:**

* `strategy`
* `should_optimize`
* `vram_status`

**Notas:**

* Usa `VRAMMonitorTool`
* Lógica defensiva si no se puede parsear la VRAM

---

## 🚀 3. `execution_node.py`

**Responsabilidad:** Invocar al modelo local usando la estrategia y el modelo previamente seleccionados.

**Entradas:**

* `selected_model`
* `strategy`

**Salidas:**

* `output`
* `analysis_result`

**Notas:**

* Usa `build_local_llm_tool_node`
* Devuelve errores si falló la ejecución

---

## ✅ 4. `output_validator_node.py`

**Responsabilidad:** Validar la salida del modelo. Si se detectan errores, se activa `retry`.

**Entradas:**

* `output`, `retry_count`

**Salidas:**

* `retry`, `retry_count` incrementado

**Notas:**

* Detecta patrones comunes de fallos
* Está diseñado para usarse justo después de `execution_node`

---

## 📚 5. `history_reader_node.py`

**Responsabilidad:** Recuperar el último output guardado (si existe) para fines de referencia o sumarización.

**Entradas:**

* `selected_model` (identificador de modelo usado)

**Salidas:**

* `last_output`

**Notas:**

* Busca en la carpeta `outputs/`
* Diseñado para tareas tipo `analysis`, donde el contexto previo puede ser útil

---

## 🧾 6. `summary_node.py`

**Responsabilidad:** Generar un resumen final del proceso de ejecución.

**Entradas:**

* Casi todo el estado acumulado (`task_type`, `strategy`, `vram_status`, etc.)

**Salidas:**

* `final_summary`

**Notas:**

* Concatenación textual para trazabilidad humana
* Útil para logging o dashboards

---

## 📌 Estructura Recomendada

```bash
langchain_integration/
└── langgraph/
    ├── nodes/
    │   ├── task_analyzer_node.py
    │   ├── resource_monitor_node.py
    │   ├── execution_node.py
    │   ├── output_validator_node.py
    │   ├── history_reader_node.py
    │   └── summary_node.py
```

Cada archivo define una función principal (el nodo) y puede incluir funciones auxiliares, validaciones o llamadas a herramientas específicas.

---

## 🧪 Testing

Cada nodo está cubierto por pruebas unitarias ubicadas en:

```
tests/langgraph/
```

Ejemplos:

* `test_task_analyzer_node.py`
* `test_output_validator_node.py`
* `test_history_reader_node.py`

---

## ✅ Siguientes pasos sugeridos

* Agregar decoradores de logging estructurado.
* Separar nodos en subcarpetas si crecen (e.g. `validation/`, `llm/`, `tools/`)
* Integrar más tipos de nodos como: `reranker`, `post_processor`, `metrics_logger`.
