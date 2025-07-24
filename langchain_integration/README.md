# IA Agent Lab - LangGraph Integration

Este entorno está diseñado para ejecutar y orquestar múltiples modelos de lenguaje locales utilizando LangChain y LangGraph. La carpeta `langgraph_integration/` contiene una arquitectura modular y escalable basada en nodos y grafos de decisión. Esta guía documenta la estructura y propósito de cada archivo clave.

---

## 📊 Estructura general

```
langchain_integration/
├── langgraph/
│   ├── routing_agent.py
│   ├── validators.py
│   ├── local_llm_node.py
│   └── llm_graph.py
├── tools/
│   ├── lab_tools.py
│   └── history_tools.py
└── wrappers/
    └── local_model_wrapper.py
```

---

## 🔍 langgraph/

### `routing_agent.py`

Agente principal de múltiples nodos con lógica condicional y validaciones robustas.

* \`\`: Estado compartido entre nodos. Incluye `input`, `output`, `task_type`, `strategy`, `retry_count`, `last_output`, etc.
* **Nodos**:

  * `analyzer`: Determina el tipo de tarea y modelo.
  * `monitor`: Verifica recursos y decide estrategia (`standard` u `optimized`).
  * `executor`: Ejecuta el modelo local usando `local_llm_node`.
  * `validator`: Verifica si el output es válido o requiere reintento.
  * `history`: Lee el último output previo del modelo desde disco.
  * `summarizer`: Genera resumen final del proceso.
* **Routing condicional**:

  * Después de `analyzer`: decide si salta el nodo `monitor`.
  * Después de `validator`: retry o continuar.
  * Después de `history`: decide si incluir historial.

### `validators.py`

* \`\`: Verifica la calidad del output generado. Aplica retry si es muy corto o contiene errores.
* \`\`: Devuelve `retry_execution` o `continue` según el estado `retry`.
* `MAX_RETRIES` configurable.

### `local_llm_node.py`

* Contiene `build_local_llm_tool_node()`, una función que genera un `ToolNode` configurado para ejecutar modelos locales usando wrappers.
* Internamente llama al `LocalModelWrapper`, que abstrae lógica de carga, inferencia, logging y métricas.

### `llm_graph.py`

* Versión simple y directa para probar un modelo local en un grafo lineal (`input -> model -> output`).
* Sirve como entorno mínimo de prueba.

---

## 🔧 tools/

### `lab_tools.py`

Herramientas reutilizables como LangChain Tools o nodos:

* `VRAMMonitorTool`: Devuelve estado actual de la GPU.
* `ModelSelectorTool`: Devuelve un modelo sugerido según input.
* `get_lab_tools()`: Devuelve todas las herramientas disponibles.

Estas herramientas son usadas tanto dentro de nodos como en evaluación previa al grafo.

### `history_tools.py`

* `history_reader_node`: Nodo para leer el último output generado por el modelo usado (`outputs/modelo/runs/*.txt`).
* `should_include_history()`: Decide si se debe incluir historial (por ejemplo, para tareas `code` o `analysis`).
* Exporta `HistoryReaderNode` como `RunnableLambda`.

---

## 🏠 wrappers/

### `local_model_wrapper.py`

* Wrapper que abstrae la ejecución del modelo.
* Maneja:

  * Carga desde HuggingFace con `transformers`.
  * Elección de estrategia (`standard`, `optimized`, `streaming`).
  * Registro estructurado (métricas, VRAM, logs por modelo).
* Incluye validaciones, logger personalizado y persistencia del output.

---

## 🏆 Recomendaciones de uso

* Ejecutar `routing_agent.py` para tener todo el sistema con routing activo.
* Usar `llm_graph.py` si se desea probar un solo modelo sin lógica condicional.
* Todos los logs se guardan en `outputs/<modelo>/runs/`, con métricas separadas por ejecución.

---

## 🌐 Requisitos

* Python 3.12
* LangGraph >= 0.2.50
* Transformers, Accelerate, Pydantic, etc. (ver `requirements.txt`)
* GPU con al menos 8 GB VRAM (recomendado)

---

## 🚀 Roadmap futuro

*

---

Este entorno permite testear arquitecturas de agentes autónomos con decisiones dinámicas y evaluación automática del output. Su diseño modular facilita la incorporación de nuevos modelos, herramientas y estrategias.
