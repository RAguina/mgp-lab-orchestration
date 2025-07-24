# Carpeta: utils/analysis

Scripts utilitarios para el análisis manual de outputs, logs y métricas del laboratorio.

## Estructura

- `log_reader.py`: lee y resume todos los logs JSON de un modelo.
- `metrics_reader.py`: carga y analiza métricas CSV.
- `output_comparator.py`: compara outputs de diferentes modelos para el mismo prompt.

> **NOTA:** Estos scripts NO son parte del pipeline automático de procesamiento; están pensados para debug, exploración y análisis exploratorio. Si necesitas análisis productivo, implementá un worker real en `workers/analyzers/`.

---

# Carpeta: workers/analyzers

Workers (componentes reusables y escalables) para el análisis automático de métricas y outputs generados.

## Estructura

- `metrics_analyzer.py`: worker base para analizar métricas, tiempos, calidad de outputs, y comparación entre modelos. Listo para ser invocado por un pipeline automatizado (o por un orquestador humano/manual).

> **TIP:** Todos los workers deben tener una interfaz clara y desacoplada: input (dict), output (dict, JSON, archivo). Así pueden integrarse fácilmente a cualquier pipeline.
