# 🤖 NODES CATALOG - AI Agent Lab V3

> **Referencia rápida:** Catálogo de nodos especializados en el sistema de orquestación LangGraph

---

## 🏗️ ARQUITECTURA DE NODOS

El sistema usa **nodos especializados** en lugar de agentes tradicionales, organizados en flujos configurables.

### 📋 NODOS CORE (Linear Flow)

| Nodo | Rol | Input Principal | Output Principal | Archivo |
|------|-----|-----------------|------------------|---------|
| **TaskAnalyzer** | Clasifica tarea y selecciona modelo/estrategia | `user_input` | `task_type`, `selected_model`, `strategy` | `task_analyzer_node.py` |
| **ResourceMonitor** | Monitorea VRAM y recursos GPU | `task_type` | `vram_status`, `memory_available` | `resource_monitor_node.py` |  
| **ExecutionNode** | Ejecuta generación vía ProviderGateway | `prompt`, `model`, `strategy` | `output`, `execution_metrics` | `execution_node.py` |
| **OutputValidator** | Valida calidad y maneja reintentos | `output`, `task_type` | `validation_score`, retry signals | `output_validator_node.py` |
| **HistoryReader** | Lee contexto histórico relevante | `user_input` | `history_context` | `history_reader_node.py` |
| **SummaryNode** | Genera resumen final con métricas | `output`, `metrics` | `final_summary`, `flow_metrics` | `summary_node.py` |

### 🥊 NODOS CHALLENGE FLOW (LLM Deliberation)

| Nodo | Rol | Template | Modelo Sugerido | Config |
|------|-----|-----------|-----------------|---------|
| **Creator** | Genera solución inicial | `"Genera una solución para: {user_input}"` | `qwen` | `temperature: 0.7` |
| **Challenger** | Critica y encuentra problemas | Analiza críticamente + security/structure check | `mistral` | `temperature: 0.3` |
| **Refiner** | Mejora basándose en críticas | Mejora solución con críticas recibidas | `claude` | `temperature: 0.5` |

### 🔮 NODOS MULTI-PERSPECTIVE (Configurado)

| Nodo | Especialización | Focus | Estado |
|------|----------------|--------|--------|
| **SecurityExpert** | Análisis de seguridad | Vulnerabilidades, best practices | 🔧 Configurado |
| **PerformanceExpert** | Análisis de performance | Optimización, escalabilidad | 🔧 Configurado |
| **UsabilityExpert** | Análisis UX | Usabilidad, experiencia usuario | 🔧 Configurado |
| **Synthesizer** | Síntesis final | Combina todas las perspectivas | 🔧 Configurado |

---

## 🔄 FLUJOS DISPONIBLES

### 1️⃣ **Linear Flow** (Tradicional)
```
TaskAnalyzer → ResourceMonitor → ExecutionNode → OutputValidator → HistoryReader → SummaryNode
                    ↓ (skip_monitor)
                ExecutionNode
```

### 2️⃣ **Challenge Flow** ✅ (LLM Deliberation)
```
Creator → Challenger → Refiner
```

### 3️⃣ **Multi-Perspective Flow** 🔧 (Paralelo + Síntesis)
```
         ┌─ SecurityExpert ─┐
Input ──┼─ PerformanceExpert ┼─→ Synthesizer → Output
         └─ UsabilityExpert ─┘
```

---

## ⚙️ CARACTERÍSTICAS TÉCNICAS

### **Configurabilidad:**
- ✅ Templates dinámicos con contexto
- ✅ Model hints por nodo  
- ✅ Parámetros configurables (temperature, etc.)
- ✅ Routing condicional

### **Estado del Sistema:**
- ✅ Challenge Flow probado exitosamente
- ✅ Persistencia en JSON estructurado  
- ✅ Cache inteligente (51s → 19s)
- ✅ Métricas detalladas por nodo

### **Extensibilidad:**
- ✅ Registry de nodos dinámico (`GraphBuilder.register_node()`)
- ✅ Configuraciones declarativas
- 🔧 Soporte futuro para JSON configs

---

## 🎯 CASOS DE USO PROBADOS

### **Challenge Flow:**
- ✅ Validación de contraseñas con crítica automática
- ✅ Debate efectivo entre 3 modelos secuenciales
- ✅ Críticas técnicamente válidas y específicas

### **Linear Flow:**
- ✅ Inferencia simple y orquestada
- ✅ Resource monitoring dinámico
- ✅ Validation y retry automático

---

*Documentación actualizada: $(date)*  
*Fuente: Análisis del código actual V3*