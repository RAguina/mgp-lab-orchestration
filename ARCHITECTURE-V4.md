# AI Agent Lab - Architecture V4

> **Estado:** Documentación actualizada y verificada basada en código real  
> **Fecha:** Septiembre 2024  
> **Versión anterior:** [ARCHITECTURE-V3.md](docs/archive/ARCHITECTURE-V3.md) (histórico)

---

## 🎯 Executive Summary

**AI Agent Lab V4** es un sistema de orquestación de LLMs locales con flujos configurables de deliberación entre modelos.

### **¿Qué hace hoy?**
- ✅ **Inferencia local** con modelos HuggingFace (Mistral, Llama3, DeepSeek)
- ✅ **Challenge Flow** - Debate automático entre 3 LLMs (Creator→Challenger→Refiner)
- ✅ **Linear Flow** - Pipeline tradicional con análisis, validación y contexto histórico
- ✅ **API REST** para integración frontend/backend
- ✅ **Provider Gateway** para abstracción de modelos
- ✅ **Cache inteligente** y gestión de recursos GPU

### **¿Qué NO hace?**
- ❌ **Providers remotos** (OpenAI, Anthropic) - solo referencias de código
- ❌ **Multi-Perspective Flow** - configurado pero no probado
- ❌ **Frontend integrado** - desconectado tras refactor V3
- ❌ **Persistencia de conversaciones** - solo outputs individuales

### **¿Por qué V4?**
V3 tenía aspiraciones no implementadas mezcladas con funcionalidad real. V4 documenta **exactamente lo que existe y funciona** hoy.

---

## 🚧 Scope & Non-Goals

### **✅ In Scope (V4)**
- Orquestación de modelos locales HuggingFace
- Challenge Flow para deliberación LLM
- API REST estable para inferencia
- Gestión de recursos GPU y VRAM
- Sistema de nodos modulares LangGraph

### **❌ Non-Goals / Deprecated**
- **Providers remotos** → Futuro (V5+)
- **Multi-Perspective Flow** → Experimental, no production-ready  
- **AutoGen integration** → Archivado en `/archive/`
- **Direct inference scripts** → Referencias inexistentes removidas
- **Strategy registry** → Usando `constants/strategies.py` (legacy pero funcional)

---

## 🏗️ Modules & Ownership

### **🧪 Lab Core (Local Execution)**
- **Purpose:** Gestión de modelos locales y ejecución
- **Entrypoints:** `local_models/model_manager.py`, `local_models/model_executor.py`
- **Owner:** Sistema de cache y lifecycle de modelos HF
- **Status:** ✅ Estable, production-ready

### **🔄 Orchestration Engine (LangGraph)**
- **Purpose:** Flujos configurables entre múltiples LLMs
- **Entrypoints:** `langchain_integration/langgraph/routing_agent.py`
- **Owner:** Graph building, flow execution, metrics
- **Status:** ✅ Challenge Flow probado, Linear Flow estable

### **🌉 Provider Gateway**
- **Purpose:** Abstracción unificada para diferentes providers
- **Entrypoints:** `providers/provider_gateway.py`
- **Owner:** Request routing, model resolution
- **Status:** ✅ Local provider operativo, remote providers pendientes

### **🔌 REST API (FastAPI)**
- **Purpose:** Endpoints para frontend y integración externa
- **Entrypoints:** `api/server.py`
- **Owner:** HTTP interface, request/response handling
- **Status:** ✅ Funcional, endpoints `/inference` y `/orchestrate`

### **🖥️ Frontend (Desconectado)**
- **Purpose:** UI para interacción con flujos
- **Status:** ⚠️ Existe pero desconectado tras refactor V3
- **Next:** Requiere reconexión con nuevos endpoints

---

## 📋 Contratos

### **Unified Inference Request**
```typescript
interface InferenceRequest {
  prompt: string
  model?: "mistral7b" | "llama3" | "deepseek7b" | "deepseek-coder"  // default: "mistral7b"
  strategy?: "standard" | "optimized" | "streaming"                 // default: "optimized"
  flow_type?: "linear" | "challenge"                                // default: "linear"
  max_tokens?: number                                               // default: 1024
  temperature?: number                                              // default: 0.7
}
```

### **Unified Response**
```typescript
interface InferenceResponse {
  success: boolean
  output: string
  metrics: {
    total_time: number           // milliseconds
    cache_hit: boolean
    tokens_generated: number
    model_used: string
    source: "local"              // future: "openai" | "anthropic"
  }
  flow?: {                       // only for orchestrated flows
    type: string
    nodes: FlowNode[]
    execution_id: string
  }
  error?: ErrorInfo
}
```

### **Error Handling**
```typescript
interface ErrorInfo {
  code: "MODEL_NOT_FOUND" | "VRAM_INSUFFICIENT" | "GENERATION_FAILED" | "VALIDATION_FAILED"
  message: string              // human-readable Spanish/English
  details?: object            // technical details for debugging
}
```

---

## 🔄 Catálogo de Flujos Vigentes

### **1. Linear Flow** ✅ **Production Ready**
```
TaskAnalyzer → ResourceMonitor → ExecutionNode → OutputValidator → HistoryReader → SummaryNode
```
- **Uso:** Inferencia tradicional con contexto y validación
- **Tiempo típico:** 30-60 segundos
- **Casos probados:** Chat, análisis técnico, generación de código

### **2. Challenge Flow** ✅ **Production Ready**
```
Creator → Challenger → Refiner
```
- **Uso:** Deliberación automática entre modelos para mejorar calidad
- **Tiempo típico:** 90 segundos (3 modelos secuenciales)
- **Casos probados:** Validación de contraseñas, funciones Python, análisis de seguridad
- **Persistencia:** JSON en `outputs/challenge_flows/{execution_id}.json`

### **3. Multi-Perspective Flow** 🚫 **No Production Ready**
```
Input → [Security Expert, Performance Expert, UX Expert] → Synthesizer
```
- **Status:** Configurado pero no completamente probado
- **Blocker:** Necesita testing exhaustivo y debugging
- **Roadmap:** Candidato para V4.1

---

## 🚀 Run/Operate

### **Variables de Entorno**
```bash
# Core
APP_ENV=dev                    # dev | prod
LOG_LEVEL=INFO                 # DEBUG | INFO | WARN | ERROR
PORT=8001

# Local Models  
MAX_VRAM_USAGE_GB=6.0         # GPU memory limit
MODEL_CACHE_DIR=.cache/models # HF models cache

# API Keys (future)
OPENAI_API_KEY=               # not implemented yet
ANTHROPIC_API_KEY=            # not implemented yet
```

### **Startup**
```bash
# 1. Environment
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # Linux/Mac

# 2. Dependencies
pip install -r requirements.txt

# 3. Start API server
python api/server.py

# 4. Health check
curl http://localhost:8001/health
```

### **Verificar Salud del Sistema**
```bash
# API status
GET http://localhost:8001/health

# GPU status  
GET http://localhost:8001/models

# Test linear flow
POST http://localhost:8001/api/inference
{
  "prompt": "Hello world",
  "model": "mistral7b"
}

# Test challenge flow
POST http://localhost:8001/api/orchestrate  
{
  "prompt": "Create a password validator",
  "flow_type": "challenge"
}
```

---

## 🗑️ Deprecations

### **Removido en V4:**
- `scripts/direct_inference.py` → **Nunca existió**, referencias eliminadas
- `providers/remote/` → **No implementado**, movido a roadmap futuro
- `examples/` directory → **Archivado** en `/archive/examples/`
- `constants/models.py` hardcoded → **Migrado** a `providers/registries/model_registry.py`

### **Mantenido (Legacy pero funcional):**
- `constants/strategies.py` → Debería ser `providers/registries/strategy_registry.py` pero funciona
- `local_models/` directory → Debería estar en `providers/local/manager/` pero funciona

### **Frontend Disconnected:**
- Interfaz web existe pero no conectada a nuevos endpoints V3
- Requiere trabajo de reconexión (no roto, solo desconectado)

---

## 🗓️ Roadmap Corto (4-6 semanas)

### **Sprint 1 (Semana 1-2): Stabilización**
1. **Reconectar Frontend** - Actualizar UI para usar endpoints V3 ✅
2. **Multi-Perspective Flow Testing** - Debugging y casos de prueba 🔧
3. **Error Handling Unificado** - Implementar códigos de error estándar 📋

### **Sprint 2 (Semana 3-4): Extensibilidad**  
4. **Strategy Registry Migration** - `constants/strategies.py` → `providers/registries/` 🔄
5. **JSON Flow Configs** - Migrar de Python configs a JSON declarativo 📄
6. **Flow Validation Framework** - Testing automatizado de nuevos flujos ✅

### **Sprint 3 (Semana 5-6): Remote Providers**
7. **OpenAI Adapter** - Implementar `providers/remote/openai_provider.py` 🌐
8. **Fallback Logic** - Local → Remote fallback automático 🔄

### **Criterio de Éxito V4:**
- ✅ Frontend reconectado y funcional
- ✅ Multi-Perspective Flow production-ready  
- ✅ Al menos 1 provider remoto implementado
- ✅ Documentación V4 publicada y linkeada en README

---

## 📚 Documentation Structure

```
ai-agent-lab/
├── ARCHITECTURE-V4.md           # Este documento
├── README.md                    # Actualizado con link V4
└── docs/
    ├── NODES-SUMMARY.md         # Catálogo de nodos
    ├── FLOWS-CATALOG.md         # Guía visual de flujos  
    ├── ARCHITECTURE-CORRECTIONS.md # Discrepancias V3
    └── archive/
        └── ARCHITECTURE-V3.md   # Histórico
```

---

**Status:** V4 Draft Ready  
**Next Action:** Update README.md to reference V4  
**Validation:** Code-verified, production-tested features only