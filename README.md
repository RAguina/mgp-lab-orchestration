# AI-Agent-Lab 🧪🤖

> **Arquitectura:** [ARCHITECTURE-V4.md](ARCHITECTURE-V4.md) ← **DOCUMENTACIÓN PRINCIPAL**

1)
.\venv\Activate\

2)Entry-Point:
python -m langchain_integration.langgraph.routing_agent
python -m local_models.llm_launcher

Sistema de orquestación de LLMs locales con flujos configurables de deliberación entre modelos.

## ✨ Features

- ✅ **Challenge Flow** - LLM Deliberation (Creator→Challenger→Refiner)
- ✅ **Linear Flow** - Traditional pipeline with validation
- ✅ **Local Models** - HuggingFace integration (Mistral, Llama3, DeepSeek)
- ✅ **REST API** - FastAPI endpoints for integration
- ✅ **Smart Caching** - Intelligent model loading (51s → 19s)

Laboratorio personal para experimentar con agentes LLM usando **LangChain**,
**LangGraph** y **FastAPI** y modelos locales (Llama 3, Mistral 7B, DeepSeek 7B, DeepSeek-Coder,
etc.).

## Requisitos

| Componente | Versión recomendada |
|------------|--------------------|
| Python     | 3.10 – 3.12 (64 bit) |
| GPU        | NVIDIA con ≥ 8 GB VRAM (opcional pero recomendado) |
| CUDA       | 12.1 (la rueda Torch usada en `requirements.txt`) |

## Instalación rápida

```bash
git clone https://github.com/tu-usuario/ai-agent-lab.git
cd ai-agent-lab

# 1) entorno virtual
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/macOS
source .venv/bin/activate

# 2) variables de entorno
copy .env.example .env      # o crea .env y añade OPENAI_API_KEY=<tu-clave>

# 3) dependencias
pip install --upgrade pip
pip install -r requirements.txt



**Para solucionar problemas**
chcp 65001
set PYTHONIOENCODING=utf-8


python api/server.py --reload-exclude="logs/*" --reload-exclude="*.log" --reload-exclude="outputs/*" --reload-exclude="metrics/*"


Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope 
  Process