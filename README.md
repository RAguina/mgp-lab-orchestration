# AI-Agent-Lab 🧪🤖

Entry-Point:
python -m local_models.llm_launcher


Laboratorio personal para experimentar con agentes LLM usando **LangChain**,
**AutoGen** y modelos locales (Llama 3, Mistral 7B, DeepSeek 7B, DeepSeek-Coder,
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
