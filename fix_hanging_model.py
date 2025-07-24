# fix_hanging_model.py
"""
Script para diagnosticar y solucionar el problema del modelo colgado
"""
import torch
import os
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import psutil
import time

def check_model_cache():
    """Verifica el caché de modelos de HuggingFace"""
    cache_dir = os.path.expanduser("~/.cache/huggingface/hub")
    print(f"📁 Directorio de caché: {cache_dir}")
    
    if os.path.exists(cache_dir):
        # Listar modelos en caché
        models = os.listdir(cache_dir)
        print(f"📦 Modelos en caché: {len(models)}")
        for model in models:
            if "mistral" in model.lower():
                path = os.path.join(cache_dir, model)
                size = sum(os.path.getsize(os.path.join(dirpath, filename))
                          for dirpath, dirnames, filenames in os.walk(path)
                          for filename in filenames)
                print(f"   - {model}: {size / (1024**3):.2f} GB")

def test_simple_generation():
    """Prueba generación simple con configuración mínima"""
    print("\n🧪 PRUEBA SIMPLE DE GENERACIÓN")
    print("="*40)
    
    model_name = "mistralai/Mistral-7B-Instruct-v0.2"
    
    try:
        # 1. Intentar con device_map específico (no 'auto')
        print("1️⃣ Cargando con device_map='cuda:0'...")
        
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Cargar SOLO en GPU, no device_map auto
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="cuda:0",  # Específico, no 'auto'
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            offload_folder="offload",  # Carpeta para offload si es necesario
        )
        
        print("✅ Modelo cargado")
        
        # Generar directamente sin pipeline
        prompt = "Write a Python function to calculate prime numbers"
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
        
        print("2️⃣ Generando...")
        start = time.time()
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=50,  # Pocos tokens para prueba rápida
                do_sample=False,    # Greedy para ser determinístico
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
        
        elapsed = time.time() - start
        output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        print(f"✅ Generación completada en {elapsed:.2f}s")
        print(f"📝 Output: {output_text[:200]}...")
        
        # Limpiar
        del model
        torch.cuda.empty_cache()
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def monitor_resources():
    """Monitorea recursos del sistema"""
    print("\n📊 RECURSOS DEL SISTEMA")
    print("="*40)
    
    # CPU
    print(f"CPU: {psutil.cpu_percent()}% uso")
    
    # RAM
    ram = psutil.virtual_memory()
    print(f"RAM: {ram.percent}% uso ({ram.available / (1024**3):.1f}GB disponible)")
    
    # GPU
    if torch.cuda.is_available():
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        allocated = torch.cuda.memory_allocated() / (1024**3)
        print(f"GPU: {allocated:.1f}GB / {gpu_mem:.1f}GB usado")
    
    # Disco
    disk = psutil.disk_usage('/')
    print(f"Disco: {disk.percent}% uso ({disk.free / (1024**3):.1f}GB libre)")

def kill_hanging_process():
    """Mata procesos de Python que puedan estar colgados"""
    print("\n🔍 Buscando procesos Python colgados...")
    
    current_pid = os.getpid()
    killed = 0
    
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            if proc.info['name'] == 'python.exe' or proc.info['name'] == 'python':
                if proc.info['pid'] != current_pid:
                    cmdline = ' '.join(proc.info['cmdline'] or [])
                    if 'main_simple_fixed.py' in cmdline or 'llm_launcher' in cmdline:
                        print(f"⚠️ Encontrado proceso sospechoso: PID {proc.info['pid']}")
                        proc.kill()
                        killed += 1
                        print(f"✅ Proceso {proc.info['pid']} terminado")
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    
    if killed > 0:
        print(f"✅ {killed} procesos terminados")
    else:
        print("✅ No se encontraron procesos colgados")

def main():
    """Diagnóstico y solución principal"""
    print("🔧 DIAGNÓSTICO Y SOLUCIÓN PARA MODELO COLGADO")
    print("="*50)
    
    # 1. Verificar recursos
    monitor_resources()
    
    # 2. Verificar caché
    check_model_cache()
    
    # 3. Buscar procesos colgados
    kill_hanging_process()
    
    # 4. Hacer prueba simple
    print("\n¿Ejecutar prueba simple de generación? [y/N]: ", end="")
    if input().strip().lower() == 'y':
        test_simple_generation()
    
    print("\n✅ Diagnóstico completado")
    print("\n💡 RECOMENDACIONES:")
    print("1. Si el modelo se cuelga con device_map='auto', usa device_map='cuda:0'")
    print("2. Reduce max_tokens a 50-100 para pruebas")
    print("3. Considera usar la estrategia 'optimized' con quantización 4-bit")
    print("4. Si el disco está lleno, limpia el caché de HuggingFace")

if __name__ == "__main__":
    main()