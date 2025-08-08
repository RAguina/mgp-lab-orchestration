#!/usr/bin/env python3
"""
Main simple fixed - Versión completa mejorada
Laboratorio de IA con launcher automático y gestión robusta de errores
"""

import os
import sys
import time
import gc
from datetime import datetime
from typing import Dict, Any

# -----------------------------------------------------------------------------
# Configuración y estructura básica
# -----------------------------------------------------------------------------

def ensure_basic_structure() -> None:
    """Crea directorios básicos si no existen"""
    dirs = ["outputs", "logs", "metrics", "utils", "local_models"]
    for d in dirs:
        os.makedirs(d, exist_ok=True)
        # Crear __init__.py si no existe
        if d in ["utils", "local_models"]:
            init_file = os.path.join(d, "__init__.py")
            if not os.path.exists(init_file):
                with open(init_file, "w") as f:
                    pass

# -----------------------------------------------------------------------------
# Detección y gestión del launcher
# -----------------------------------------------------------------------------

class LabManager:
    """Clase principal que encapsula el estado del laboratorio"""
    
    def __init__(self):
        self.launcher_info = self._detect_launcher()
        self.session_start = datetime.now()
        
    def _detect_launcher(self) -> Dict[str, Any]:
        """Detecta automáticamente el launcher disponible"""
        # Asegurar que local_models está en el path
        if 'local_models' not in sys.path and os.path.exists('local_models'):
            sys.path.insert(0, os.path.abspath('.'))
        
        info = {
            "available": False,
            "type": "none",
            "models": {},
            "launch_model": None,
            "launch_model_optimized": None,
            "launch_model_with_streaming": None,
            "ModelLauncher": None
        }
        
        try:
            import importlib
            launcher_module = importlib.import_module('local_models.llm_launcher')
            
            info["launch_model"] = getattr(launcher_module, "launch_model", None)
            info["launch_model_optimized"] = getattr(launcher_module, "launch_model_optimized", None)
            info["launch_model_with_streaming"] = getattr(launcher_module, "launch_model_with_streaming", None)
            info["ModelLauncher"] = getattr(launcher_module, "ModelLauncher", None)
            info["models"] = getattr(launcher_module, "MODELS", {})
            
            if info["launch_model"] and info["models"]:
                info["available"] = True
                info["type"] = "modular" if info["launch_model_optimized"] else "simple"
                
                print(f"✅ Launcher {info['type']} encontrado")
                print(f"✅ {len(info['models'])} modelos disponibles")
                if info["type"] == "modular":
                    print(f"✅ Estrategias: Standard, Optimized, Streaming")
            
        except ImportError as e:
            print(f"❌ No se pudo importar launcher: {e}")
        
        return info
    
    def get_available_strategies(self) -> list:
        """Retorna lista de estrategias disponibles"""
        if not self.launcher_info["available"]:
            return []
        
        strategies = ["standard"]
        
        if self.launcher_info["type"] == "modular":
            # Verificar bitsandbytes para estrategia optimized
            if _has_bitsandbytes():
                strategies.append("optimized")
            
            # Streaming siempre disponible en modular
            if self.launcher_info["launch_model_with_streaming"]:
                strategies.append("streaming")
        
        return strategies

# -----------------------------------------------------------------------------
# Gestión de GPU (con caché para optimizar rendimiento)
# -----------------------------------------------------------------------------

class GPUManager:
    """Gestor de GPU con caché para optimizar llamadas"""
    
    def __init__(self):
        self._gpu_cache = None
        self._cache_time = 0
        self._cache_duration = 5  # Cache válido por 5 segundos
    
    def get_gpu_info(self) -> Dict[str, Any]:
        """Obtiene información GPU con caché"""
        current_time = time.time()
        
        # Usar caché si es reciente
        if (self._gpu_cache and 
            current_time - self._cache_time < self._cache_duration):
            return self._gpu_cache
        
        # Actualizar caché
        try:
            from utils.gpu_guard import get_gpu_info as gpu_info_detailed
            self._gpu_cache = gpu_info_detailed()
        except ImportError:
            # Fallback básico
            self._gpu_cache = self._get_basic_gpu_info()
        
        self._cache_time = current_time
        return self._gpu_cache
    
    def _get_basic_gpu_info(self) -> Dict[str, Any]:
        """Fallback básico para obtener info GPU"""
        try:
            import torch
            if torch.cuda.is_available():
                device = torch.cuda.current_device()
                props = torch.cuda.get_device_properties(device)
                free_b, total_b = torch.cuda.mem_get_info(device)
                free_gb = free_b / (1024**3)
                total_gb = total_b / (1024**3)
                
                return {
                    "cuda": True,
                    "device": props.name,
                    "free_gb": round(free_gb, 1),
                    "total_gb": round(total_gb, 1),
                    "compute_capability": f"{props.major}.{props.minor}"
                }
            else:
                return {"cuda": False, "reason": "CUDA not available"}
        except Exception as e:
            return {"cuda": False, "reason": str(e)}
    
    def clear_gpu_memory(self) -> Dict[str, Any]:
        """Limpieza GPU invalidando caché"""
        try:
            from utils.gpu_guard import clear_gpu_memory as clear_gpu_detailed
            result = clear_gpu_detailed()
        except ImportError:
            # Fallback básico
            result = self._basic_gpu_cleanup()
        
        # Invalidar caché después de limpieza
        self._gpu_cache = None
        return result
    
    def _basic_gpu_cleanup(self) -> Dict[str, Any]:
        """Fallback básico para limpieza GPU"""
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                gc.collect()
                return {"action": "basic_cleanup", "cuda": True}
        except:
            pass
        return {"action": "no_cleanup", "cuda": False}

# -----------------------------------------------------------------------------
# Funciones del menú principal
# -----------------------------------------------------------------------------

def show_menu(lab_manager: LabManager, gpu_manager: GPUManager) -> None:
    """Menú principal mejorado con manejo robusto de errores"""
    while True:
        try:
            print("\n" + "="*50)
            print("🧪 LABORATORIO DE IA - MENÚ PRINCIPAL")
            print("="*50)
            
            # Estado GPU (con caché)
            gpu_info = gpu_manager.get_gpu_info()
            if gpu_info.get("cuda"):
                device_name = gpu_info.get("device", "GPU")
                free_gb = gpu_info.get("free_gb", 0)
                total_gb = gpu_info.get("total_gb", 0)
                compute = gpu_info.get("compute_capability", "N/A")
                print(f"🟢 GPU: {device_name} ({free_gb:.1f}/{total_gb:.1f}GB)")
                print(f"   Compute: {compute}")
            else:
                print(f"🔴 GPU: {gpu_info.get('reason', 'No disponible')}")
            
            # Estado del launcher
            if lab_manager.launcher_info["available"]:
                print(f"✅ Launcher: {lab_manager.launcher_info['type']} ({len(lab_manager.launcher_info['models'])} modelos)")
                strategies = lab_manager.get_available_strategies()
                if len(strategies) > 1:
                    print(f"   Estrategias: {', '.join(strategies).title()}")
            else:
                print("❌ Launcher: No disponible")
            
            print(f"🕐 Sesión: {datetime.now().strftime('%H:%M:%S')} (activa {_format_duration(lab_manager.session_start)})")
            
            print("\n📋 OPCIONES:")
            print("1. 🤖 Ejecutar modelo")
            print("2. 📊 Listar modelos disponibles")
            print("3. 🧹 Limpiar memoria GPU")
            print("4. 🔧 Info del sistema")
            print("5. 📈 Analizar logs")
            print("6. ❌ Salir")
            
            choice = input("\n👉 Opción [1-6]: ").strip()
            
            if not choice:  # Enter vacío
                continue
                
            if choice == "1":
                run_model_with_strategy(lab_manager, gpu_manager)
            elif choice == "2":
                list_models(lab_manager.launcher_info["models"])
            elif choice == "3":
                clean_memory(gpu_manager)
            elif choice == "4":
                show_system_info(gpu_manager)
            elif choice == "5":
                analyze_logs()
            elif choice == "6":
                print("👋 ¡Hasta luego!")
                break
            else:
                print("❌ Opción inválida")
                
        except KeyboardInterrupt:
            print("\n\n🛑 Interrumpido")
            break
        except Exception as e:
            print(f"❌ Error en menú: {e}")
            import traceback
            traceback.print_exc()
        finally:
            # Invalidar caché GPU para refrescar estado en próxima iteración
            gpu_manager._gpu_cache = None

def run_model_with_strategy(lab_manager: LabManager, gpu_manager: GPUManager) -> None:
    """Ejecuta modelo con selección de estrategia mejorada"""
    if not lab_manager.launcher_info["available"]:
        print("❌ No hay launcher disponible")
        return
    
    try:
        models = lab_manager.launcher_info["models"]
        
        print("\n🤖 EJECUTAR MODELO")
        print("-" * 30)
        
        # Seleccionar modelo
        model_key = _select_model(models)
        if not model_key:
            return
        
        # Seleccionar estrategia
        available_strategies = lab_manager.get_available_strategies()
        strategy = _select_strategy(available_strategies)
        
        # Validar combinación modelo-estrategia
        if not _validate_model_strategy(model_key, strategy, gpu_manager):
            return
        
        # Obtener parámetros
        prompt = input("\nPrompt: ").strip()
        if not prompt:
            print("❌ Prompt vacío")
            return
        
        try:
            max_tokens = int(input("Max tokens [128]: ").strip() or "128")
            if max_tokens <= 0:
                print("❌ Max tokens debe ser positivo")
                return
        except ValueError:
            max_tokens = 128
        
        # Mostrar configuración y confirmar
        print(f"\n📋 Configuración:")
        print(f"   Modelo: {model_key}")
        print(f"   Estrategia: {strategy}")
        print(f"   Max tokens: {max_tokens}")
        print(f"   Device: cuda:0")
        
        confirm = input("\n¿Continuar? [S/n]: ").strip().lower()
        if confirm == 'n':
            print("❌ Cancelado")
            return
        
        # Ejecutar modelo
        _execute_model(lab_manager, model_key, strategy, prompt, max_tokens, gpu_manager)
        
    except KeyboardInterrupt:
        print("\n🛑 Ejecución interrumpida por usuario")
    except Exception as e:
        print(f"❌ Error durante ejecución: {e}")
        import traceback
        traceback.print_exc()

def _select_model(models: Dict[str, str]) -> str:
    """Selecciona modelo de la lista disponible"""
    print("\nModelos disponibles:")
    model_list = list(models.keys())
    for i, key in enumerate(model_list, 1):
        print(f"  {i}. {key}")
    
    choice = input("\nModelo [número o nombre]: ").strip()
    if choice.isdigit():
        idx = int(choice) - 1
        if 0 <= idx < len(model_list):
            return model_list[idx]
        else:
            print("❌ Número inválido")
            return None
    elif choice in models:
        return choice
    else:
        print("❌ Modelo no encontrado")
        return None

def _select_strategy(available_strategies: list) -> str:
    """Selecciona estrategia basada en disponibilidad"""
    if len(available_strategies) == 1:
        return available_strategies[0]
    
    print("\n📦 Estrategias disponibles:")
    strategy_descriptions = {
        "standard": "Standard (FP16, ~8GB VRAM)",
        "optimized": "Optimized (4-bit, ~4GB VRAM) 🎯",
        "streaming": "Streaming (ver output en tiempo real)"
    }
    
    for i, strategy in enumerate(available_strategies, 1):
        desc = strategy_descriptions.get(strategy, strategy.title())
        print(f"  {i}. {desc}")
    
    # Estrategia por defecto (optimized si está disponible, sino standard)
    default_choice = "2" if "optimized" in available_strategies else "1"
    
    strat_choice = input(f"\nEstrategia [1-{len(available_strategies)}, default={default_choice}]: ").strip() or default_choice
    
    try:
        idx = int(strat_choice) - 1
        if 0 <= idx < len(available_strategies):
            return available_strategies[idx]
    except ValueError:
        pass
    
    # Fallback a estrategia por defecto
    return available_strategies[0]

def _validate_model_strategy(model_key: str, strategy: str, gpu_manager: GPUManager) -> bool:
    """Valida que la combinación modelo-estrategia sea viable"""
    gpu_info = gpu_manager.get_gpu_info()
    
    if not gpu_info.get("cuda"):
        print("⚠️ GPU no disponible - el modelo será muy lento en CPU")
        response = input("¿Continuar de todos modos? [s/N]: ").strip().lower()
        return response == 's'
    
    # Verificar memoria disponible
    free_gb = gpu_info.get("free_gb", 0)
    
    # Estimaciones aproximadas de memoria requerida
    memory_requirements = {
        ("standard", "7b"): 8.0,
        ("standard", "3b"): 4.0,
        ("optimized", "7b"): 4.0,
        ("optimized", "3b"): 2.0,
        ("streaming", "7b"): 8.0,
        ("streaming", "3b"): 4.0,
    }
    
    # Detectar tamaño del modelo basado en el nombre
    model_size = "7b" if "7b" in model_key.lower() else "3b"
    required_gb = memory_requirements.get((strategy, model_size), 6.0)
    
    if free_gb < required_gb:
        print(f"⚠️ Memoria insuficiente: {free_gb:.1f}GB disponible, ~{required_gb:.1f}GB requerido")
        print("💡 Sugerencias:")
        if strategy == "standard" and _has_bitsandbytes():
            print("   - Usar estrategia 'optimized' para reducir uso de memoria")
        print("   - Ejecutar 'Limpiar memoria GPU' antes de continuar")
        
        response = input("¿Continuar de todos modos? [s/N]: ").strip().lower()
        return response == 's'
    
    return True

def _execute_model(lab_manager: LabManager, model_key: str, strategy: str, 
                  prompt: str, max_tokens: int, gpu_manager: GPUManager) -> None:
    """Ejecuta el modelo con timeout y manejo de errores"""
    print(f"\n🔄 Ejecutando...")
    start_time = time.time()
    
    # Mostrar progreso
    print("⏳ Cargando modelo...", end="", flush=True)
    
    try:
        # Seleccionar función apropiada según estrategia
        launcher_funcs = {
            "standard": lab_manager.launcher_info["launch_model"],
            "optimized": lab_manager.launcher_info["launch_model_optimized"],
            "streaming": lab_manager.launcher_info["launch_model_with_streaming"]
        }
        
        launch_func = launcher_funcs.get(strategy)
        if not launch_func:
            print(f"\n❌ Estrategia '{strategy}' no disponible")
            return
        
        # Parámetros según estrategia
        kwargs = {"device_map": "cuda:0"}
        
        if strategy == "optimized":
            kwargs["use_quantization"] = True
            # No agregar max_memory aquí - el launcher optimized lo maneja internamente
        
        # Ejecutar con manejo de OOM
        try:
            result = launch_func(model_key, prompt, max_tokens, **kwargs)
        except Exception as e:
            # Intentar detectar OutOfMemoryError
            if "out of memory" in str(e).lower() or "cuda error" in str(e).lower():
                print(f"\n❌ Error de memoria GPU: {e}")
                print("💡 Sugerencias:")
                print("   - Usar estrategia 'optimized' si no la estás usando")
                print("   - Reducir max_tokens")
                print("   - Limpiar memoria GPU y reintentar")
                return None
            else:
                raise  # Re-lanzar otros errores
        
        elapsed = time.time() - start_time
        
        if result:
            print(f"\n✅ Completado en {elapsed:.2f}s")
            
            # Mostrar resultado (con límite para outputs largos)
            if strategy != "streaming":  # Streaming ya muestra el output
                print("\n📄 RESULTADO:")
                print("-" * 30)
                display_text = result[:800] + ("..." if len(result) > 800 else "")
                print(display_text)
            
            # Mostrar info de memoria después
            gpu_after = gpu_manager.get_gpu_info()
            if gpu_after.get("cuda"):
                print(f"\n💾 VRAM después: {gpu_after['free_gb']:.1f}GB libre")
                
            # Sugerir limpieza si hay mucho uso de memoria
            if gpu_after.get("free_gb", 100) < 2.0:
                print("💡 Considera ejecutar 'Limpiar memoria GPU' para liberar recursos")
        else:
            print(f"\n❌ Falló después de {elapsed:.2f}s")
            
    except KeyboardInterrupt:
        print("\n\n🛑 Generación interrumpida por usuario")
    except Exception as e:
        elapsed = time.time() - start_time
        print(f"\n❌ Error durante la generación ({elapsed:.2f}s): {e}")
        
        # Log error para debugging
        try:
            import traceback
            error_details = traceback.format_exc()
            print(f"💡 Para debugging: {error_details}")
        except:
            pass

def list_models(models: Dict[str, str]) -> None:
    """Lista modelos disponibles con detalles mejorados"""
    print("\n📋 MODELOS DISPONIBLES")
    print("-" * 40)
    
    if not models:
        print("❌ No hay modelos disponibles")
        return
    
    model_items = list(models.items())
    for i, (key, model_path) in enumerate(model_items, 1):
        model_name = model_path.split('/')[-1]
        print(f"\n{i}. {key}")
        print(f"   📦 {model_name}")
        print(f"   🔗 {model_path}")
        
        # Estimar tamaño en VRAM con más precisión
        if "7b" in model_name.lower():
            print("   💾 ~8GB (FP16) / ~4GB (4-bit)")
        elif "3b" in model_name.lower():
            print("   💾 ~4GB (FP16) / ~2GB (4-bit)")
        elif "1b" in model_name.lower():
            print("   💾 ~2GB (FP16) / ~1GB (4-bit)")
        else:
            print("   💾 Tamaño variable")
        
        # Paginación cada 5 modelos
        if i % 5 == 0 and i < len(model_items):
            try:
                input("\n[Enter para continuar...]")
            except KeyboardInterrupt:
                print("\n🛑 Lista interrumpida")
                break

def clean_memory(gpu_manager: GPUManager) -> None:
    """Limpia memoria GPU con detalles mejorados"""
    print("\n🧹 LIMPIEZA DE MEMORIA")
    print("-" * 30)
    
    before = gpu_manager.get_gpu_info()
    if before.get("cuda"):
        print(f"Antes: {before.get('free_gb', 0):.1f}GB libre")
        allocated = before.get('allocated_gb', 0)
        if allocated > 0:
            print(f"       {allocated:.1f}GB asignado")
    
    print("\nLimpiando...", end="", flush=True)
    result = gpu_manager.clear_gpu_memory()
    print(" ✓")
    
    after = gpu_manager.get_gpu_info()
    if after.get("cuda") and before.get("cuda"):
        freed = after.get('free_gb', 0) - before.get('free_gb', 0)
        print(f"\nDespués: {after.get('free_gb', 0):.1f}GB libre")
        allocated_after = after.get('allocated_gb', 0)
        if allocated_after > 0:
            print(f"         {allocated_after:.1f}GB asignado")
        print(f"\nLiberado: {freed:+.2f}GB")
        
        if freed > 0.1:
            print("✅ Limpieza exitosa")
        elif freed > 0:
            print("✅ Limpieza menor")
        else:
            print("⚠️ Sin cambios significativos")
    else:
        print("⚠️ No se pudo determinar el efecto de la limpieza")

def show_system_info(gpu_manager: GPUManager) -> None:
    """Muestra info detallada del sistema"""
    print("\n🔧 INFORMACIÓN DEL SISTEMA")
    print("-" * 40)
    
    try:
        from utils.gpu_guard import get_system_info
        system = get_system_info()
        
        # GPU
        gpu = system["gpu"]
        print("\n🖥️ GPU:")
        if gpu.get("cuda"):
            print(f"   Dispositivo: {gpu['device']}")
            print(f"   VRAM: {gpu['free_gb']:.1f}GB libre / {gpu['total_gb']:.1f}GB total")
            print(f"   Utilización: {gpu.get('utilization_pct', 0):.1f}%")
            print(f"   Compute: {gpu.get('compute_capability', 'N/A')}")
            print(f"   Estado: {gpu.get('memory_status', 'unknown')}")
        else:
            print(f"   {gpu.get('reason', 'No disponible')}")
        
        # CPU
        cpu = system["cpu"]
        print(f"\n💻 CPU:")
        print(f"   Núcleos: {cpu['cores']}")
        print(f"   Uso: {cpu['usage_percent']}%")
        freq = cpu.get('frequency_mhz', 'unknown')
        if freq != 'unknown':
            print(f"   Frecuencia: {freq:.0f} MHz")
        
        # RAM
        memory = system["memory"]
        print(f"\n🧠 RAM:")
        print(f"   Total: {memory['total_gb']:.1f}GB")
        print(f"   Disponible: {memory['available_gb']:.1f}GB")
        print(f"   Uso: {memory['usage_percent']:.1f}%")
        
        # Python Environment
        env = system.get("python_env", {})
        print(f"\n🐍 Python:")
        print(f"   PyTorch: {env.get('torch_version', 'N/A')}")
        print(f"   CUDA: {env.get('cuda_version', 'N/A')}")
        
    except Exception as e:
        print(f"Error obteniendo info del sistema: {e}")
        # Fallback básico
        gpu_info = gpu_manager.get_gpu_info()
        if gpu_info.get("cuda"):
            print(f"GPU: {gpu_info['device']} ({gpu_info['free_gb']:.1f}GB libre)")
        else:
            print("GPU: No disponible")
    
    # Verificar dependencias importantes
    print("\n📦 Dependencias:")
    dependencies = [
        ("transformers", "Transformers"),
        ("torch", "PyTorch"),
        ("accelerate", "Accelerate"),
        ("bitsandbytes", "BitsAndBytes (4-bit)"),
        ("safetensors", "SafeTensors")
    ]
    
    for module, name in dependencies:
        try:
            __import__(module)
            print(f"   ✅ {name}")
        except ImportError:
            print(f"   ❌ {name} - instalar con: pip install {module}")

def analyze_logs() -> None:
    """Analiza logs de ejecuciones previas"""
    print("\n📈 ANÁLISIS DE LOGS")
    print("-" * 30)
    
    try:
        from utils.logger import LogAnalyzer
    except ImportError:
        print("❌ LogAnalyzer no disponible")
        print("   Asegúrate de que utils/logger.py contiene la clase LogAnalyzer")
        return
    
    try:
        log_dir = "logs"
        if not os.path.exists(log_dir):
            print("❌ No hay logs disponibles")
            return
        
        model_dirs = [d for d in os.listdir(log_dir) 
                     if os.path.isdir(os.path.join(log_dir, d))]
        if not model_dirs:
            print("❌ No hay logs de modelos")
            return
        
        print("\nModelos con logs:")
        for i, model in enumerate(model_dirs, 1):
            print(f"  {i}. {model}")
        
        choice = input("\nAnalizar modelo [número]: ").strip()
        if choice.isdigit():
            idx = int(choice) - 1
            if 0 <= idx < len(model_dirs):
                model_key = model_dirs[idx]
                
                print(f"\nAnalizando logs de {model_key}...")
                analysis = LogAnalyzer.analyze_model_performance(log_dir, model_key)
                
                if "error" in analysis:
                    print(f"❌ {analysis['error']}")
                    return
                
                print(f"\n📊 Resultados:")
                print(f"   Runs exitosos: {analysis.get('successful_runs', 0)}")
                print(f"   Runs fallidos: {analysis.get('failed_runs', 0)}")
                print(f"   Tasa de éxito: {analysis.get('success_rate', 0):.1f}%")
                
                if "load_time_stats" in analysis:
                    stats = analysis["load_time_stats"]
                    print(f"\n⏱️ Tiempos de carga:")
                    print(f"   Promedio: {stats['avg_seconds']:.1f}s")
                    print(f"   Mínimo: {stats['min_seconds']:.1f}s")
                    print(f"   Máximo: {stats['max_seconds']:.1f}s")
                
                if "inference_time_stats" in analysis:
                    stats = analysis["inference_time_stats"]
                    print(f"\n⚡ Tiempos de inferencia:")
                    print(f"   Promedio: {stats['avg_seconds']:.1f}s")
                    print(f"   Mínimo: {stats['min_seconds']:.1f}s")
                    print(f"   Máximo: {stats['max_seconds']:.1f}s")
            else:
                print("❌ Número inválido")
        else:
            print("❌ Entrada inválida")
        
    except Exception as e:
        print(f"❌ Error analizando logs: {e}")

# -----------------------------------------------------------------------------
# Utilidades auxiliares
# -----------------------------------------------------------------------------

def _has_bitsandbytes() -> bool:
    """Verifica si bitsandbytes está disponible"""
    try:
        import bitsandbytes
        return True
    except ImportError:
        return False

def _format_duration(start_time: datetime) -> str:
    """Formatea duración desde start_time hasta ahora"""
    duration = datetime.now() - start_time
    total_seconds = int(duration.total_seconds())
    
    if total_seconds < 60:
        return f"{total_seconds}s"
    elif total_seconds < 3600:
        minutes = total_seconds // 60
        seconds = total_seconds % 60
        return f"{minutes}m {seconds}s"
    else:
        hours = total_seconds // 3600
        minutes = (total_seconds % 3600) // 60
        return f"{hours}h {minutes}m"

# -----------------------------------------------------------------------------
# Función principal
# -----------------------------------------------------------------------------

def main() -> None:
    """Función principal"""
    print("🚀 Iniciando laboratorio de IA...")
    
    # Setup básico
    ensure_basic_structure()
    
    # Inicializar managers
    lab_manager = LabManager()
    gpu_manager = GPUManager()
    
    # Verificar launcher
    if not lab_manager.launcher_info["available"]:
        print("\n❌ No se encontró launcher funcional")
        print("\n💡 Asegúrate de que existe:")
        print("   local_models/llm_launcher.py")
        print("   local_models/loading_strategies.py")
        print("\nO copia tu llm_launcher.py original a local_models/")
        return
    
    # Verificar GPU (warning, no blocking)
    gpu_info = gpu_manager.get_gpu_info()
    if not gpu_info.get("cuda"):
        print("\n⚠️ GPU no disponible - los modelos serán lentos en CPU")
        response = input("¿Continuar de todos modos? [s/N]: ").strip().lower()
        if response != 's':
            return
    
    # Mostrar menú principal
    show_menu(lab_manager, gpu_manager)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n🛑 Interrumpido por usuario")
    except Exception as e:
        print(f"\n💥 Error fatal: {e}")
        import traceback
        traceback.print_exc()