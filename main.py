#!/usr/bin/env python3
"""
Main enhanced - Punto de entrada principal del laboratorio de IA
Incluye sistema de health checks, menús interactivos y monitoreo
Version fixed con mejor manejo de errores CUDA
"""

import os
import sys
import json
import time
from datetime import datetime
from typing import Dict, Any, Optional

# Importar módulos locales con manejo de errores
try:
    from utils.gpu_guard import get_gpu_info, check_gpu_health, run_gpu_diagnostics, diagnose_cuda_issues
    from utils.logger import get_session_logger
except ImportError as e:
    print(f"❌ Error importando módulos: {e}")
    print("Asegúrate de que la estructura de directorios sea correcta")
    sys.exit(1)

# Importar launcher con fallback
try:
    from llm_launcher_enhanced import launch_model, MODELS
except ImportError:
    print("⚠️ llm_launcher_enhanced no encontrado, usando versión básica")
    try:
        from llm_launcher import launch_model, MODELS
    except ImportError:
        print("❌ No se encontró ningún launcher. Verifica los archivos.")
        sys.exit(1)

class LabEnvironment:
    """
    Gestiona el entorno completo del laboratorio de IA
    """
    
    def __init__(self):
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Crear directorio de logs si no existe
        os.makedirs("logs/main", exist_ok=True)
        
        try:
            self.logger = get_session_logger("lab_main", "logs/main", self.session_id)
        except Exception as e:
            print(f"⚠️ Error creando logger: {e}")
            self.logger = None
            
        self.health_status = None
        
    def _log_safe(self, level: str, event: str, **kwargs):
        """Log seguro que no falla si logger no está disponible"""
        if self.logger:
            try:
                getattr(self.logger, level)(event, **kwargs)
            except Exception as e:
                print(f"⚠️ Error en logging: {e}")
        
    def startup_checks(self) -> bool:
        """
        Ejecuta checks de startup críticos
        """
        print("🚀 INICIANDO LABORATORIO DE IA")
        print("=" * 50)
        
        self._log_safe("info", "lab_startup", session_id=self.session_id)
        
        # Check 1: Validar estructura de directorios
        print("📁 Verificando estructura de directorios...")
        dirs_ok = self._ensure_directory_structure()
        if not dirs_ok:
            print("❌ Error en estructura de directorios")
            return False
        print("✅ Directorios OK")
        
        # Check 2: Health check de GPU
        print("🖥️ Verificando estado de GPU...")
        self.health_status = check_gpu_health()
        
        if self.health_status["overall_status"] == "fail":
            print("❌ GPU health check falló")
            self._print_health_details()
            
            print("\n🔧 Ejecutando diagnóstico detallado...")
            try:
                diagnose_cuda_issues()
            except Exception as e:
                print(f"⚠️ Error en diagnóstico: {e}")
            
            print("\n" + "="*50)
            print("💡 OPCIONES:")
            print("1. Continuar con CPU (rendimiento reducido)")
            print("2. Instalar/reparar CUDA")
            print("3. Salir y revisar configuración")
            
            choice = input("\nSelecciona opción [1/2/3]: ").strip()
            
            if choice == "1":
                print("⚠️ Continuando con CPU (rendimiento reducido)")
                return True
            elif choice == "2":
                print("\n📝 INSTRUCCIONES PARA INSTALAR CUDA:")
                print("1. Instalar drivers NVIDIA más recientes")
                print("2. Instalar CUDA Toolkit 11.8 o 12.1")
                print("3. Reinstalar PyTorch con CUDA:")
                print("   pip uninstall torch torchvision torchaudio")
                print("   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
                print("\n4. Reiniciar y volver a ejecutar")
                return False
            else:
                return False
        
        elif self.health_status["overall_status"] == "warning":
            print("⚠️ GPU health check con advertencias")
            self._print_health_details()
            
            use_anyway = input("\n¿Continuar de todas formas? [y/N]: ").strip().lower()
            if use_anyway != 'y':
                return False
        
        else:
            print("✅ GPU OK")
            gpu_info = get_gpu_info()
            if gpu_info["cuda"]:
                print(f"   {gpu_info['device']} - {gpu_info['free_gb']:.1f}GB libres")
        
        # Check 3: Verificar dependencias críticas
        print("📦 Verificando dependencias...")
        deps_ok = self._check_dependencies()
        if not deps_ok:
            print("❌ Dependencias faltantes")
            self._show_dependency_install_guide()
            return False
        print("✅ Dependencias OK")
        
        self._log_safe("info", "startup_complete", 
                      health_status=self.health_status["overall_status"],
                      gpu_available=self.health_status.get("checks", {}).get("cuda_available", {}).get("status") == "pass")
        
        return True
    
    def _ensure_directory_structure(self) -> bool:
        """Crea estructura de directorios necesaria"""
        required_dirs = [
            "outputs",
            "logs", 
            "logs/main",
            "metrics",
            "utils"
        ]
        
        # Crear directorios por modelo solo si MODELS está definido
        if 'MODELS' in globals():
            for model_key in MODELS.keys():
                required_dirs.extend([
                    f"outputs/{model_key}",
                    f"outputs/{model_key}/runs",
                    f"logs/{model_key}",
                    f"metrics/{model_key}"
                ])
        
        try:
            for dir_path in required_dirs:
                os.makedirs(dir_path, exist_ok=True)
            return True
        except Exception as e:
            self._log_safe("error", "directory_creation_failed", error=str(e))
            print(f"Error creando directorios: {e}")
            return False
    
    def _check_dependencies(self) -> bool:
        """Verifica dependencias críticas"""
        dependencies_ok = True
        
        # Check PyTorch
        try:
            import torch
            print(f"   ✅ PyTorch {torch.__version__}")
        except ImportError:
            print("   ❌ PyTorch no instalado")
            dependencies_ok = False
        
        # Check Transformers
        try:
            import transformers
            print(f"   ✅ Transformers {transformers.__version__}")
        except ImportError:
            print("   ❌ Transformers no instalado")
            dependencies_ok = False
        
        # Check psutil
        try:
            import psutil
            print(f"   ✅ psutil {psutil.__version__}")
        except ImportError:
            print("   ❌ psutil no instalado")
            dependencies_ok = False
        
        return dependencies_ok
    
    def _show_dependency_install_guide(self):
        """Muestra guía de instalación de dependencias"""
        print("\n📝 GUÍA DE INSTALACIÓN DE DEPENDENCIAS:")
        print("=" * 40)
        print("pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
        print("pip install transformers")
        print("pip install psutil")
        print("pip install python-dotenv")
        print("\nO usar requirements.txt si está disponible:")
        print("pip install -r requirements.txt")
    
    def _print_health_details(self):
        """Imprime detalles del health check"""
        if not self.health_status:
            return
            
        print("\n📊 Detalles del health check:")
        for check_name, result in self.health_status.get("checks", {}).items():
            status_emoji = {"pass": "✅", "warning": "⚠️", "fail": "❌"}
            emoji = status_emoji.get(result["status"], "❓")
            print(f"   {emoji} {check_name}: {result['status']}")
            
            # Mostrar detalles adicionales para algunos checks
            if "recommendation" in result:
                print(f"      → {result['recommendation']}")
            elif "error" in result:
                print(f"      → Error: {result['error']}")
    
    def show_main_menu(self):
        """Muestra el menú principal interactivo"""
        while True:
            print("\n" + "=" * 60)
            print("🧪 LABORATORIO DE IA - MENÚ PRINCIPAL")
            print("=" * 60)
            
            # Mostrar estado actual
            try:
                gpu_info = get_gpu_info()
                if gpu_info["cuda"]:
                    status_icon = "🟢" if gpu_info["memory_status"] in ["excellent", "good"] else "🟡" if gpu_info["memory_status"] == "warning" else "🔴"
                    print(f"{status_icon} GPU: {gpu_info['device']} ({gpu_info['free_gb']:.1f}GB libre)")
                else:
                    print("🔴 GPU: No disponible")
                    if "reason" in gpu_info:
                        print(f"   Razón: {gpu_info['reason']}")
            except Exception as e:
                print(f"⚠️ Error obteniendo info GPU: {e}")
            
            print(f"🕐 Sesión: {self.session_id}")
            
            print("\n📋 OPCIONES:")
            print("1. 🤖 Ejecutar modelo individual")
            print("2. 🔄 Ejecutar comparación entre modelos")
            print("3. 📊 Ver estadísticas de modelos")
            print("4. 🔧 Diagnósticos completos")
            print("5. 🧹 Limpiar memoria GPU")
            print("6. ❓ Ayuda y troubleshooting")
            print("7. ❌ Salir")
            
            try:
                choice = input("\n👉 Selecciona una opción [1-7]: ").strip()
                
                if choice == "1":
                    self._run_single_model()
                elif choice == "2":
                    self._run_model_comparison()
                elif choice == "3":
                    self._show_model_stats()
                elif choice == "4":
                    self._run_diagnostics()
                elif choice == "5":
                    self._clean_gpu_memory()
                elif choice == "6":
                    self._show_help()
                elif choice == "7":
                    print("\n👋 ¡Hasta luego!")
                    self._log_safe("info", "lab_shutdown", session_id=self.session_id)
                    break
                else:
                    print("❌ Opción inválida")
                    
            except KeyboardInterrupt:
                print("\n\n🛑 Interrumpido por usuario")
                break
            except Exception as e:
                print(f"❌ Error: {e}")
                self._log_safe("error", "menu_error", error=str(e))
    
    def _run_single_model(self):
        """Ejecuta un modelo individual"""
        if 'MODELS' not in globals():
            print("❌ Modelos no disponibles - verificar launcher")
            return
            
        print("\n🤖 EJECUCIÓN DE MODELO INDIVIDUAL")
        print("-" * 40)
        
        # Mostrar modelos disponibles
        print("Modelos disponibles:")
        for i, (key, name) in enumerate(MODELS.items(), 1):
            print(f"  {i}. {key} ({name.split('/')[-1]})")
        
        # Selección de modelo
        try:
            choice = input("\nSelecciona modelo: ").strip()
            if choice.isdigit():
                model_keys = list(MODELS.keys())
                if 1 <= int(choice) <= len(model_keys):
                    model_key = model_keys[int(choice) - 1]
                else:
                    print("❌ Número inválido")
                    return
            elif choice in MODELS:
                model_key = choice
            else:
                print("❌ Modelo no encontrado")
                return
            
            # Parámetros de ejecución
            prompt = input("Prompt: ").strip()
            if not prompt:
                print("❌ Prompt vacío")
                return
            
            try:
                max_tokens = int(input("Max tokens [128]: ").strip() or "128")
                temperature = float(input("Temperature [0.7]: ").strip() or "0.7")
                timeout = int(input("Timeout segundos [300]: ").strip() or "300")
            except ValueError:
                max_tokens, temperature, timeout = 128, 0.7, 300
                print("⚠️ Usando valores por defecto")
            
            # Ejecutar
            print(f"\n🔄 Ejecutando {model_key}...")
            start_time = time.time()
            
            try:
                result = launch_model(model_key, prompt, max_tokens, temperature, timeout)
                elapsed = time.time() - start_time
                
                if result:
                    print(f"\n✅ Ejecución completada en {elapsed:.2f}s")
                    print("\n📄 RESULTADO:")
                    print("-" * 30)
                    print(result[:500] + ("..." if len(result) > 500 else ""))
                else:
                    print(f"\n❌ Ejecución falló después de {elapsed:.2f}s")
                    
            except Exception as e:
                elapsed = time.time() - start_time
                print(f"\n💥 Error durante ejecución: {e}")
                print(f"   Tiempo transcurrido: {elapsed:.2f}s")
                
        except Exception as e:
            print(f"❌ Error: {e}")
    
    def _run_model_comparison(self):
        """Ejecuta comparación entre modelos"""
        print("\n🔄 COMPARACIÓN ENTRE MODELOS")
        print("-" * 40)
        
        try:
            # Intentar importar comparador
            from quick_model_comparator import interactive_comparison
            interactive_comparison()
        except ImportError:
            print("⚠️ Comparador no disponible")
            print("   Implementación básica próximamente...")
            
            if 'MODELS' in globals() and len(MODELS) >= 2:
                print("\n💡 Tip: Puedes ejecutar modelos individualmente y comparar outputs manualmente")
                print("   Los resultados se guardan en outputs/[modelo]/runs/")
    
    def _show_model_stats(self):
        """Muestra estadísticas de modelos"""
        print("\n📊 ESTADÍSTICAS DE MODELOS")
        print("-" * 40)
        
        if 'MODELS' not in globals():
            print("❌ Modelos no disponibles")
            return
        
        try:
            from utils.logger import LogAnalyzer
            
            found_stats = False
            for model_key in MODELS.keys():
                print(f"\n🤖 {model_key}:")
                try:
                    stats = LogAnalyzer.analyze_model_performance("logs", model_key)
                    
                    if "error" in stats:
                        print(f"   📭 {stats['error']}")
                        continue
                    
                    found_stats = True
                    print(f"   🏃 Ejecuciones: {stats['successful_runs']} exitosas, {stats['failed_runs']} fallidas")
                    print(f"   📈 Tasa de éxito: {stats['success_rate']}%")
                    
                    if "load_time_stats" in stats:
                        load_stats = stats["load_time_stats"]
                        print(f"   ⏱️ Carga: {load_stats['avg_seconds']}s promedio")
                    
                    if "inference_time_stats" in stats:
                        inf_stats = stats["inference_time_stats"]
                        print(f"   🧠 Inferencia: {inf_stats['avg_seconds']}s promedio")
                        
                except Exception as e:
                    print(f"   ⚠️ Error analizando {model_key}: {e}")
            
            if not found_stats:
                print("\n💡 No hay estadísticas disponibles aún.")
                print("   Ejecuta algunos modelos primero para generar datos.")
                
        except ImportError:
            print("⚠️ Analizador de logs no disponible")
    
    def _run_diagnostics(self):
        """Ejecuta diagnósticos completos"""
        print("\n🔧 DIAGNÓSTICOS COMPLETOS")
        print("-" * 40)
        
        try:
            run_gpu_diagnostics()
            print("\n" + "="*40)
            diagnose_cuda_issues()
        except Exception as e:
            print(f"❌ Error ejecutando diagnósticos: {e}")
    
    def _clean_gpu_memory(self):
        """Limpia memoria GPU manualmente"""
        print("\n🧹 LIMPIEZA DE MEMORIA GPU")
        print("-" * 40)
        
        try:
            from utils.gpu_guard import clear_gpu_memory
            
            before = get_gpu_info()
            if before["cuda"]:
                print(f"Memoria antes: {before['free_gb']:.2f}GB libre")
            
            print("Limpiando memoria...")
            result = clear_gpu_memory()
            
            if result.get("cuda"):
                if result["improvement"]:
                    print(f"✅ Liberados {result['freed_gb']:.2f}GB")
                else:
                    print("ℹ️ No se encontró memoria significativa para liberar")
            else:
                print("⚠️ CUDA no disponible")
                
        except Exception as e:
            print(f"❌ Error limpiando memoria: {e}")
    
    def _show_help(self):
        """Muestra ayuda y troubleshooting"""
        print("\n❓ AYUDA Y TROUBLESHOOTING")
        print("=" * 40)
        
        print("\n🔧 PROBLEMAS COMUNES:")
        print("1. GPU no detectada:")
        print("   → Verificar drivers NVIDIA")
        print("   → Reinstalar PyTorch con CUDA")
        print("   → Reiniciar sistema")
        
        print("\n2. Memoria insuficiente:")
        print("   → Usar modelos más pequeños")
        print("   → Reducir max_tokens")
        print("   → Limpiar memoria GPU (opción 5)")
        
        print("\n3. Modelos muy lentos:")
        print("   → Verificar que usa GPU, no CPU")
        print("   → Cerrar otras aplicaciones")
        print("   → Aumentar timeout")
        
        print("\n4. Errores de dependencias:")
        print("   → pip install torch transformers psutil")
        print("   → Usar entorno virtual (venv)")
        
        print("\n📁 ARCHIVOS IMPORTANTES:")
        print("   logs/       → Logs detallados")
        print("   outputs/    → Resultados generados")
        print("   metrics/    → Métricas de performance")
        
        print("\n🆘 Si nada funciona:")
        print("   1. Ejecutar diagnósticos completos (opción 4)")
        print("   2. Verificar versiones de dependencias")
        print("   3. Revisar logs de error en logs/main/")
        
        input("\nPresiona Enter para continuar...")

def main():
    """Función principal"""
    try:
        # Inicializar entorno
        lab = LabEnvironment()
        
        # Ejecutar checks de startup
        if not lab.startup_checks():
            print("\n❌ Startup falló - revisar configuración")
            print("\n💡 Consejos:")
            print("- Ejecutar diagnósticos: python utils/gpu_guard.py")
            print("- Instalar dependencias faltantes")
            print("- Verificar drivers GPU")
            sys.exit(1)
        
        print("\n✅ Laboratorio iniciado correctamente")
        
        # Mostrar menú principal
        lab.show_main_menu()
        
    except KeyboardInterrupt:
        print("\n\n🛑 Laboratorio interrumpido por usuario")
    except Exception as e:
        print(f"\n💥 Error crítico: {e}")
        print("\n🔍 Para debug:")
        print("python utils/gpu_guard.py")
        sys.exit(1)

if __name__ == "__main__":
    main()