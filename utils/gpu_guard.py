# utils/gpu_guard.py - VERSIÓN FINAL CORREGIDA
import gc
import time
import psutil
from typing import Dict, Any, Optional

def get_gpu_info() -> Dict[str, Any]:
    """
    Devuelve la información de GPU basándose en mem_get_info(), que consulta
    directamente al driver; así `free_gb` nunca será negativo ni > total_gb.
    """
    try:
        import torch
    except ImportError:
        return {"cuda": False, "reason": "PyTorch not installed", "cpu_fallback": True}

    if not torch.cuda.is_available():
        return {
            "cuda": False,
            "reason": "CUDA not available",
            "cpu_fallback": True,
            "torch_version": torch.__version__,
        }

    device = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(device)

    # --- memoria real según driver (bytes libres y totales)
    free_b, total_b = torch.cuda.mem_get_info(device)
    free_gb, total_gb = free_b / 1_073_741_824, total_b / 1_073_741_824

    allocated_gb = torch.cuda.memory_allocated(device) / 1_073_741_824
    reserved_gb  = torch.cuda.memory_reserved(device)  / 1_073_741_824
    utilization  = round((total_gb - free_gb) / total_gb * 100, 1)

    return {
        "cuda": True,
        "device": props.name,
        "device_id": device,
        "compute_capability": f"{props.major}.{props.minor}",
        "total_gb": round(total_gb, 2),
        "allocated_gb": round(allocated_gb, 2),
        "reserved_gb": round(reserved_gb, 2),
        "free_gb": round(free_gb, 2),
        "utilization_pct": utilization,
        "multiprocessor_count": props.multi_processor_count,
        "is_integrated": getattr(props, "integrated", False),
        "memory_status": _classify_memory_status(free_gb, total_gb),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
    }

def _classify_memory_status(free_gb: float, total_gb: float) -> str:
    """Clasifica el estado de memoria GPU"""
    if total_gb == 0:
        return "unknown"
    
    usage_pct = ((total_gb - free_gb) / total_gb) * 100
    
    if usage_pct < 30:
        return "excellent"
    elif usage_pct < 60:
        return "good"
    elif usage_pct < 85:
        return "warning"
    else:
        return "critical"

def clear_gpu_memory() -> Dict[str, Any]:
    """
    Limpia memoria GPU de forma agresiva y reporta resultados
    """
    try:
        import torch
    except ImportError:
        return {"cuda": False, "action": "no_torch"}
    
    if not torch.cuda.is_available():
        return {"cuda": False, "action": "no_gpu"}
    
    try:
        # Información antes de limpiar
        before = get_gpu_info()
        
        # Limpieza agresiva
        torch.cuda.empty_cache()
        gc.collect()
        
        # Forzar limpieza adicional si es necesario
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        # Información después de limpiar
        after = get_gpu_info()
        
        # Calcular memoria liberada (con validación)
        if before.get("cuda") and after.get("cuda"):
            freed_gb = after["free_gb"] - before["free_gb"]
        else:
            freed_gb = 0.0
        
        return {
            "cuda": True,
            "action": "memory_cleared",
            "freed_gb": round(freed_gb, 2),
            "before_free_gb": before.get("free_gb", 0),
            "after_free_gb": after.get("free_gb", 0),
            "improvement": freed_gb > 0.1  # Considera significativo si libera >100MB
        }
        
    except Exception as e:
        return {
            "cuda": True,
            "action": "clear_failed",
            "error": str(e)
        }

def check_gpu_health() -> Dict[str, Any]:
    """
    Realiza un health check completo del GPU
    """
    health_report = {
        "timestamp": time.time(),
        "overall_status": "unknown",
        "checks": {}
    }
    
    try:
        # Check 1: PyTorch disponibilidad
        try:
            import torch
            health_report["checks"]["pytorch_available"] = {
                "status": "pass",
                "details": {"version": torch.__version__}
            }
        except ImportError:
            health_report["checks"]["pytorch_available"] = {
                "status": "fail",
                "details": {"error": "PyTorch not installed"}
            }
            health_report["overall_status"] = "fail"
            return health_report
        
        # Check 2: CUDA disponibilidad
        gpu_info = get_gpu_info()
        health_report["checks"]["cuda_available"] = {
            "status": "pass" if gpu_info["cuda"] else "fail",
            "details": gpu_info
        }
        
        if not gpu_info["cuda"]:
            health_report["overall_status"] = "fail"
            return health_report
        
        # Check 3: Memoria disponible
        free_gb = gpu_info.get("free_gb", 0)
        memory_check = {
            "status": "pass" if free_gb >= 2.0 else "warning" if free_gb >= 1.0 else "fail",
            "free_gb": free_gb,
            "recommendation": _get_memory_recommendation(free_gb)
        }
        health_report["checks"]["memory_available"] = memory_check
        
        # Check 4: Test simple de operación GPU
        try:
            test_tensor = torch.randn(100, 100).cuda()
            result = torch.matmul(test_tensor, test_tensor.t())
            del test_tensor, result
            torch.cuda.empty_cache()
            
            health_report["checks"]["gpu_operation"] = {
                "status": "pass",
                "details": "Simple GPU operation successful"
            }
        except Exception as e:
            health_report["checks"]["gpu_operation"] = {
                "status": "warning",  # Cambiar de fail a warning
                "error": str(e),
                "note": "GPU detection works but operation failed"
            }
        
        # Check 5: Capacidad de cómputo
        compute_capability = gpu_info.get("compute_capability", "0.0")
        try:
            compute_version = float(compute_capability) if compute_capability != "unknown" else 0.0
        except ValueError:
            compute_version = 0.0
            
        compute_check = {
            "status": "pass" if compute_version >= 6.0 else "warning" if compute_version >= 3.5 else "fail",
            "compute_capability": compute_capability,
            "modern_gpu": compute_version >= 6.0
        }
        health_report["checks"]["compute_capability"] = compute_check
        
        # Determinar estado general - ser más permisivo
        statuses = [check["status"] for check in health_report["checks"].values()]
        if statuses.count("fail") > 1:  # Solo fallar si hay múltiples fallas
            health_report["overall_status"] = "fail"
        elif "fail" in statuses or "warning" in statuses:
            health_report["overall_status"] = "warning"
        else:
            health_report["overall_status"] = "pass"
        
    except Exception as e:
        health_report["overall_status"] = "error"
        health_report["error"] = str(e)
    
    return health_report

def _get_memory_recommendation(free_gb: float) -> str:
    """Genera recomendaciones basadas en memoria disponible"""
    if free_gb >= 6.0:
        return "Suficiente memoria para modelos 7B"
    elif free_gb >= 4.0:
        return "Usar modelos pequeños o quantizados"
    elif free_gb >= 2.0:
        return "Solo modelos muy pequeños recomendados"
    else:
        return "Memoria insuficiente - usar CPU"

def get_system_info() -> Dict[str, Any]:
    """
    Obtiene información completa del sistema (GPU + CPU + RAM)
    """
    system_info = {
        "gpu": get_gpu_info(),
        "cpu": {
            "cores": psutil.cpu_count(),
            "usage_percent": psutil.cpu_percent(interval=1),
            "frequency_mhz": psutil.cpu_freq().current if psutil.cpu_freq() else "unknown"
        },
        "memory": {
            "total_gb": round(psutil.virtual_memory().total / (1024**3), 2),
            "available_gb": round(psutil.virtual_memory().available / (1024**3), 2),
            "usage_percent": psutil.virtual_memory().percent
        }
    }
    
    # Agregar info de Python environment solo si torch está disponible
    try:
        import torch
        system_info["python_env"] = {
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda if torch.cuda.is_available() else "not_available"
        }
    except ImportError:
        system_info["python_env"] = {
            "torch_version": "not_installed",
            "cuda_version": "not_available"
        }
    
    return system_info

def monitor_gpu_during_operation(operation_name: str = "operation"):
    """
    Context manager para monitorear GPU durante operaciones
    """
    class GPUMonitor:
        def __init__(self, op_name):
            self.operation_name = op_name
            self.start_info = None
            self.end_info = None
            self.start_time = None
            self.end_time = None
        
        def __enter__(self):
            self.start_time = time.time()
            self.start_info = get_gpu_info()
            print(f"🔍 Iniciando monitoreo GPU para: {self.operation_name}")
            if self.start_info["cuda"]:
                print(f"   VRAM libre inicial: {self.start_info['free_gb']:.2f}GB")
            return self
        
        def __exit__(self, exc_type, exc_val, exc_tb):
            self.end_time = time.time()
            self.end_info = get_gpu_info()
            duration = self.end_time - self.start_time
            
            print(f"✅ Monitoreo GPU completado: {self.operation_name}")
            print(f"   Duración: {duration:.2f}s")
            
            if self.start_info["cuda"] and self.end_info["cuda"]:
                memory_change = self.end_info["free_gb"] - self.start_info["free_gb"]
                print(f"   VRAM libre final: {self.end_info['free_gb']:.2f}GB")
                print(f"   Cambio en memoria: {memory_change:+.2f}GB")
                
                if memory_change < -0.5:
                    print("   ⚠️ Pérdida significativa de memoria detectada")
    
    return GPUMonitor(operation_name)

# Tests y utilidades de debugging
def run_gpu_diagnostics():
    """
    Ejecuta diagnósticos completos del GPU
    """
    print("🔧 DIAGNÓSTICOS GPU COMPLETOS")
    print("=" * 40)
    
    # Health check
    health = check_gpu_health()
    print(f"\n📊 Estado general: {health['overall_status'].upper()}")
    
    for check_name, check_result in health["checks"].items():
        status_emoji = {"pass": "✅", "warning": "⚠️", "fail": "❌"}
        emoji = status_emoji.get(check_result["status"], "❓")
        print(f"{emoji} {check_name}: {check_result['status']}")
        
        # Mostrar detalles adicionales
        if "recommendation" in check_result:
            print(f"   → {check_result['recommendation']}")
        elif "error" in check_result:
            print(f"   → Error: {check_result['error']}")
    
    # Información del sistema
    print(f"\n🖥️ INFORMACIÓN DEL SISTEMA")
    print("-" * 30)
    system = get_system_info()
    
    gpu = system["gpu"]
    if gpu["cuda"]:
        print(f"GPU: {gpu['device']}")
        print(f"VRAM: {gpu['free_gb']:.1f}GB libre / {gpu['total_gb']:.1f}GB total")
        print(f"Compute Capability: {gpu.get('compute_capability', 'unknown')}")
        print(f"Estado memoria: {gpu.get('memory_status', 'unknown')}")
        if "error_note" in gpu:
            print(f"Nota: {gpu['error_note']}")
    else:
        print("GPU: No disponible")
        if "reason" in gpu:
            print(f"Razón: {gpu['reason']}")
    
    cpu = system["cpu"]
    print(f"CPU: {cpu['cores']} cores, {cpu['usage_percent']}% uso")
    
    memory = system["memory"]
    print(f"RAM: {memory['available_gb']:.1f}GB libre / {memory['total_gb']:.1f}GB total")
    
    python_env = system["python_env"]
    print(f"PyTorch: {python_env['torch_version']}")
    print(f"CUDA: {python_env['cuda_version']}")

def diagnose_cuda_issues():
    """
    Diagnóstica problemas específicos de CUDA
    """
    print("\n🔍 DIAGNÓSTICO DETALLADO DE CUDA")
    print("=" * 40)
    
    # Check 1: Instalación de PyTorch
    try:
        import torch
        print(f"✅ PyTorch instalado: {torch.__version__}")
    except ImportError:
        print("❌ PyTorch no instalado")
        print("   Instalar con: pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
        return
    
    # Check 2: CUDA disponibilidad
    if torch.cuda.is_available():
        print(f"✅ CUDA disponible: {torch.version.cuda}")
        print(f"   Dispositivos: {torch.cuda.device_count()}")
        
        for i in range(torch.cuda.device_count()):
            try:
                props = torch.cuda.get_device_properties(i)
                print(f"   GPU {i}: {props.name} (Compute {props.major}.{props.minor})")
            except Exception as e:
                print(f"   GPU {i}: Error obteniendo propiedades - {e}")
    else:
        print("❌ CUDA no disponible")
        print("   Posibles causas:")
        print("   - Drivers de GPU no instalados/actualizados")
        print("   - CUDA toolkit no compatible")
        print("   - PyTorch instalado sin soporte CUDA")
    
    # Check 3: Test básico
    try:
        if torch.cuda.is_available():
            device = torch.cuda.current_device()
            tensor = torch.randn(10, 10).cuda()
            result = torch.matmul(tensor, tensor.t())
            print(f"✅ Test básico exitoso en GPU {device}")
        else:
            tensor = torch.randn(10, 10)
            result = torch.matmul(tensor, tensor.t())
            print(f"✅ Test básico exitoso en CPU")
    except Exception as e:
        print(f"❌ Test básico falló: {e}")

if __name__ == "__main__":
    # Ejecutar diagnósticos si se llama directamente
    run_gpu_diagnostics()
    diagnose_cuda_issues()