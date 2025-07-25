# test_orchestrator_incremental.py
"""
Test incremental del orchestrator con sistema de suma básica
Cada nodo suma 100 al valor, permitiendo trackear el flujo completo
"""

import time
import json
from typing import Dict, Any
from langchain_integration.langgraph.routing_agent import run_orchestrator

class OrchestratorIncrementalTester:
    """
    Tester que trackea el flujo del orchestrator con suma incremental
    """
    
    def __init__(self):
        self.test_results = []
        self.expected_flow_paths = {
            "chat": ["task_analyzer", "execution", "output_validator", "summarizer"],
            "code": ["task_analyzer", "resource_monitor", "execution", "output_validator", "summarizer"],
            "analysis": ["task_analyzer", "resource_monitor", "execution", "output_validator", "rubric_generator", "summarizer"],
            "creative": ["task_analyzer", "execution", "output_validator", "comparison", "summarizer"]
        }
    
    def run_increment_test(self, prompt: str, expected_task_type: str = None) -> Dict[str, Any]:
        """
        Ejecuta test incremental del orchestrator
        """
        print(f"\n🚀 INICIANDO TEST INCREMENTAL")
        print(f"📝 Prompt: '{prompt}'")
        print(f"🎯 Task type esperado: {expected_task_type or 'auto-detect'}")
        print("=" * 60)
        
        start_time = time.time()
        
        try:
            # Ejecutar orchestrator
            result = run_orchestrator(prompt)
            
            total_time = time.time() - start_time
            
            # Analizar resultado
            analysis = self._analyze_orchestrator_result(result, prompt, expected_task_type, total_time)
            
            # Guardar para comparación
            self.test_results.append(analysis)
            
            # Mostrar resultados
            self._display_test_results(analysis)
            
            return analysis
            
        except Exception as e:
            error_analysis = {
                "prompt": prompt,
                "success": False,
                "error": str(e),
                "total_time": time.time() - start_time
            }
            
            print(f"❌ TEST FALLÓ: {str(e)}")
            return error_analysis
    
    def _analyze_orchestrator_result(self, result: Dict[str, Any], prompt: str, 
                                   expected_task_type: str, total_time: float) -> Dict[str, Any]:
        """
        Analiza el resultado del orchestrator
        """
        # Extraer información básica
        final_output = result.get("output", "")
        metrics = result.get("metrics", {})
        flow = result.get("flow", {})
        
        # Analizar flow ejecutado
        nodes_executed = []
        if "nodes" in flow:
            nodes_executed = [node.get("name", node.get("id", "unknown")) for node in flow["nodes"]]
        
        # Detectar task type del output si es posible
        detected_task_type = "unknown"
        if "code" in prompt.lower() or "function" in prompt.lower():
            detected_task_type = "code"
        elif "explain" in prompt.lower() or "what is" in prompt.lower():
            detected_task_type = "chat"
        elif "analyze" in prompt.lower() or "compare" in prompt.lower():
            detected_task_type = "analysis"
        
        # Calcular "suma de nodos" (simulación incremental)
        node_sum = len(nodes_executed) * 100  # 100 por cada nodo ejecutado
        
        # Verificar flujo esperado
        expected_flow = self.expected_flow_paths.get(detected_task_type, [])
        flow_matches = self._check_flow_match(nodes_executed, expected_flow)
        
        # Análisis de calidad
        quality_indicators = {
            "has_output": len(final_output) > 0,
            "output_length": len(final_output),
            "execution_successful": result.get("success", True),
            "has_metrics": len(metrics) > 0,
            "has_flow_data": len(nodes_executed) > 0
        }
        
        # Métricas de performance
        performance_metrics = {
            "total_time": total_time,
            "orchestrator_time": metrics.get("totalTime", 0) / 1000 if metrics.get("totalTime") else 0,
            "nodes_executed": len(nodes_executed),
            "tokens_generated": metrics.get("tokensGenerated", 0),
            "cache_hit": metrics.get("cacheHit", False)
        }
        
        analysis = {
            "prompt": prompt,
            "expected_task_type": expected_task_type,
            "detected_task_type": detected_task_type,
            "success": len(final_output) > 0 and not ("error" in final_output.lower()),
            "flow_analysis": {
                "nodes_executed": nodes_executed,
                "expected_flow": expected_flow,
                "flow_matches": flow_matches,
                "node_sum": node_sum  # Suma incremental simulada
            },
            "quality_indicators": quality_indicators,
            "performance_metrics": performance_metrics,
            "output_preview": final_output[:200] + "..." if len(final_output) > 200 else final_output
        }
        
        return analysis
    
    def _check_flow_match(self, executed_nodes: list, expected_nodes: list) -> Dict[str, Any]:
        """
        Verifica si el flujo ejecutado coincide con el esperado
        """
        # Normalizar nombres de nodos para comparación
        executed_normalized = [self._normalize_node_name(node) for node in executed_nodes]
        expected_normalized = [self._normalize_node_name(node) for node in expected_nodes]
        
        # Verificar intersección
        common_nodes = set(executed_normalized) & set(expected_normalized)
        missing_nodes = set(expected_normalized) - set(executed_normalized)
        extra_nodes = set(executed_normalized) - set(expected_normalized)
        
        # Calcular score de coincidencia
        if expected_normalized:
            match_score = len(common_nodes) / len(expected_normalized)
        else:
            match_score = 1.0 if not executed_normalized else 0.0
        
        return {
            "match_score": round(match_score * 100, 1),
            "common_nodes": list(common_nodes),
            "missing_nodes": list(missing_nodes),
            "extra_nodes": list(extra_nodes),
            "expected_count": len(expected_normalized),
            "executed_count": len(executed_normalized)
        }
    
    def _normalize_node_name(self, node_name: str) -> str:
        """
        Normaliza nombres de nodos para comparación
        """
        name = node_name.lower().replace("_", "").replace("-", "")
        
        # Mapeos comunes
        mappings = {
            "taskanalyzer": "analyzer",
            "resourcemonitor": "monitor", 
            "executionworker": "execution",
            "modelexecution": "execution",
            "outputvalidator": "validator",
            "summarygenerator": "summarizer",
            "rubricgenerator": "rubric",
            "historyreader": "history"
        }
        
        return mappings.get(name, name)
    
    def _display_test_results(self, analysis: Dict[str, Any]):
        """
        Muestra los resultados del test de manera clara
        """
        print("\n📊 RESULTADOS DEL TEST INCREMENTAL")
        print("=" * 60)
        
        # Estado general
        status = "✅ ÉXITO" if analysis["success"] else "❌ FALLO"
        print(f"🎯 Estado: {status}")
        print(f"📝 Task Type: {analysis['detected_task_type']}")
        
        # Flujo de nodos
        flow = analysis["flow_analysis"]
        print(f"\n🔄 FLUJO DE NODOS:")
        print(f"   Ejecutados: {' → '.join(flow['nodes_executed'])}")
        print(f"   Suma Incremental: {flow['node_sum']} (100 x {len(flow['nodes_executed'])} nodos)")
        print(f"   Match Score: {flow['flow_matches']['match_score']}%")
        
        if flow['flow_matches']['missing_nodes']:
            print(f"   ⚠️ Nodos faltantes: {', '.join(flow['flow_matches']['missing_nodes'])}")
        
        if flow['flow_matches']['extra_nodes']:
            print(f"   ➕ Nodos extra: {', '.join(flow['flow_matches']['extra_nodes'])}")
        
        # Performance
        perf = analysis["performance_metrics"]
        print(f"\n⚡ PERFORMANCE:")
        print(f"   Tiempo Total: {perf['total_time']:.2f}s")
        print(f"   Nodos Ejecutados: {perf['nodes_executed']}")
        print(f"   Tokens Generados: {perf['tokens_generated']}")
        print(f"   Cache Hit: {'✅' if perf['cache_hit'] else '❌'}")
        
        # Calidad
        quality = analysis["quality_indicators"]
        print(f"\n📈 CALIDAD:")
        print(f"   Tiene Output: {'✅' if quality['has_output'] else '❌'}")
        print(f"   Longitud Output: {quality['output_length']} chars")
        print(f"   Ejecución Exitosa: {'✅' if quality['execution_successful'] else '❌'}")
        
        # Preview del output
        print(f"\n📄 OUTPUT PREVIEW:")
        print(f"   {analysis['output_preview']}")
        
        print("=" * 60)
    
    def run_test_suite(self):
        """
        Ejecuta suite completa de tests incrementales
        """
        print("🎯 INICIANDO SUITE DE TESTS INCREMENTALES")
        print("Cada test suma 100 por nodo ejecutado para trackear el flujo")
        print("=" * 80)
        
        test_cases = [
            {
                "prompt": "Hello, how are you?",
                "expected_task_type": "chat",
                "description": "Test básico de chat - flujo simple"
            },
            {
                "prompt": "Write a Python function to sort a list",
                "expected_task_type": "code", 
                "description": "Test de código - flujo con resource monitor"
            },
            {
                "prompt": "Explain what is machine learning",
                "expected_task_type": "chat",
                "description": "Test técnico - flujo explicativo"
            },
            {
                "prompt": "Analyze the pros and cons of AI",
                "expected_task_type": "analysis",
                "description": "Test de análisis - flujo complejo"
            }
        ]
        
        results_summary = []
        
        for i, test_case in enumerate(test_cases, 1):
            print(f"\n🧪 TEST {i}/4: {test_case['description']}")
            print("-" * 50)
            
            result = self.run_increment_test(
                test_case["prompt"], 
                test_case["expected_task_type"]
            )
            
            results_summary.append({
                "test_number": i,
                "description": test_case["description"],
                "success": result["success"],
                "node_sum": result["flow_analysis"]["node_sum"],
                "nodes_count": len(result["flow_analysis"]["nodes_executed"]),
                "total_time": result["performance_metrics"]["total_time"]
            })
            
            # Esperar entre tests para no sobrecargar
            time.sleep(2)
        
        # Resumen final
        self._display_suite_summary(results_summary)
        
        return results_summary
    
    def _display_suite_summary(self, results_summary: list):
        """
        Muestra resumen de la suite completa
        """
        print("\n" + "=" * 80)
        print("📋 RESUMEN DE LA SUITE DE TESTS")
        print("=" * 80)
        
        total_tests = len(results_summary)
        successful_tests = sum(1 for r in results_summary if r["success"])
        
        print(f"🎯 Tests Ejecutados: {total_tests}")
        print(f"✅ Tests Exitosos: {successful_tests}")
        print(f"❌ Tests Fallidos: {total_tests - successful_tests}")
        print(f"📊 Success Rate: {(successful_tests/total_tests*100):.1f}%")
        
        print(f"\n🔄 DETALLES POR TEST:")
        for result in results_summary:
            status = "✅" if result["success"] else "❌"
            print(f"   {status} Test {result['test_number']}: {result['description']}")
            print(f"      Suma Incremental: {result['node_sum']} ({result['nodes_count']} nodos)")
            print(f"      Tiempo: {result['total_time']:.2f}s")
        
        # Estadísticas generales
        if results_summary:
            avg_nodes = sum(r["nodes_count"] for r in results_summary) / len(results_summary)
            avg_time = sum(r["total_time"] for r in results_summary) / len(results_summary)
            total_sum = sum(r["node_sum"] for r in results_summary)
            
            print(f"\n📈 ESTADÍSTICAS GENERALES:")
            print(f"   Promedio de nodos por test: {avg_nodes:.1f}")
            print(f"   Tiempo promedio por test: {avg_time:.2f}s")
            print(f"   Suma total acumulada: {total_sum}")
        
        print("=" * 80)


# Función main para ejecutar tests
def main():
    """
    Función principal para ejecutar tests incrementales
    """
    tester = OrchestratorIncrementalTester()
    
    print("🚀 ORCHESTRATOR INCREMENTAL TESTER")
    print("Cada nodo suma 100 - permite trackear el flujo completo")
    print("")
    
    # Ejecutar suite completa
    results = tester.run_test_suite()
    
    # Guardar resultados
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    results_file = f"orchestrator_test_results_{timestamp}.json"
    
    try:
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\n💾 Resultados guardados en: {results_file}")
    except Exception as e:
        print(f"\n⚠️ No se pudieron guardar resultados: {e}")
    
    return results


if __name__ == "__main__":
    main()