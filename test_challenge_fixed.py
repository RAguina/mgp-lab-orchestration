"""
Test del Challenge Flow - Versión ARREGLADA
"""

from langchain_integration.langgraph.routing_agent import run_routing_agent, run_orchestrator

def test_challenge_flow_fixed():
    """Test del challenge flow con análisis mejorado."""
    
    print("🥊 Testing Challenge Flow - Versión Arreglada")
    print("=" * 60)
    
    # Prompt simple para test rápido
    prompt = "Crea una función Python para validar emails"
    
    print(f"📝 Prompt: {prompt}")
    print(f"🔄 Flow: challenge (Creator → Challenger → Refiner)")
    print("\n" + "=" * 60)
    
    try:
        # Usar run_orchestrator para obtener flow data
        api_result = run_orchestrator(prompt, flow_type="challenge")
        
        print("\n🎯 RESULTADO DEL CHALLENGE FLOW:")
        print("=" * 60)
        print(api_result.get('output', 'Sin output'))
        
        # Analizar el flow construido
        flow_data = api_result.get('flow', {})
        nodes = flow_data.get('nodes', [])
        edges = flow_data.get('edges', [])
        
        print("\n🔍 ANÁLISIS DEL FLUJO (ARREGLADO):")
        print("=" * 60)
        
        # Verificar nodos por ID
        node_ids = [node['id'] for node in nodes]
        expected_nodes = ['creator', 'challenger', 'refiner']
        
        for expected_node in expected_nodes:
            if expected_node in node_ids:
                node_data = next(node for node in nodes if node['id'] == expected_node)
                print(f"✅ {expected_node.capitalize()}: {node_data['status']}")
                print(f"   Output: {node_data['output'][:80]}...")
            else:
                print(f"❌ {expected_node.capitalize()}: no encontrado")
        
        # Análisis de calidad basado en el output final
        output = api_result.get('output', '').lower()
        quality_indicators = {
            "validación": "valid" in output or "validar" in output,
            "email": "email" in output or "@" in output,
            "regex": "regex" in output or "re." in output,
            "función": "def " in output or "function" in output,
            "python": "python" in output or "def " in output
        }
        
        print(f"\n📊 ANÁLISIS DE CALIDAD:")
        print("=" * 60)
        for indicator, found in quality_indicators.items():
            status = "✅" if found else "❌"
            print(f"{status} {indicator.capitalize()}: {'Presente' if found else 'Ausente'}")
        
        quality_score = sum(quality_indicators.values()) / len(quality_indicators) * 100
        
        # Métricas
        metrics = api_result.get('metrics', {})
        total_time = metrics.get('totalTime', 0)
        workers = metrics.get('workersExecuted', 0)
        
        print(f"\n📈 MÉTRICAS:")
        print("=" * 60)
        print(f"⏱️  Tiempo total: {total_time/1000:.1f}s")
        print(f"👥 Workers ejecutados: {workers}")
        print(f"📊 Score de calidad: {quality_score:.1f}%")
        print(f"🔄 Flow type: {metrics.get('flowType', 'unknown')}")
        
        # Análisis de ejecuciones secuenciales
        models_used = metrics.get('modelsUsed', [])
        print(f"🤖 Modelos utilizados: {', '.join(models_used)}")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Error en challenge flow: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_challenge_flow_fixed()
    
    if success:
        print("\n🎉 Challenge flow test arreglado completado!")
        print("💡 Ahora deberías ver correctamente los 3 nodos ejecutados")
    else:
        print("\n💥 Challenge flow test aún tiene problemas")