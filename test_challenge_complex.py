"""
Test complejo para Challenge Flow - Arquitectura de software
"""

from langchain_integration.langgraph.routing_agent import run_routing_agent

def test_challenge_flow_complex():
    """Test del challenge flow con un problema más complejo."""
    
    print("🥊 Testing Challenge Flow - Arquitectura Compleja")
    print("=" * 60)
    
    # Prompt más desafiante que realmente se beneficia del debate
    prompt = """Diseña una arquitectura de microservicios para un e-commerce que maneje:
- 100,000 usuarios concurrentes
- Pagos seguros
- Inventario en tiempo real
- Recomendaciones personalizadas
- Sistema de reviews
Incluye patrones de diseño, tecnologías y consideraciones de escalabilidad."""
    
    print(f"📝 Prompt: {prompt[:100]}...")
    print(f"🔄 Flow: challenge (Creator → Challenger → Refiner)")
    print("🎯 Objetivo: Ver si el debate mejora significativamente la arquitectura")
    print("\n" + "=" * 60)
    
    try:
        result = run_routing_agent(
            prompt, 
            flow_type="challenge", 
            verbose=True
        )
        
        print("\n" + "🎯 RESULTADO DEL CHALLENGE FLOW:")
        print("=" * 60)
        output = result.get('output', 'Sin output')
        print(output)
        
        # Análisis de calidad
        print("\n" + "🔍 ANÁLISIS DE CALIDAD:")
        print("=" * 60)
        
        # Buscar indicadores de calidad en el output
        quality_indicators = {
            "microservicios": "microservice" in output.lower() or "microservicio" in output.lower(),
            "escalabilidad": "escalab" in output.lower() or "scale" in output.lower(),
            "seguridad": "segur" in output.lower() or "security" in output.lower(),
            "patrones": "pattern" in output.lower() or "patrón" in output.lower(),
            "tecnologías": "tech" in output.lower() or "tecnolog" in output.lower(),
            "concurrencia": "concurren" in output.lower() or "concurrent" in output.lower()
        }
        
        for indicator, found in quality_indicators.items():
            status = "✅" if found else "❌"
            print(f"{status} {indicator.capitalize()}: {'Mencionado' if found else 'No mencionado'}")
        
        # Verificar ejecución de nodos
        print("\n" + "🔍 ANÁLISIS DEL FLUJO:")
        print("=" * 60)
        
        node_outputs = ['creator_output', 'challenger_output', 'refiner_output']
        for node_output in node_outputs:
            if node_output in result:
                print(f"✅ {node_output.replace('_output', '').capitalize()} ejecutado")
            else:
                print(f"❌ {node_output.replace('_output', '').capitalize()} no encontrado")
        
        # Métricas
        metrics = result.get('execution_metrics', {})
        total_time = metrics.get('total_time_ms', 0)
        print(f"\n⏱️ Tiempo total: {total_time/1000:.2f}s")
        
        # Scoring
        quality_score = sum(quality_indicators.values()) / len(quality_indicators) * 100
        print(f"📊 Score de calidad: {quality_score:.1f}%")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Error en challenge flow: {e}")
        return False

if __name__ == "__main__":
    success = test_challenge_flow_complex()
    
    if success:
        print("\n🎉 Challenge flow complejo completado!")
        print("💡 Observa cómo el Challenger identifica problemas y el Refiner los soluciona")
    else:
        print("\n💥 Challenge flow complejo falló")