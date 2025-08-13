"""
Test específico para el Challenge Flow
"""

from langchain_integration.langgraph.routing_agent import run_routing_agent

def test_challenge_flow():
    """Test del flujo de challenge/debate entre modelos."""
    
    print("🥊 Testing Challenge Flow...")
    print("=" * 50)
    
    # Prompt que se beneficia del debate entre modelos
    prompt = "Crea una función Python para validar contraseñas seguras"
    
    print(f"📝 Prompt: {prompt}")
    print(f"🔄 Flow: challenge (Creator → Challenger → Refiner)")
    print("\n" + "=" * 50)
    
    try:
        result = run_routing_agent(
            prompt, 
            flow_type="challenge", 
            verbose=True
        )
        
        print("\n" + "🎯 RESULTADO DEL CHALLENGE FLOW:")
        print("=" * 50)
        print(result.get('output', 'Sin output'))
        
        # Verificar que se ejecutaron los nodos específicos del challenge
        print("\n" + "🔍 ANÁLISIS DEL FLUJO:")
        print("=" * 50)
        
        if 'creator_output' in result:
            print("✅ Creator ejecutado")
        else:
            print("❌ Creator no encontrado")
            
        if 'challenger_output' in result:
            print("✅ Challenger ejecutado")
        else:
            print("❌ Challenger no encontrado")
            
        if 'refiner_output' in result:
            print("✅ Refiner ejecutado")
        else:
            print("❌ Refiner no encontrado")
        
        # Mostrar métricas
        metrics = result.get('execution_metrics', {})
        total_time = metrics.get('total_time_ms', 0)
        print(f"\n⏱️ Tiempo total: {total_time/1000:.2f}s")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Error en challenge flow: {e}")
        return False

if __name__ == "__main__":
    success = test_challenge_flow()
    
    if success:
        print("\n🎉 Challenge flow test completado!")
        print("💡 Tip: Compara el resultado con el linear flow para ver la diferencia")
    else:
        print("\n💥 Challenge flow test falló")