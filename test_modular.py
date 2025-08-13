"""
Test script para verificar que la estructura modular funciona correctamente
"""

def test_imports():
    """Test 1: Verificar imports"""
    print("🔍 Test 1: Verificando imports...")
    try:
        from langchain_integration.langgraph.orchestration import build_routing_graph
        from langchain_integration.langgraph.routing_agent import run_routing_agent, run_orchestrator
        print("✅ Todos los imports funcionan correctamente")
        return True
    except Exception as e:
        print(f"❌ Error en imports: {e}")
        return False

def test_graph_builder():
    """Test 2: Verificar GraphBuilder"""
    print("\n🔍 Test 2: Verificando GraphBuilder...")
    try:
        from langchain_integration.langgraph.orchestration import get_graph_builder
        builder = get_graph_builder()
        
        # Verificar que tiene los nodos por defecto
        expected_nodes = ['analyzer', 'monitor', 'executor', 'validator', 'history', 'summarizer']
        actual_nodes = list(builder.registered_nodes.keys())
        
        print(f"📋 Nodos registrados: {actual_nodes}")
        
        for node in expected_nodes:
            if node not in actual_nodes:
                print(f"❌ Nodo faltante: {node}")
                return False
        
        print("✅ GraphBuilder tiene todos los nodos esperados")
        return True
    except Exception as e:
        print(f"❌ Error en GraphBuilder: {e}")
        return False

def test_graph_building():
    """Test 3: Verificar construcción de grafos"""
    print("\n🔍 Test 3: Verificando construcción de grafos...")
    try:
        from langchain_integration.langgraph.orchestration import build_routing_graph
        
        # Test linear flow
        graph_linear = build_routing_graph("linear")
        print("✅ Grafo linear construido correctamente")
        
        # Test challenge flow (debería hacer fallback a linear)
        graph_challenge = build_routing_graph("challenge")
        print("✅ Grafo challenge construido (fallback a linear)")
        
        # Test unknown flow (debería hacer fallback a linear)
        graph_unknown = build_routing_graph("unknown_flow")
        print("✅ Grafo unknown construido (fallback a linear)")
        
        return True
    except Exception as e:
        print(f"❌ Error construyendo grafos: {e}")
        return False

def test_simple_execution():
    """Test 4: Verificar ejecución simple"""
    print("\n🔍 Test 4: Verificando ejecución simple...")
    try:
        from langchain_integration.langgraph.routing_agent import run_routing_agent
        
        # Test básico con prompt simple
        result = run_routing_agent(
            "Hola, esto es una prueba", 
            flow_type="linear", 
            verbose=False
        )
        
        # Verificar que el resultado tiene las claves esperadas
        expected_keys = ['input', 'output', 'task_type', 'selected_model']
        for key in expected_keys:
            if key not in result:
                print(f"❌ Clave faltante en resultado: {key}")
                return False
        
        print(f"✅ Ejecución completada. Task type: {result.get('task_type')}")
        print(f"📝 Output (primeros 100 chars): {result.get('output', '')[:100]}...")
        return True
    except Exception as e:
        print(f"❌ Error en ejecución: {e}")
        return False

def test_api_wrapper():
    """Test 5: Verificar wrapper de API"""
    print("\n🔍 Test 5: Verificando wrapper de API...")
    try:
        from langchain_integration.langgraph.routing_agent import run_orchestrator
        
        result = run_orchestrator("Test simple para API", flow_type="linear")
        
        # Verificar estructura de respuesta API
        expected_keys = ['flow', 'output', 'metrics']
        for key in expected_keys:
            if key not in result:
                print(f"❌ Clave faltante en respuesta API: {key}")
                return False
        
        # Verificar que flow tiene nodes y edges
        flow = result['flow']
        if 'nodes' not in flow or 'edges' not in flow:
            print("❌ Flow no tiene estructura nodes/edges")
            return False
        
        print(f"✅ API wrapper funcionando. Nodos: {len(flow['nodes'])}")
        return True
    except Exception as e:
        print(f"❌ Error en API wrapper: {e}")
        return False

def main():
    """Ejecuta todos los tests"""
    print("🧪 Testing estructura modular del orquestador")
    print("=" * 50)
    
    tests = [
        test_imports,
        test_graph_builder,
        test_graph_building,
        test_simple_execution,
        test_api_wrapper
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"❌ Test falló con excepción: {e}")
            results.append(False)
    
    # Resumen
    print("\n" + "=" * 50)
    print("📊 Resumen de tests:")
    
    passed = sum(results)
    total = len(results)
    
    for i, (test, result) in enumerate(zip(tests, results), 1):
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  Test {i} ({test.__name__}): {status}")
    
    print(f"\n🎯 Total: {passed}/{total} tests pasaron")
    
    if passed == total:
        print("🎉 ¡Todos los tests pasaron! La estructura modular funciona correctamente.")
        print("\n🚀 Estás listo para implementar el challenge flow.")
    else:
        print("⚠️  Algunos tests fallaron. Revisa los errores antes de continuar.")
    
    return passed == total

if __name__ == "__main__":
    main()