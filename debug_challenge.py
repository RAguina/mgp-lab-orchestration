"""
Debug script para analizar el problema con challenge flow config
"""

def debug_challenge_config():
    """Debug de la configuración del challenge flow."""
    
    print("🔍 Debugging Challenge Flow Configuration")
    print("=" * 50)
    
    try:
        # Test 1: Import básico
        print("1. Testing imports...")
        from langchain_integration.langgraph.orchestration.graph_configs import get_challenge_flow_config
        print("✅ Import successful")
        
        # Test 2: Crear configuración
        print("\n2. Creating challenge config...")
        config = get_challenge_flow_config()
        print(f"✅ Config created: {config}")
        
        # Test 3: Verificar estructura
        print("\n3. Analyzing config structure...")
        print(f"   Name: {config.name}")
        print(f"   Description: {config.description}")
        print(f"   Entry point: {config.entry_point}")
        
        # Test 4: Verificar nodos
        print("\n4. Analyzing nodes...")
        if config.nodes:
            print(f"   Nodes count: {len(config.nodes)}")
            for i, node in enumerate(config.nodes):
                print(f"   Node {i}: {node.id} (type: {node.type})")
        else:
            print("   ❌ No nodes found!")
        
        # Test 5: Verificar edges
        print("\n5. Analyzing edges...")
        if config.edges:
            print(f"   Edges count: {len(config.edges)}")
            for i, edge in enumerate(config.edges):
                print(f"   Edge {i}: {edge.source} → {edge.target}")
        else:
            print("   ❌ No edges found!")
        
        # Test 6: Verificar tipos de datos
        print("\n6. Verifying data types...")
        print(f"   config.nodes type: {type(config.nodes)}")
        print(f"   config.edges type: {type(config.edges)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error during debug: {e}")
        import traceback
        traceback.print_exc()
        return False

def debug_graph_builder():
    """Debug del graph builder con challenge flow."""
    
    print("\n🔧 Debugging Graph Builder")
    print("=" * 50)
    
    try:
        from langchain_integration.langgraph.orchestration import get_graph_builder
        
        builder = get_graph_builder()
        print("✅ Graph builder obtained")
        
        # Intentar construir challenge flow
        print("\nBuilding challenge flow...")
        graph = builder.build_challenge_flow_graph()
        print("✅ Challenge flow built successfully!")
        
        return True
        
    except Exception as e:
        print(f"❌ Error building challenge flow: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🚨 Challenge Flow Debug Session")
    print("=" * 60)
    
    # Debug configuración
    config_ok = debug_challenge_config()
    
    if config_ok:
        # Debug builder
        builder_ok = debug_graph_builder()
        
        if builder_ok:
            print("\n🎉 Challenge flow debugging successful!")
        else:
            print("\n💥 Challenge flow building failed")
    else:
        print("\n💥 Challenge flow config failed")