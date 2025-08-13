"""
Script para rastrear exactamente dónde ocurre el error 'NoneType' object is not iterable
"""

import traceback

def trace_challenge_error():
    """Rastrea el error paso a paso con stack trace completo."""
    
    print("🔍 Rastreando error 'NoneType' object is not iterable")
    print("=" * 60)
    
    try:
        # Paso 1: Imports
        print("1. Importing modules...")
        from langchain_integration.langgraph.orchestration.graph_configs import get_challenge_flow_config
        from langchain_integration.langgraph.orchestration import get_graph_builder
        print("✅ Imports successful")
        
        # Paso 2: Get config
        print("\n2. Getting challenge config...")
        config = get_challenge_flow_config()
        print("✅ Config obtained")
        
        # Paso 3: Get builder
        print("\n3. Getting graph builder...")
        builder = get_graph_builder()
        print("✅ Builder obtained")
        
        # Paso 4: Build graph FROM CONFIG (aquí está el error)
        print("\n4. Building graph from config...")
        print("   Calling build_graph_from_config()...")
        
        # ESTE ES EL PASO QUE FALLA
        graph = builder.build_graph_from_config(config)
        print("✅ Graph built successfully!")
        
    except Exception as e:
        print(f"\n❌ ERROR CAPTURADO: {e}")
        print(f"❌ ERROR TYPE: {type(e)}")
        print("\n📍 STACK TRACE COMPLETO:")
        print("=" * 60)
        traceback.print_exc()
        print("=" * 60)
        
        # Análisis adicional
        print("\n🔬 ANÁLISIS ADICIONAL:")
        if "'NoneType' object is not iterable" in str(e):
            print("✅ Confirmado: Este es el error que buscamos")
            print("💡 El error indica que algo que esperamos que sea una lista/tupla es None")
            print("💡 Buscar en el stack trace qué variable es None")
        else:
            print("❓ Este es un error diferente")

if __name__ == "__main__":
    trace_challenge_error()