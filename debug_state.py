"""
Debug script para ver qué hay en el estado después del challenge flow
"""

from langchain_integration.langgraph.routing_agent import run_routing_agent

def debug_challenge_state():
    """Debug del estado del challenge flow."""
    
    print("🔍 Debugging Challenge Flow State")
    print("=" * 50)
    
    # Ejecutar challenge flow
    result = run_routing_agent(
        "Crea una función Python simple para sumar dos números", 
        flow_type="challenge", 
        verbose=False
    )
    
    print("\n📋 KEYS EN EL ESTADO FINAL:")
    print("=" * 50)
    for key in sorted(result.keys()):
        value = result[key]
        if isinstance(value, str):
            preview = value[:100] + "..." if len(value) > 100 else value
        else:
            preview = str(value)
        print(f"  {key}: {preview}")
    
    print("\n🔍 BUSCANDO OUTPUTS ESPECÍFICOS:")
    print("=" * 50)
    
    challenge_outputs = ['creator_output', 'challenger_output', 'refiner_output']
    for output_key in challenge_outputs:
        if output_key in result:
            content = result[output_key]
            print(f"✅ {output_key}: {len(content)} chars")
            print(f"   Preview: {content[:100]}...")
        else:
            print(f"❌ {output_key}: NO ENCONTRADO")
    
    return result

if __name__ == "__main__":
    debug_challenge_state()