#!/usr/bin/env python3
"""
Test rápido del endpoint RAG extendido
"""

import requests
import json

def test_rag_endpoint():
    """Test básico del endpoint con configuración RAG"""
    
    endpoint = "http://localhost:8000/orchestrate"
    
    # Test 1: Request RAG con mock
    rag_request = {
        "prompt": "¿Qué es machine learning?",
        "model": "mistral7b",
        "flow_type": "rag_simple",  # Será convertido a 'challenge' con mock RAG
        "embedding_model": "bge-m3",
        "vector_store": "milvus",
        "rag_config": {
            "top_k": 3,
            "similarity_threshold": 0.7
        }
    }
    
    print("Testing RAG endpoint...")
    print(f"Request: {json.dumps(rag_request, indent=2)}")
    
    try:
        response = requests.post(endpoint, json=rag_request, timeout=30)
        print(f"\nStatus: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"Response received")
            print(f"   Success: {result.get('success')}")
            print(f"   Output length: {len(result.get('output', ''))}")
            print(f"   Flow nodes: {len(result.get('flow', {}).get('nodes', []))}")
            print(f"   Tools detected in logs: Check server logs for RAG tool processing")
        else:
            print(f"Error: {response.text}")
            
    except requests.exceptions.ConnectionError:
        print("Server not running. Start with:")
        print("   python api/server.py")
    except Exception as e:
        print(f"Test failed: {e}")

def test_health_endpoint():
    """Test del health check con nuevos campos RAG"""
    
    health_url = "http://localhost:8000/orchestrate/"
    
    print("\nTesting health endpoint...")
    
    try:
        response = requests.get(health_url, timeout=10)
        if response.status_code == 200:
            health = response.json()
            print("Health check:")
            print(f"   Orchestrator available: {health.get('orchestrator_available')}")
            print(f"   Supported flows: {health.get('supported_flows')}")
            print(f"   Supported embeddings: {health.get('supported_embeddings')}")
            print(f"   Supported vector stores: {health.get('supported_vector_stores')}")
        else:
            print(f"Health check failed: {response.status_code}")
    except Exception as e:
        print(f"Health check error: {e}")

if __name__ == "__main__":
    test_health_endpoint()
    test_rag_endpoint()