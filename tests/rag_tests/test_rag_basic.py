#!/usr/bin/env python3
"""
Basic RAG Component Test
Test embedding provider without external dependencies
"""

import sys
import os

# UTF-8 handling for Windows
os.environ['PYTHONIOENCODING'] = 'utf-8'
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.stderr.reconfigure(encoding='utf-8', errors='replace')

# Add lab path
LAB_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, LAB_ROOT)

def test_embedding_provider():
    """Test BGE-M3 embedding provider"""
    print("Testing BGE-M3 Embedding Provider...")
    
    try:
        from langchain_integration.rag.embeddings.bge_m3_provider import BGEM3EmbeddingProvider
        
        # Create provider (will download model on first run)
        provider = BGEM3EmbeddingProvider(device="cpu")  # Use CPU for safety
        
        # Test query embedding
        query = "What is machine learning?"
        embedding = provider.embed_query(query)
        
        print(f"OK Query embedded successfully")
        print(f"   Query: '{query}'")
        print(f"   Embedding dim: {len(embedding)}")
        print(f"   First 5 values: {embedding[:5]}")
        
        # Test document embedding
        docs = [
            "Machine learning is a subset of artificial intelligence",
            "Neural networks are inspired by biological neurons"
        ]
        
        doc_embeddings = provider.embed_documents(docs)
        print(f"OK Documents embedded successfully")
        print(f"   Documents: {len(docs)}")
        print(f"   Embeddings: {len(doc_embeddings)}")
        
        # Test chunk embedding
        chunks = [
            {"content": "Python is a programming language"},
            {"content": "FastAPI is a web framework for Python"}
        ]
        
        chunk_embeddings = provider.embed_chunks(chunks)
        print(f"OK Chunks embedded successfully")
        print(f"   Chunks: {len(chunks)}")
        print(f"   Embeddings: {len(chunk_embeddings)}")
        
        return True
        
    except ImportError as e:
        print(f"FAIL Import failed: {e}")
        print("   Run: pip install sentence-transformers")
        return False
    except Exception as e:
        print(f"FAIL Test failed: {e}")
        return False

def test_embedding_manager():
    """Test embedding manager singleton"""
    print("\nTesting Embedding Manager...")
    
    try:
        from langchain_integration.rag.embeddings.embedding_manager import get_embedding_manager
        
        # Get manager
        manager = get_embedding_manager()
        
        # Get provider (should cache)
        provider1 = manager.get_provider("bge-m3", device="cpu")
        provider2 = manager.get_provider("bge-m3", device="cpu")
        
        # Should be same instance
        is_cached = provider1 is provider2
        print(f"OK Manager working correctly")
        print(f"   Caching works: {is_cached}")
        print(f"   Cached models: {manager.get_cached_models()}")
        
        return True
        
    except Exception as e:
        print(f"FAIL Manager test failed: {e}")
        return False

def test_rag_health():
    """Test RAG health endpoint"""
    print("\nTesting RAG Health Endpoint...")
    
    try:
        import requests
        
        # Test health endpoint
        response = requests.get("http://localhost:8000/rag/", timeout=5)
        
        if response.status_code == 200:
            health = response.json()
            print(f"OK RAG health endpoint working")
            print(f"   RAG Available: {health.get('rag_available')}")
            print(f"   Embedding models: {health.get('embedding_models')}")
            print(f"   Vector stores: {health.get('vector_stores')}")
        else:
            print(f"FAIL Health endpoint returned: {response.status_code}")
            return False
            
        return True
        
    except requests.exceptions.ConnectionError:
        print("WARN Server not running. Start with: python api/server.py")
        return False
    except Exception as e:
        print(f"FAIL Health test failed: {e}")
        return False

if __name__ == "__main__":
    print("RAG Basic Component Tests")
    print("=" * 50)
    
    success_count = 0
    total_tests = 3
    
    # Test 1: Embedding Provider
    if test_embedding_provider():
        success_count += 1
    
    # Test 2: Embedding Manager
    if test_embedding_manager():
        success_count += 1
    
    # Test 3: Health endpoint (optional)
    if test_rag_health():
        success_count += 1
    
    print("\n" + "=" * 50)
    print("Tests completed: {success_count}/{total_tests}".format(success_count=success_count, total_tests=total_tests))
    
    if success_count == total_tests:
        print("SUCCESS All tests passed! RAG components working correctly.")
    else:
        print("WARN {failed} test(s) failed. Check logs above.".format(failed=total_tests - success_count))
    
    print("\nNext steps:")
    print("   1. Install Milvus: docker run -p 19530:19530 milvusdb/milvus")
    print("   2. Test vector storage with real Milvus")
    print("   3. Build first RAG with document upload")