"""
Mock RAG implementation para testing sin dependencies reales
"""

from typing import List, Dict, Any
import json
from datetime import datetime

class MockEmbeddingProvider:
    """Mock embedding provider que simula BGE-M3 y otros"""
    
    def __init__(self, model_name: str = "bge-m3"):
        self.model_name = model_name
        
        # Mock documents para testing
        self.mock_documents = [
            {
                "id": "doc1",
                "content": "Machine Learning es una rama de la inteligencia artificial que permite a las computadoras aprender patrones de datos sin ser programadas explícitamente.",
                "metadata": {"source": "ml_basics.txt", "score": 0.95}
            },
            {
                "id": "doc2", 
                "content": "Los modelos de embeddings como BGE-M3 convierten texto en vectores numéricos que capturan el significado semántico del contenido.",
                "metadata": {"source": "embeddings_guide.txt", "score": 0.87}
            },
            {
                "id": "doc3",
                "content": "RAG (Retrieval-Augmented Generation) combina recuperación de información con generación de texto para dar respuestas más precisas y actualizadas.",
                "metadata": {"source": "rag_overview.txt", "score": 0.92}
            },
            {
                "id": "doc4",
                "content": "Milvus es una base de datos vectorial de código abierto diseñada para aplicaciones de AI que requieren búsqueda de similitud en vectores de alta dimensión.",
                "metadata": {"source": "vector_db_comparison.txt", "score": 0.89}
            }
        ]
    
    def embed_query(self, query: str) -> List[float]:
        """Simula embedding de query (vector falso pero consistente)"""
        # Mock vector basado en hash del query para consistencia
        hash_val = hash(query) % 1000
        return [float(hash_val + i) / 1000 for i in range(384)]  # Simula 384-dim vector
    
    def embed_documents(self, docs: List[str]) -> List[List[float]]:
        """Simula embedding de documentos"""
        return [self.embed_query(doc) for doc in docs]

class MockVectorStore:
    """Mock vector store que simula Milvus/Weaviate/Pinecone"""
    
    def __init__(self, store_type: str = "milvus"):
        self.store_type = store_type
        self.embedding_provider = MockEmbeddingProvider()
        
        # Pre-indexar mock documents
        self.indexed_docs = []
        for doc in self.embedding_provider.mock_documents:
            self.indexed_docs.append({
                **doc,
                "vector": self.embedding_provider.embed_query(doc["content"])
            })
    
    def similarity_search(self, query: str, top_k: int = 3, threshold: float = 0.7) -> List[Dict[str, Any]]:
        """Simula búsqueda por similitud"""
        query_vector = self.embedding_provider.embed_query(query)
        
        # Mock scoring: usar palabras clave simples para simular relevancia
        results = []
        query_lower = query.lower()
        
        for doc in self.indexed_docs:
            # Score mock basado en palabras clave
            score = 0.5  # Base score
            if "machine learning" in query_lower and "machine learning" in doc["content"].lower():
                score += 0.4
            if "embedding" in query_lower and "embedding" in doc["content"].lower():
                score += 0.4
            if "rag" in query_lower and "rag" in doc["content"].lower():
                score += 0.4
            if "milvus" in query_lower and "milvus" in doc["content"].lower():
                score += 0.4
            
            if score >= threshold:
                results.append({
                    "content": doc["content"],
                    "metadata": {**doc["metadata"], "similarity_score": score},
                    "id": doc["id"]
                })
        
        # Ordenar por score y devolver top_k
        results.sort(key=lambda x: x["metadata"]["similarity_score"], reverse=True)
        return results[:top_k]

def mock_rag_retrieval(query: str, embedding_model: str = "bge-m3", 
                      vector_store: str = "milvus", rag_config: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Función principal para simular RAG completo
    """
    if rag_config is None:
        rag_config = {}
    
    top_k = rag_config.get("top_k", 3)
    threshold = rag_config.get("similarity_threshold", 0.7)
    
    # Inicializar providers mock
    embedding_provider = MockEmbeddingProvider(embedding_model)
    vector_store_client = MockVectorStore(vector_store)
    
    # Realizar búsqueda
    retrieval_results = vector_store_client.similarity_search(query, top_k, threshold)
    
    # Construir contexto para el LLM
    context_parts = []
    for i, result in enumerate(retrieval_results, 1):
        context_parts.append(f"[Documento {i}]")
        context_parts.append(f"Fuente: {result['metadata']['source']}")
        context_parts.append(f"Contenido: {result['content']}")
        context_parts.append("")
    
    rag_context = "\n".join(context_parts)
    
    # Construir prompt enriquecido
    enriched_prompt = f"""Basándote en los siguientes documentos recuperados, responde la pregunta del usuario:

{rag_context}

Pregunta del usuario: {query}

Proporciona una respuesta precisa basada en la información encontrada. Si la información no es suficiente, indícalo claramente."""

    return {
        "enriched_prompt": enriched_prompt,
        "retrieved_documents": retrieval_results,
        "metadata": {
            "embedding_model": embedding_model,
            "vector_store": vector_store,
            "documents_found": len(retrieval_results),
            "query": query,
            "timestamp": datetime.now().isoformat(),
            "config": rag_config
        }
    }

# Test function
if __name__ == "__main__":
    # Test mock RAG
    test_query = "¿Qué es machine learning?"
    result = mock_rag_retrieval(
        query=test_query,
        embedding_model="bge-m3",
        vector_store="milvus", 
        rag_config={"top_k": 2, "similarity_threshold": 0.8}
    )
    
    print("=== MOCK RAG TEST ===")
    print(f"Query: {test_query}")
    print(f"Documents found: {result['metadata']['documents_found']}")
    print("\nEnriched prompt:")
    print(result['enriched_prompt'][:500] + "...")