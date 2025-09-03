# RAG Implementation Changelog

## 2025-01-14

### Initial Structure
- **Created:** `RAG_CHANGELOG.md` - Track all RAG implementation changes
- **Created:** `langchain_integration/rag/` - Main RAG module structure
- **Created:** `langchain_integration/rag/__init__.py` - RAG module init v0.1.0
- **Created:** `langchain_integration/rag/embeddings/__init__.py` - Embeddings package
- **Created:** `langchain_integration/rag/storage/__init__.py` - Storage package  
- **Created:** `langchain_integration/rag/processing/__init__.py` - Processing package
- **Created:** `langchain_integration/rag/rerank/__init__.py` - Rerank package

### BGE-M3 Embedding Implementation
- **Created:** `langchain_integration/rag/embeddings/bge_m3_provider.py` - BGE-M3 provider with L2 normalization
- **Created:** `langchain_integration/rag/embeddings/embedding_manager.py` - Singleton manager for cached providers

### Milvus Vector Store Implementation
- **Created:** `langchain_integration/rag/storage/milvus_store.py` - Milvus store with HNSW index and partition support

### Dependencies Installation
- **Installed:** sentence-transformers, scikit-learn, python-multipart - Core RAG dependencies

### API Endpoints
- **Created:** `api/endpoints/rag.py` - RAG endpoints (upload, build, search, delete)
- **Modified:** `api/server.py` - Added conditional RAG router import

### Testing
- **Created:** `test_rag_basic.py` - Basic component testing script
- **Fixed:** Unicode encoding issues in test script for Windows compatibility

### Security Fix
- **Upgraded:** torch from 2.5.1 to 2.8.0 - Resolve CVE-2025-32434 vulnerability