# RAG Implementation Roadmap - Hybrid Approach

## Overview
Combining our Lab architecture expertise with GPT-5's production-ready RAG insights.

## Day 1-2: Core RAG Infrastructure

### Backend Setup
```bash
# Dependencies  
pip install sentence-transformers  # BGE-M3 embeddings
pip install pymilvus>=2.4.0        # ✅ Latest for partition_key support
pip install PyPDF2 python-docx    # Document parsing
pip install redis                 # Optional: progress tracking

# ✅ GPT-5 additions for quality
# pip install tiktoken             # ❌ REMOVED - using HuggingFace tokenizer instead
pip install numpy scikit-learn     # Evaluation metrics
pip install minio                  # S3-compatible storage (CORE - not optional)
pip install python-multipart       # FastAPI file uploads
```

### Core Components
```python
# langchain_integration/rag/
├── __init__.py
├── embeddings/
│   ├── bge_m3_provider.py        # BGE-M3 integration
│   └── embedding_manager.py      # Model loading/unloading
├── storage/  
│   ├── milvus_store.py           # Vector storage
│   ├── minio_store.py            # ✅ S3-compatible document storage (CORE)
│   └── document_store.py         # File management coordination
├── processing/
│   ├── document_parser.py        # PDF/DOCX/TXT parsing
│   ├── smart_chunker.py          # GPT-5 style chunking
│   └── deduplicator.py           # Semantic deduplication
└── rerank/
    └── bge_reranker.py           # Reranking integration
```

### API Endpoints  
```python
# api/endpoints/rag.py
@router.post("/rag/upload")
async def upload_documents(files: List[UploadFile], workspace_id: str = Header(None)):
    # Handle file upload, return upload_id

@router.post("/rag/build") 
async def build_rag(request: RAGBuildRequest, workspace_id: str = Header(None)):
    # Async job: ingest → chunk → embed → index
    # Return rag_id, start progress tracking

@router.get("/rag/{rag_id}/status")
async def get_build_status(rag_id: str, workspace_id: str = Header(None)):
    # Return: stage, percentage, eta, metrics

@router.post("/rag/{rag_id}/search")
async def search_rag(
    rag_id: str, 
    query: str, 
    top_k: int = 5,
    ef_search: int = 96,  # ✅ Tuneable HNSW parameter
    include_full_content: bool = False,  # ✅ P95 stable: uri+excerpt by default
    use_reranker: bool = True,  # ✅ Reranker ON by default for quality
    filters: dict | None = None,  # ✅ Future: metadata filtering 
    mmr_lambda: float | None = None,  # ✅ Future: MMR diversification
    workspace_id: str = Header(None)  # ✅ Multi-tenancy
):
    """✅ Vector search + rerank with tuneable parameters"""
    # ✅ C) Efficiency: Use singleton/cached instances (not create every request)
    embedder = get_embedding_manager().get_provider("bge-m3")  # Cached
    query_embedding = embedder.embed_query(query)
    
    # Search with parameters (top-50 for reranking, or top_k if no rerank)
    search_top_k = min(50, top_k * 10) if use_reranker else top_k
    rag_store = get_vector_store_manager().get_store("milvus")  # Cached
    results = rag_store.search(
        rag_id=rag_id,
        query_embedding=query_embedding,
        top_k=search_top_k,
        ef_search=ef_search,
        include_full_content=include_full_content or use_reranker  # Need content for reranker
    )
    
    # ✅ B) Apply reranker as per Golden Path (top-50 → top-5)
    if use_reranker and results:
        reranker = get_reranker_manager().get_reranker("bge-base")  # Cached
        # Prepare input: use content if available, fallback to excerpt
        rerank_input = []
        for r in results:
            content = r.get("content") or r.get("excerpt", "")
            rerank_input.append({**r, "content": content})
            
        reranked = reranker.rerank(
            query, 
            rerank_input, 
            top_k=min(top_k, len(results)), 
            max_passages=min(64, len(results))
        )
        final_results = reranked
    else:
        final_results = results[:top_k]
    
    # ✅ A) Return structured response (was missing!)
    return {
        "rag_id": rag_id,
        "query": query,
        "params": {
            "top_k": top_k,
            "ef_search": ef_search,
            "include_full_content": include_full_content,
            "use_reranker": use_reranker,
            "rerank_applied": use_reranker and len(results) > 0
        },
        "candidates": final_results,
        "total_found": len(results),
        "returned_count": len(final_results)
    }

@router.post("/rag/{rag_id}/query")  
async def query_rag(rag_id: str, query: str, model: str = "mistral7b", workspace_id: str = Header(None)):
    # Full RAG: search + LLM generation + citations

@router.delete("/rag/{rag_id}")
async def delete_rag(rag_id: str, workspace_id: str = Header(None)):
    """✅ GC endpoint: Drop partition + delete MinIO objects (idempotent)"""
    # 1. Verify workspace ownership
    # 2. Drop Milvus partition  
    # 3. Delete MinIO/S3 objects
    # 4. Clean up metadata in DB
    # 5. Return deletion summary
```

## Day 3: Document Processing Pipeline

### Smart Chunking Implementation
```python
class SmartChunker:
    def __init__(self, chunk_size=800, overlap=100, min_size=120):
        # ✅ GPT-5 improvement: Token-aware chunking
        self.chunk_size = chunk_size  # in tokens, not chars
        self.overlap = overlap
        self.min_size = min_size
        
        # ✅ Token-aware processing aligned to embedding model
        try:
            from transformers import AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-m3")  # Not tiktoken
            self.token_aware = True
        except ImportError:
            self.tokenizer = None
            self.token_aware = False
            
    def count_tokens(self, txt: str) -> int:
        """Count tokens using BGE-M3 tokenizer for accurate chunking"""
        if self.token_aware:
            return len(self.tokenizer.encode(txt, add_special_tokens=False))
        else:
            # Fallback: rough estimation
            return len(txt.split()) * 1.3
        
    def chunk_document(self, text, doc_metadata):
        """
        GPT-5 style chunking:
        - Respect paragraph boundaries
        - Preserve heading context
        - Score chunk quality
        - Extract section types
        """
        chunks = []
        
        # 1. Structure detection
        sections = self._detect_sections(text)
        
        # 2. Semantic chunking within sections
        for section in sections:
            section_chunks = self._chunk_section(section)
            chunks.extend(section_chunks)
            
        # 3. Quality scoring
        scored_chunks = self._score_chunks(chunks)
        
        # 4. Deduplication
        unique_chunks = self._deduplicate(scored_chunks)
        
        return unique_chunks
        
    def _detect_sections(self, text):
        # Identify: headers, paragraphs, code blocks, tables, lists
        pass
        
    def _score_chunks(self, chunks):
        # Quality metrics: length, coherence, information density
        pass
```

### Document Parsing
```python  
class DocumentParser:
    def parse_pdf(self, file_path):
        # PyPDF2 + OCR fallback
        pass
        
    def parse_docx(self, file_path):
        # python-docx with structure preservation
        pass
        
    def parse_markdown(self, file_path):
        # Markdown parsing with header hierarchy
        pass
```

## Day 4: Embedding & Vector Storage

### BGE-M3 Integration
```python
class BGEM3EmbeddingProvider:
    def __init__(self, device="cuda"):
        self.model = SentenceTransformer("BAAI/bge-m3", device=device)
        
    def embed_chunks(self, chunks):
        """Generate normalized embeddings"""
        embeddings = self.model.encode(
            [chunk["content"] for chunk in chunks],
            normalize_embeddings=True  # L2 normalization
        )
        return embeddings.tolist()
        
    def embed_query(self, query):
        """Generate normalized query embedding"""  
        embedding = self.model.encode([query], normalize_embeddings=True)
        return embedding[0].tolist()

# ✅ C) Singleton managers for efficiency
class EmbeddingManager:
    """Cached embedding providers to avoid recreating heavy models"""
    _instance = None
    _providers = {}
    
    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
        
    def get_provider(self, model_name: str):
        if model_name not in self._providers:
            if model_name == "bge-m3":
                self._providers[model_name] = BGEM3EmbeddingProvider()
            else:
                raise ValueError(f"Unknown embedding model: {model_name}")
        return self._providers[model_name]

# Global singleton accessors
def get_embedding_manager() -> EmbeddingManager:
    return EmbeddingManager.get_instance()

class VectorStoreManager:
    """Cached vector store instances"""
    _instance = None
    _stores = {}
    
    @classmethod  
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
        
    def get_store(self, store_type: str):
        if store_type not in self._stores:
            if store_type == "milvus":
                # Use default config - could be made configurable
                self._stores[store_type] = MilvusRAGStore(embed_dim=1024)
            else:
                raise ValueError(f"Unknown vector store: {store_type}")
        return self._stores[store_type]

def get_vector_store_manager() -> VectorStoreManager:
    return VectorStoreManager.get_instance()

class RerankerManager:
    """Cached reranker instances"""
    _instance = None
    _rerankers = {}
    
    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
        
    def get_reranker(self, model_name: str):
        if model_name not in self._rerankers:
            if model_name == "bge-base":
                self._rerankers[model_name] = BGEReranker(device="cpu")
            else:
                raise ValueError(f"Unknown reranker: {model_name}")
        return self._rerankers[model_name]

def get_reranker_manager() -> RerankerManager:
    return RerankerManager.get_instance()
```

### Milvus Storage
```python
class MilvusRAGStore:
    def __init__(self, document_store=None, embed_dim=1024):  # ✅ BGE-M3 hardcoded for now
        self.document_store = document_store
        self.embed_dim = embed_dim
        # Single collection with partitions
        self.collection_name = "ai_lab_chunks"
        self._ensure_collection()
        
    def _ensure_collection(self):
        # ✅ Use injected dimension, no model loading here
        dim = self.embed_dim
        
        schema = CollectionSchema([
            FieldSchema("id", DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema("vector", DataType.FLOAT_VECTOR, dim=dim),  # Dynamic BGE-M3 dim
            FieldSchema("rag_id", DataType.VARCHAR, max_length=64),
            FieldSchema("doc_id", DataType.VARCHAR, max_length=64), 
            FieldSchema("chunk_id", DataType.VARCHAR, max_length=64),
            # ✅ GPT-5 improvement: URI + excerpt instead of full content
            FieldSchema("uri", DataType.VARCHAR, max_length=256),      # s3://bucket/chunks/doc_id/chunk_id.json
            FieldSchema("excerpt", DataType.VARCHAR, max_length=512),  # Preview only
            FieldSchema("metadata", DataType.JSON),                    # Rich metadata with versioning
            FieldSchema("quality_score", DataType.FLOAT),
        ], enable_dynamic_field=True)  # ✅ For flexible metadata
        
        # Create collection if not exists
        if not utility.has_collection(self.collection_name):
            collection = Collection(self.collection_name, schema)
            
            # HNSW index as per GPT-5 recommendations
            index_params = {
                "index_type": "HNSW",
                "metric_type": "COSINE", 
                "params": {"M": 32, "efConstruction": 200}
            }
            collection.create_index("vector", index_params)
            
    def store_chunks(self, rag_id, chunks, embeddings):
        """Store chunks in partitioned collection"""
        collection = Collection(self.collection_name)
        
        # Create partition if needed
        partition_name = f"rag_{rag_id}"
        if not collection.has_partition(partition_name):
            collection.create_partition(partition_name)
            
        # Insert data
        data = self._prepare_data(rag_id, chunks, embeddings)
        collection.insert(data, partition_name=partition_name)
        
        # ✅ Flush after large insertions
        collection.flush()
        # collection.compact() if doing bulk deletes
        
    def search(self, rag_id, query_embedding, top_k=50, ef_search=96, include_full_content=False):
        """✅ Search within specific RAG partition with tunable parameters"""  
        collection = Collection(self.collection_name)
        # Load only the specific partition, not entire collection
        
        search_params = {"metric_type": "COSINE", "params": {"ef": ef_search}}
        
        # ✅ Load only needed partition
        collection.load(partition_names=[f"rag_{rag_id}"])
        
        results = collection.search(
            data=[query_embedding],
            anns_field="vector", 
            param=search_params,
            limit=top_k,
            partition_names=[f"rag_{rag_id}"],
            output_fields=["uri", "excerpt", "metadata", "quality_score"]  # ✅ Fixed output fields
        )
        
        return self._format_results(results, include_full=include_full_content)
        
    def _format_results(self, results, include_full: bool = False):
        """✅ Format search results with optional full content"""
        formatted = []
        for hits in results:
            for hit in hits:
                item = {
                    "uri": hit.entity.get("uri"),
                    "excerpt": hit.entity.get("excerpt"),
                    "metadata": hit.entity.get("metadata"),
                    "quality_score": hit.entity.get("quality_score"),
                    "similarity_score": hit.score
                }
                if include_full:
                    item["content"] = self._retrieve_content(item["uri"])
                formatted.append(item)
        return formatted
        
    def _retrieve_content(self, uri):
        """Retrieve full chunk content from MinIO/S3"""
        # ✅ Core functionality with MinIO
        if uri.startswith("s3://") or uri.startswith("minio://"):
            return self.document_store.get_object_content(uri)
        elif uri.startswith("file://"):
            # Local fallback for development
            import json
            file_path = uri.replace("file://", "")
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)["content"]
        else:
            raise ValueError(f"Unsupported URI scheme: {uri}")
```

## Day 5: Frontend Integration & Reranking

### Evaluation Endpoint (GPT-5 Priority)
```python
@router.post("/rag/{rag_id}/eval")
async def evaluate_rag(rag_id: str, goldset_file: UploadFile = None):
    """
    ✅ GPT-5 critical addition: Evaluation from day 1
    Run goldset queries and generate reproducible metrics
    """
    goldset = await load_goldset(goldset_file) if goldset_file else DEFAULT_GOLDSET
    
    metrics = {}
    run_metadata = {
        "rag_id": rag_id,
        "timestamp": datetime.now().isoformat(),
        "embedding_model": "bge-m3",
        "chunking_config": {"size": 800, "overlap": 100},
        "hnsw_params": {"M": 32, "ef": 96},
        "reranker": "bge-reranker-base"
    }
    
    # Run evaluation
    for query_item in goldset:
        query = query_item["query"] 
        expected_docs = query_item["relevant"]  # ✅ Unified schema
        
        # Search and measure latency
        import time
        t0 = time.perf_counter()
        results = await search_rag(rag_id, query, top_k=10)
        search_latency_ms = (time.perf_counter() - t0) * 1000
        
        # Calculate metrics
        recall_10 = calculate_recall_at_k(results, expected_docs, k=10)
        ndcg_10 = calculate_ndcg_at_k(results, expected_docs, k=10)
        
        metrics[query] = {
            "recall@10": recall_10,
            "ndcg@10": ndcg_10,
            "latency_ms": search_latency_ms
        }
    
    # Aggregate metrics
    avg_metrics = {
        "recall@10": np.mean([m["recall@10"] for m in metrics.values()]),
        "ndcg@10": np.mean([m["ndcg@10"] for m in metrics.values()]),
        "avg_latency_ms": np.mean([m["latency_ms"] for m in metrics.values()])
    }
    
    # ✅ Generate proper run ID and save reproducible manifest
    import uuid
    run_id = uuid.uuid4().hex
    
    # Get git commit for reproducibility
    try:
        import subprocess
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()
    except:
        commit = "unknown"
    
    # ✅ Complete reproducible manifest  
    run_report = {
        "run_id": run_id,
        "rag_id": rag_id,
        "timestamp": run_metadata["timestamp"],
        "embed_model": "BAAI/bge-m3",
        "embed_version": "latest",  # Could be model hash
        "chunking": {"size": 800, "overlap": 100, "min_tokens": 120},
        "index": {"type": "HNSW", "M": 32, "efConstruction": 200, "efSearch": 96},
        "retrieval": {"top_k": 10},
        "reranker": {"name": "bge-reranker-base", "top_n": 5},
        "metrics": avg_metrics,
        "detailed_metrics": metrics,
        "commit": commit
    }
    
    # Save to MinIO/S3 for persistence
    await save_evaluation_report(run_id, run_report)
    
    return {
        "success": True,
        "run_id": run_id,
        "metrics": avg_metrics,
        "baseline_met": avg_metrics["recall@10"] >= 0.85,  # GPT-5 threshold
        "report_uri": f"s3://rag-reports/{run_id}/report.json"
    }
```

## Day 5: Frontend Integration & Reranking

### Reranker Integration
```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

class BGEReranker:
    def __init__(self, device="cpu"):
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-reranker-base")
        self.model = AutoModelForSequenceClassification.from_pretrained("BAAI/bge-reranker-base")
        self.model.to(self.device)
        self.model.eval()
        
    def rerank(self, query, passages, top_k=5, max_passages=64, batch_size=16, device="cpu"):
        """
        ✅ Enhanced rerank with batch processing and limits
        Rerank top-N to top-K with batching for efficiency
        """
        # Limit input size
        passages = passages[:max_passages]
        
        if not passages:
            return []
            
        pairs = [(query, passage.get("content", passage.get("excerpt", ""))) 
                for passage in passages]
        
        scores = []
        
        # Process in batches
        with torch.inference_mode():  # ✅ More efficient than no_grad
            for i in range(0, len(pairs), batch_size):
                batch_pairs = pairs[i:i + batch_size]
                
                inputs = self.tokenizer(
                    batch_pairs, 
                    padding=True, 
                    truncation=True,
                    return_tensors="pt", 
                    max_length=512
                ).to(self.device)
                
                batch_scores = self.model(**inputs, return_dict=True).logits.view(-1).float()
                scores.extend(batch_scores.cpu().tolist())
                
                # ✅ Free GPU memory after each batch
                if self.device.startswith("cuda"):
                    torch.cuda.empty_cache()
            
        # Sort by relevance score
        scored_passages = list(zip(passages, scores))
        scored_passages.sort(key=lambda x: x[1], reverse=True)
        
        return [passage for passage, score in scored_passages[:top_k]]
```

### Progress Tracking with Resume Capability
```python  
class RAGProgressTracker:
    def __init__(self):
        self.redis_client = redis.Redis() if REDIS_AVAILABLE else None
        
    async def update_progress(self, rag_id, stage, percentage, metadata=None, last_ok_step=None):
        """✅ Enhanced progress tracking with resume capability"""
        progress = {
            "stage": stage,
            "percentage": percentage,
            "timestamp": datetime.now().isoformat(),
            "last_ok_step": last_ok_step,  # ✅ For resuming failed builds
            "attempt": metadata.get("attempt", 1) if metadata else 1,
            "metadata": metadata or {}
        }
        
        # Store in Redis with longer TTL
        if self.redis_client:
            self.redis_client.setex(
                f"rag_progress:{rag_id}", 
                7200,  # 2 hours TTL for resume capability
                json.dumps(progress)
            )
            
        # Also persist in database for auditing
        await self._persist_progress_to_db(rag_id, progress)
            
        # Broadcast via WebSocket + SSE fallback
        await self.websocket_manager.broadcast(f"rag_progress:{rag_id}", progress)
        
    async def get_resume_point(self, rag_id):
        """Get last successful step for resuming failed builds"""
        if self.redis_client:
            progress_json = self.redis_client.get(f"rag_progress:{rag_id}")
            if progress_json:
                progress = json.loads(progress_json)
                return progress.get("last_ok_step"), progress.get("attempt", 1)
        return None, 1
        
    async def _persist_progress_to_db(self, rag_id, progress):
        """Persist progress to database for auditing"""
        # Implementation: save to rag_runs table
        pass
```

### Frontend RAG Panel  
```typescript
// Frontend components needed:
// - RAGCreator.tsx: Upload + config UI
// - RAGProgress.tsx: Real-time progress tracking  
// - RAGList.tsx: List user's RAGs
// - RAGTester.tsx: Query testing interface

interface RAGCreatorProps {
  onRAGCreated: (ragId: string) => void;
}

const RAGCreator: React.FC<RAGCreatorProps> = ({ onRAGCreated }) => {
  const [files, setFiles] = useState<File[]>([]);
  const [config, setConfig] = useState({
    chunk_size: 800,
    chunk_overlap: 100,
    embedding_model: "bge-m3",
    use_reranker: true
  });
  const [isBuilding, setIsBuilding] = useState(false);
  
  // File upload, progress tracking, WebSocket connection
}
```

## Success Metrics

### Phase 1 Success Criteria:
- [ ] Upload PDF/TXT/DOCX files successfully
- [ ] Smart chunking with quality scores > 0.7
- [ ] BGE-M3 embeddings generated and stored  
- [ ] Basic vector search working
- [ ] Reranking improving top-5 relevance by >30%
- [ ] Real-time progress updates via WebSocket

### Phase 2 Goals:
- [ ] Search latency < 200ms for 10k documents
- [ ] Recall@10 ≥ 0.85 on goldset
- [ ] Support 10+ concurrent RAG builds
- [ ] Workspace isolation working correctly
- [ ] Memory usage stable during concurrent embedding

## Dependencies Summary

```bash
# Core RAG
pip install sentence-transformers==2.2.2
pip install pymilvus>=2.4.0        # ✅ Latest for partition_key & performance
pip install PyPDF2==3.0.1
pip install python-docx==0.8.11
pip install minio==7.1.15           # S3-compatible storage
pip install python-multipart        # FastAPI file uploads

# Evaluation & Quality
pip install numpy scikit-learn       # Metrics calculation

# Optional but recommended  
pip install redis==5.0.1            # Progress tracking & caching

# Already have
# - torch (for BGE-M3)
# - transformers (for reranker) 
# - fastapi (for endpoints)
```

## Environment Configuration

```bash
# .env.example (✅ Added for DX)
MILVUS_URI=localhost:19530              # gRPC/TCP, not HTTP
MINIO_ENDPOINT=localhost:9000
MINIO_ACCESS_KEY=minioadmin
MINIO_SECRET_KEY=minioadmin  
MINIO_BUCKET=rag-storage
RAG_WORKSPACE=default
REDIS_URL=redis://localhost:6379
```

## Workspace Structure

```
langchain_integration/rag/
├── __init__.py
├── embeddings/
│   ├── __init__.py
│   ├── bge_m3_provider.py
│   └── embedding_manager.py
├── storage/
│   ├── __init__.py  
│   ├── milvus_store.py
│   └── document_store.py
├── processing/
│   ├── __init__.py
│   ├── document_parser.py
│   ├── smart_chunker.py
│   └── deduplicator.py
├── rerank/
│   ├── __init__.py
│   └── bge_reranker.py
├── flows/
│   ├── __init__.py
│   └── rag_orchestration_flows.py
└── progress/
    ├── __init__.py
    └── tracker.py

api/endpoints/
└── rag.py

# Frontend (new)
frontend/src/components/rag/
├── RAGCreator.tsx
├── RAGProgress.tsx  
├── RAGList.tsx
└── RAGTester.tsx
```

This roadmap combines our Lab's strengths with GPT-5's production insights for a robust, scalable RAG implementation.