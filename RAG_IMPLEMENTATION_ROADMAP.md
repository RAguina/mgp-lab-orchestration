# RAG Implementation Roadmap - Hybrid Approach

## Overview
Combining our Lab architecture expertise with GPT-5's production-ready RAG insights.

## Day 1-2: Core RAG Infrastructure

### Backend Setup
```bash
# Dependencies  
pip install sentence-transformers  # BGE-M3 embeddings
pip install pymilvus               # Vector database
pip install PyPDF2 python-docx    # Document parsing
pip install redis                 # Optional: progress tracking

# ✅ GPT-5 additions for quality
pip install tiktoken               # Token-aware chunking (deprecated for HF tokenizer)
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
async def build_rag(request: RAGBuildRequest):
    # Async job: ingest → chunk → embed → index
    # Return rag_id, start progress tracking

@router.get("/rag/{rag_id}/status")
async def get_build_status(rag_id: str):
    # Return: stage, percentage, eta, metrics

@router.post("/rag/{rag_id}/search")
async def search_rag(
    rag_id: str, 
    query: str, 
    top_k: int = 5,
    ef_search: int = 96,  # ✅ Tuneable HNSW parameter
    workspace_id: str = Header(None)  # ✅ Multi-tenancy
):
    # Vector search + rerank with tuneable parameters
    # If recall < 0.85, try ef_search=128

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
```

### Milvus Storage
```python
class MilvusRAGStore:
    def __init__(self):
        # Single collection with partitions
        self.collection_name = "ai_lab_chunks"
        self._ensure_collection()
        
    def _ensure_collection(self):
        # ✅ Dynamic dimension from model
        from sentence_transformers import SentenceTransformer
        temp_model = SentenceTransformer("BAAI/bge-m3")
        dim = temp_model.get_sentence_embedding_dimension()
        
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
        
    def search(self, rag_id, query_embedding, top_k=50):
        """Search within specific RAG partition"""  
        collection = Collection(self.collection_name)
        collection.load()
        
        search_params = {"metric_type": "COSINE", "params": {"ef": 96}}
        
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
        
        return self._format_results(results)
        
    def _format_results(self, results):
        """Format search results with content retrieval from storage"""
        formatted = []
        for hits in results:
            for hit in hits:
                # ✅ Retrieve full content from storage URI
                full_content = self._retrieve_content(hit.entity.get("uri"))
                formatted.append({
                    "content": full_content,
                    "excerpt": hit.entity.get("excerpt"), 
                    "metadata": hit.entity.get("metadata"),
                    "quality_score": hit.entity.get("quality_score"),
                    "similarity_score": hit.score
                })
        return formatted
        
    def _retrieve_content(self, uri):
        """Retrieve full chunk content from MinIO/S3"""
        # ✅ Core functionality with MinIO
        if uri.startswith("s3://") or uri.startswith("minio://"):
            return self.minio_client.get_object_content(uri)
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
        expected_docs = query_item["relevant_docs"]
        
        # Search and measure
        results = await search_rag(rag_id, query, top_k=10)
        
        # Calculate metrics
        recall_10 = calculate_recall_at_k(results, expected_docs, k=10)
        ndcg_10 = calculate_ndcg_at_k(results, expected_docs, k=10)
        
        metrics[query] = {
            "recall@10": recall_10,
            "ndcg@10": ndcg_10,
            "latency_ms": results.get("latency_ms", 0)
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
pip install pymilvus==2.3.1  # Or >=2.4 for auto partition_key
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
MILVUS_URI=http://localhost:19530
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