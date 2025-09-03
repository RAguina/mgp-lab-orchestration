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
pip install tiktoken               # Token-aware chunking
pip install numpy scikit-learn     # Evaluation metrics
pip install minio                  # S3-compatible storage (optional)
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
│   └── document_store.py         # File management
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
async def search_rag(rag_id: str, query: str, top_k: int = 5):
    # Vector search + rerank, return passages only

@router.post("/rag/{rag_id}/query")  
async def query_rag(rag_id: str, query: str, model: str = "mistral7b"):
    # Full RAG: search + LLM generation + citations
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
        
        # Token-aware processing
        try:
            import tiktoken
            self.tokenizer = tiktoken.encoding_for_model("gpt-3.5-turbo")
            self.token_aware = True
        except ImportError:
            self.tokenizer = None
            self.token_aware = False
        
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
        schema = CollectionSchema([
            FieldSchema("id", DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema("vector", DataType.FLOAT_VECTOR, dim=1024),  # BGE-M3 dim
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
        
    def search(self, rag_id, query_embedding, top_k=50):
        """Search within specific RAG partition"""  
        collection = Collection(self.collection_name)
        collection.load()
        
        search_params = {"metric_type": "COSINE", "params": {"ef": 96}}
        
        results = collection.search(
            data=[query_embedding],
            anns_field="vector", 
            param=search_params,
            limit=top_k,
            partition_names=[f"rag_{rag_id}"],
            output_fields=["content", "metadata", "quality_score"]
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
        # Implementation: fetch from storage backend
        pass
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
    
    # Save run report
    run_report = {**run_metadata, "metrics": avg_metrics, "detailed": metrics}
    await save_evaluation_report(rag_id, run_report)
    
    return {
        "success": True,
        "metrics": avg_metrics,
        "baseline_met": avg_metrics["recall@10"] >= 0.85,  # GPT-5 threshold
        "report_id": run_report["id"]
    }
```

## Day 5: Frontend Integration & Reranking

### Reranker Integration
```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

class BGEReranker:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-reranker-base")
        self.model = AutoModelForSequenceClassification.from_pretrained("BAAI/bge-reranker-base")
        self.model.eval()
        
    def rerank(self, query, passages, top_k=5):
        """Rerank top-50 to top-5"""
        pairs = [(query, passage["content"]) for passage in passages]
        
        inputs = self.tokenizer(pairs, padding=True, truncation=True, 
                               return_tensors="pt", max_length=512)
        
        with torch.no_grad():
            scores = self.model(**inputs, return_dict=True).logits.view(-1).float()
            
        # Sort by relevance score
        scored_passages = list(zip(passages, scores))
        scored_passages.sort(key=lambda x: x[1], reverse=True)
        
        return [passage for passage, score in scored_passages[:top_k]]
```

### Progress Tracking
```python  
class RAGProgressTracker:
    def __init__(self):
        self.redis_client = redis.Redis() if REDIS_AVAILABLE else None
        
    async def update_progress(self, rag_id, stage, percentage, metadata=None):
        progress = {
            "stage": stage,
            "percentage": percentage,
            "timestamp": datetime.now().isoformat(),
            "metadata": metadata or {}
        }
        
        # Store in Redis
        if self.redis_client:
            self.redis_client.setex(
                f"rag_progress:{rag_id}", 
                3600,  # 1 hour TTL
                json.dumps(progress)
            )
            
        # Broadcast via WebSocket
        await self.websocket_manager.broadcast(
            f"rag_progress:{rag_id}", 
            progress
        )
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
pip install pymilvus==2.3.1  
pip install PyPDF2==3.0.1
pip install python-docx==0.8.11

# Optional but recommended
pip install redis==5.0.1

# Already have
# - torch (for BGE-M3)
# - transformers (for reranker)
# - fastapi (for endpoints)
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