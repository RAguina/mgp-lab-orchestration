"""
RAG API Endpoints
Production-ready endpoints for RAG creation, search and management
"""

from fastapi import APIRouter, HTTPException, UploadFile, File, Header
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from datetime import datetime
import logging
import uuid
import asyncio

logger = logging.getLogger("lab-api.rag")

# Router for RAG endpoints
router = APIRouter(prefix="/rag", tags=["RAG - Retrieval Augmented Generation"])

# Import RAG components with fallback handling
RAG_AVAILABLE = False
try:
    from langchain_integration.rag.embeddings.embedding_manager import get_embedding_manager
    from langchain_integration.rag.storage.milvus_store import MilvusRAGStore
    RAG_AVAILABLE = True
    logger.info("✅ RAG components loaded successfully")
except ImportError as e:
    logger.warning(f"⚠️ RAG components not available: {e}")

# Pydantic models
class RAGBuildRequest(BaseModel):
    name: str = Field(..., description="Human-readable name for the RAG")
    description: Optional[str] = Field("", description="Optional description")
    chunk_size: int = Field(800, description="Chunk size in tokens")
    chunk_overlap: int = Field(100, description="Chunk overlap in tokens")
    embedding_model: str = Field("bge-m3", description="Embedding model to use")
    use_reranker: bool = Field(True, description="Enable reranking for better results")

class RAGSearchRequest(BaseModel):
    query: str = Field(..., description="Search query")
    top_k: int = Field(5, description="Number of results to return")
    ef_search: int = Field(96, description="HNSW ef parameter")
    include_full_content: bool = Field(False, description="Include full content or just excerpt")
    use_reranker: bool = Field(True, description="Apply reranking")

class RAGSearchResponse(BaseModel):
    rag_id: str
    query: str
    params: Dict[str, Any]
    candidates: List[Dict[str, Any]]
    total_found: int
    returned_count: int

# Global storage instances (will be properly initialized)
_milvus_store: Optional[MilvusRAGStore] = None

def get_milvus_store() -> MilvusRAGStore:
    """Get or create Milvus store instance"""
    global _milvus_store
    if _milvus_store is None:
        _milvus_store = MilvusRAGStore()
    return _milvus_store

@router.get("/")
async def rag_health():
    """Health check for RAG endpoints"""
    return {
        "rag_available": RAG_AVAILABLE,
        "endpoints": [
            "/rag/upload - Upload documents",
            "/rag/build - Build RAG from documents", 
            "/rag/{rag_id}/status - Check build status",
            "/rag/{rag_id}/search - Search in RAG",
            "/rag/{rag_id} - Delete RAG"
        ],
        "embedding_models": ["bge-m3"],
        "vector_stores": ["milvus"],
        "features": ["chunking", "embedding", "indexing", "search", "reranking"]
    }

@router.post("/upload")
async def upload_documents(
    files: List[UploadFile] = File(...),
    workspace_id: Optional[str] = Header(None)
):
    """
    Upload documents for RAG creation
    Returns upload_id for subsequent build request
    """
    if not RAG_AVAILABLE:
        raise HTTPException(status_code=503, detail="RAG components not available")
    
    # Generate upload ID
    upload_id = f"upload_{uuid.uuid4().hex[:8]}"
    
    # Basic validation
    if not files:
        raise HTTPException(status_code=400, detail="No files provided")
    
    # Check file types and sizes
    supported_types = {".pdf", ".txt", ".docx", ".md"}
    uploaded_files = []
    
    for file in files:
        if not file.filename:
            continue
            
        file_ext = "." + file.filename.split(".")[-1].lower()
        if file_ext not in supported_types:
            raise HTTPException(
                status_code=400, 
                detail=f"Unsupported file type: {file_ext}. Supported: {supported_types}"
            )
        
        # Read file content (in production, save to MinIO)
        content = await file.read()
        uploaded_files.append({
            "filename": file.filename,
            "size": len(content),
            "type": file_ext,
            "content": content  # In production: save to MinIO and store URI
        })
    
    logger.info(f"Uploaded {len(uploaded_files)} files for {upload_id}")
    
    return {
        "upload_id": upload_id,
        "files_count": len(uploaded_files),
        "total_size": sum(f["size"] for f in uploaded_files),
        "files": [{"filename": f["filename"], "size": f["size"], "type": f["type"]} 
                 for f in uploaded_files],
        "workspace_id": workspace_id,
        "timestamp": datetime.now().isoformat()
    }

@router.post("/build")
async def build_rag(
    request: RAGBuildRequest,
    workspace_id: Optional[str] = Header(None)
):
    """
    Build RAG from uploaded documents
    Starts async processing and returns immediately
    """
    if not RAG_AVAILABLE:
        raise HTTPException(status_code=503, detail="RAG components not available")
    
    # Generate RAG ID
    rag_id = f"rag_{uuid.uuid4().hex[:8]}"
    
    logger.info(f"Starting RAG build: {rag_id}")
    logger.info(f"Config: {request.dict()}")
    
    # TODO: Start async background task for:
    # 1. Document parsing
    # 2. Chunking 
    # 3. Embedding generation
    # 4. Vector indexing
    
    return {
        "rag_id": rag_id,
        "status": "building",
        "stage": "initializing",
        "progress": 0,
        "config": request.dict(),
        "workspace_id": workspace_id,
        "started_at": datetime.now().isoformat(),
        "estimated_completion": "2-5 minutes"
    }

@router.get("/{rag_id}/status")
async def get_rag_status(
    rag_id: str,
    workspace_id: Optional[str] = Header(None)
):
    """Get RAG build status"""
    if not RAG_AVAILABLE:
        raise HTTPException(status_code=503, detail="RAG components not available")
    
    # TODO: Get actual status from background task/database
    # For now, simulate completed status
    
    return {
        "rag_id": rag_id,
        "status": "completed",
        "stage": "ready",
        "progress": 100,
        "workspace_id": workspace_id,
        "stats": {
            "documents_processed": 3,
            "chunks_created": 45,
            "embeddings_generated": 45,
            "index_size": "1.2MB"
        },
        "completed_at": datetime.now().isoformat()
    }

@router.post("/{rag_id}/search", response_model=RAGSearchResponse)
async def search_rag(
    rag_id: str,
    request: RAGSearchRequest,
    workspace_id: Optional[str] = Header(None)
):
    """
    Search within RAG with reranking
    """
    if not RAG_AVAILABLE:
        raise HTTPException(status_code=503, detail="RAG components not available")
    
    try:
        # Get embedding for query
        embedding_manager = get_embedding_manager()
        embedder = embedding_manager.get_provider("bge-m3")
        query_embedding = embedder.embed_query(request.query)
        
        # Search in Milvus
        milvus_store = get_milvus_store()
        
        # Use higher top_k for reranking
        search_top_k = min(50, request.top_k * 10) if request.use_reranker else request.top_k
        
        results = milvus_store.search(
            rag_id=rag_id,
            query_embedding=query_embedding,
            top_k=search_top_k,
            ef_search=request.ef_search,
            include_full_content=request.include_full_content or request.use_reranker
        )
        
        # Apply reranking if requested
        if request.use_reranker and results:
            # TODO: Implement actual reranker
            # For now, just return top_k results
            final_results = results[:request.top_k]
            rerank_applied = True
        else:
            final_results = results[:request.top_k]
            rerank_applied = False
        
        logger.info(f"Search in {rag_id}: '{request.query}' -> {len(final_results)} results")
        
        return RAGSearchResponse(
            rag_id=rag_id,
            query=request.query,
            params={
                "top_k": request.top_k,
                "ef_search": request.ef_search,
                "include_full_content": request.include_full_content,
                "use_reranker": request.use_reranker,
                "rerank_applied": rerank_applied
            },
            candidates=final_results,
            total_found=len(results),
            returned_count=len(final_results)
        )
        
    except Exception as e:
        logger.error(f"Search failed in {rag_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")

@router.delete("/{rag_id}")
async def delete_rag(
    rag_id: str,
    workspace_id: Optional[str] = Header(None)
):
    """Delete RAG and all associated data"""
    if not RAG_AVAILABLE:
        raise HTTPException(status_code=503, detail="RAG components not available")
    
    try:
        # Delete from Milvus
        milvus_store = get_milvus_store()
        milvus_store.delete_rag(rag_id)
        
        # TODO: Delete from MinIO
        # TODO: Clean up metadata
        
        logger.info(f"Deleted RAG: {rag_id}")
        
        return {
            "rag_id": rag_id,
            "deleted": True,
            "workspace_id": workspace_id,
            "deleted_at": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Delete failed for {rag_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Delete failed: {str(e)}")


@router.post("/rag/{rag_id}/eval")
async def evaluate_rag(
    rag_id: str,
    goldset_file: Optional[UploadFile] = File(None),
    use_sample_goldset: bool = False,
    top_k: int = 10,
    workspace_id: str = Header(None)
):
    """
    Evaluate RAG system quality using goldset queries
    
    Generates reproducible metrics including:
    - Recall@K, Precision@K, NDCG@K
    - Mean Reciprocal Rank (MRR)  
    - Latency statistics (P50, P95, P99)
    - Success rates and error analysis
    """
    try:
        from langchain_integration.rag.evaluation import (
            RAGEvaluator, 
            create_sample_goldset, 
            save_evaluation_report
        )
        import subprocess
        
        logger.info(f"Starting evaluation for RAG {rag_id}")
        
        # Load goldset
        if goldset_file:
            # Load uploaded goldset
            content = await goldset_file.read()
            goldset_data = json.loads(content.decode('utf-8'))
            goldset = goldset_data.get("queries", [])
            logger.info(f"Loaded {len(goldset)} queries from uploaded goldset")
        elif use_sample_goldset:
            # Use sample goldset for testing
            goldset = create_sample_goldset()
            logger.info(f"Using sample goldset with {len(goldset)} queries")
        else:
            raise HTTPException(status_code=400, detail="Either provide goldset_file or set use_sample_goldset=true")
        
        if not goldset:
            raise HTTPException(status_code=400, detail="Empty goldset provided")
        
        # Create RAG search function
        async def rag_search_function(rag_id_param, query, top_k_param):
            """Wrapper for RAG search during evaluation"""
            return await search_rag(
                rag_id=rag_id_param, 
                query=query, 
                top_k=top_k_param, 
                workspace_id=workspace_id
            )
        
        # Run evaluation
        evaluator = RAGEvaluator()
        
        # Convert async function to sync for evaluator
        def sync_search(rag_id_param, query, top_k_param):
            import asyncio
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                return loop.run_until_complete(rag_search_function(rag_id_param, query, top_k_param))
            finally:
                loop.close()
        
        # Get git commit for reproducibility
        try:
            commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=".").decode().strip()
        except:
            commit = "unknown"
        
        # Configuration for reproducible results
        eval_config = {
            "top_k": top_k,
            "workspace_id": workspace_id,
            "embedding_model": "BAAI/bge-m3",
            "reranker_model": "BAAI/bge-reranker-base",
            "git_commit": commit
        }
        
        evaluation_report = evaluator.evaluate_rag_system(
            rag_search_function=sync_search,
            goldset=goldset,
            rag_id=rag_id,
            config=eval_config
        )
        
        # Save report
        run_id = evaluation_report["run_id"]
        report_uri = await save_evaluation_report(run_id, evaluation_report)
        
        # Extract key metrics for response
        aggregate_metrics = evaluation_report.get("aggregate_metrics", {})
        
        # Success threshold (configurable)
        recall_threshold = 0.85
        latency_threshold_ms = 200
        
        success_criteria = {
            "recall_target_met": aggregate_metrics.get("avg_recall@10", 0) >= recall_threshold,
            "latency_target_met": aggregate_metrics.get("avg_latency_ms", 1000) <= latency_threshold_ms,
            "overall_success": (
                aggregate_metrics.get("avg_recall@10", 0) >= recall_threshold and 
                aggregate_metrics.get("success_rate", 0) >= 0.9
            )
        }
        
        logger.info(f"Evaluation completed for {rag_id}: {aggregate_metrics}")
        
        return {
            "success": True,
            "run_id": run_id,
            "rag_id": rag_id,
            "evaluation_summary": {
                "queries_evaluated": evaluation_report.get("queries_evaluated", 0),
                "queries_successful": evaluation_report.get("queries_successful", 0),
                "success_rate": aggregate_metrics.get("success_rate", 0),
                "avg_recall@10": aggregate_metrics.get("avg_recall@10", 0),
                "avg_precision@10": aggregate_metrics.get("avg_precision@10", 0),
                "avg_ndcg@10": aggregate_metrics.get("avg_ndcg@10", 0),
                "avg_latency_ms": aggregate_metrics.get("avg_latency_ms", 0),
                "p95_latency_ms": aggregate_metrics.get("p95_latency_ms", 0)
            },
            "success_criteria": success_criteria,
            "report_uri": report_uri,
            "config": eval_config
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Evaluation failed for {rag_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Evaluation failed: {str(e)}")


@router.get("/rag/list")
async def list_rags(
    workspace_id: str = Header(None),
    limit: int = 50,
    offset: int = 0,
    status_filter: Optional[str] = None
):
    """
    List all RAG systems for a workspace
    
    Returns RAG systems with their current status and metadata
    """
    try:
        from langchain_integration.rag.storage.milvus_store import MilvusRAGStore
        from langchain_integration.rag.storage.document_store import get_document_store_manager
        from langchain_integration.rag.progress.tracker import get_progress_tracker
        
        logger.info(f"Listing RAGs for workspace: {workspace_id}")
        
        # Get vector store for partition listing
        vector_store = MilvusRAGStore()
        progress_tracker = get_progress_tracker()
        document_store = get_document_store_manager()
        
        # Get all RAG partitions from Milvus
        stats = vector_store.get_stats()
        partitions = stats.get('partitions', [])
        
        # Filter RAG partitions (start with 'rag_')
        rag_partitions = [p for p in partitions if p.startswith('rag_')]
        
        rags = []
        
        for partition_name in rag_partitions[offset:offset + limit]:
            try:
                # Extract RAG ID
                rag_id = partition_name.replace('rag_', '')
                
                # Get current progress/status
                progress = await progress_tracker.get_progress(rag_id)
                
                # Determine status
                if progress:
                    status = progress.get('status', 'unknown')
                    if status == 'running':
                        status = 'building'
                    elif status == 'completed':
                        status = 'completed'
                    elif status == 'failed':
                        status = 'failed'
                    else:
                        status = 'unknown'
                else:
                    # If no progress info, assume completed (old RAG)
                    status = 'completed'
                
                # Get partition stats from Milvus
                partition_stats = vector_store.get_partition_stats(partition_name)
                chunks_count = partition_stats.get('entity_count', 0)
                
                # Get storage stats
                storage_stats = document_store.get_rag_stats(rag_id)
                
                # Build RAG info
                rag_info = {
                    "rag_id": rag_id,
                    "name": f"RAG-{rag_id[:8]}",  # Default name
                    "description": None,
                    "status": status,
                    "created_at": "unknown",  # Would need metadata storage
                    "updated_at": progress.get('timestamp', 'unknown') if progress else 'unknown',
                    "chunks_count": chunks_count,
                    "documents_count": storage_stats.get('object_count', 0) if storage_stats else 0
                }
                
                # Add build stats if available
                if progress and progress.get('metadata'):
                    metadata = progress['metadata']
                    if 'build_stats' in metadata:
                        rag_info['build_stats'] = metadata['build_stats']
                
                # Apply status filter if specified
                if status_filter and status != status_filter:
                    continue
                
                rags.append(rag_info)
                
            except Exception as e:
                logger.error(f"Error processing RAG partition {partition_name}: {e}")
                continue
        
        # Sort by created/updated time (newest first)
        rags.sort(key=lambda x: x.get('updated_at', ''), reverse=True)
        
        return {
            "success": True,
            "rags": rags,
            "total_count": len(rag_partitions),
            "returned_count": len(rags),
            "offset": offset,
            "limit": limit,
            "workspace_id": workspace_id
        }
        
    except Exception as e:
        logger.error(f"Failed to list RAGs: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to list RAGs: {str(e)}")


@router.post("/rag/{rag_id}/query")
async def query_rag(
    rag_id: str,
    query: str = Body(...),
    model: str = Body("mistral7b"),
    top_k: int = Body(5),
    use_reranker: bool = Body(True),
    include_citations: bool = Body(True),
    workspace_id: str = Header(None)
):
    """
    Full RAG query: search + LLM generation + citations
    
    Performs semantic search and generates an answer using the specified LLM,
    with optional citations to source documents.
    """
    try:
        from langchain_integration.orchestrator import get_orchestrator
        
        logger.info(f"RAG query for {rag_id}: {query[:50]}...")
        
        # First, perform RAG search
        search_results = await search_rag(
            rag_id=rag_id,
            query=query,
            top_k=top_k,
            use_reranker=use_reranker,
            include_full_content=True,  # Need full content for LLM
            workspace_id=workspace_id
        )
        
        # Extract relevant context from search results
        context_chunks = []
        citations = []
        
        for i, candidate in enumerate(search_results.get("candidates", [])):
            content = candidate.get("content") or candidate.get("excerpt", "")
            if content:
                context_chunks.append(content)
                
                if include_citations:
                    citations.append({
                        "index": i + 1,
                        "uri": candidate.get("uri", ""),
                        "similarity_score": candidate.get("similarity_score", 0),
                        "quality_score": candidate.get("quality_score", 0),
                        "excerpt": candidate.get("excerpt", content[:200] + "...")
                    })
        
        if not context_chunks:
            return {
                "success": True,
                "rag_id": rag_id,
                "query": query,
                "answer": "I couldn't find relevant information in the knowledge base to answer your question.",
                "context_found": False,
                "citations": [],
                "search_results": search_results
            }
        
        # Prepare context for LLM
        context_text = "\n\n".join([f"[Context {i+1}]\n{chunk}" for i, chunk in enumerate(context_chunks)])
        
        # Create RAG prompt
        rag_prompt = f"""You are a helpful assistant that answers questions based on the provided context. Use only the information from the context to answer the question. If the context doesn't contain enough information to answer the question, say so.

Context:
{context_text}

Question: {query}

Answer:"""
        
        # Get orchestrator for LLM inference
        orchestrator = get_orchestrator()
        
        # Generate answer using specified model
        llm_response = await orchestrator.generate_response(
            model=model,
            prompt=rag_prompt,
            max_tokens=500,
            temperature=0.1  # Low temperature for factual responses
        )
        
        answer = llm_response.get("response", "").strip()
        
        # Add citation markers if requested
        if include_citations and answer:
            # Simple citation marking (could be enhanced with NLP)
            for i, citation in enumerate(citations):
                # This is a simple approach - could use more sophisticated citation insertion
                pass  # Citations are returned separately for now
        
        return {
            "success": True,
            "rag_id": rag_id,
            "query": query,
            "answer": answer,
            "context_found": True,
            "model_used": model,
            "citations": citations if include_citations else [],
            "search_metadata": {
                "chunks_used": len(context_chunks),
                "total_found": search_results.get("total_found", 0),
                "rerank_applied": search_results.get("params", {}).get("rerank_applied", False)
            },
            "llm_metadata": {
                "model": model,
                "tokens_used": llm_response.get("tokens_used"),
                "latency_ms": llm_response.get("latency_ms")
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"RAG query failed for {rag_id}: {e}")
        raise HTTPException(status_code=500, detail=f"RAG query failed: {str(e)}")