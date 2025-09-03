"""
RAG Evaluation Metrics Implementation
Quality metrics for RAG system performance evaluation
"""

import json
import logging
import time
import uuid
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import statistics

try:
    import numpy as np
    from sklearn.metrics import ndcg_score
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False

logger = logging.getLogger("rag.evaluation")


class RAGEvaluationMetrics:
    """
    Comprehensive RAG evaluation metrics calculator
    
    Metrics included:
    - Recall@K: Relevance recall at various K values
    - NDCG@K: Normalized Discounted Cumulative Gain 
    - Precision@K: Precision at various K values
    - Mean Reciprocal Rank (MRR)
    - Success@K: Binary success rate
    - Latency metrics: P50, P95, P99 response times
    """
    
    @staticmethod
    def calculate_recall_at_k(retrieved_docs: List[Dict[str, Any]], 
                             relevant_docs: List[str], 
                             k: int = 10) -> float:
        """
        Calculate Recall@K metric
        
        Args:
            retrieved_docs: List of retrieved documents with 'uri' or 'doc_id'
            relevant_docs: List of relevant document IDs
            k: Cutoff for evaluation
            
        Returns:
            Recall@K score (0.0 to 1.0)
        """
        if not relevant_docs:
            return 0.0
        
        # Extract document IDs from retrieved results (top K)
        retrieved_ids = []
        for doc in retrieved_docs[:k]:
            doc_id = doc.get('doc_id') or doc.get('uri', '').split('/')[-1]
            if doc_id:
                retrieved_ids.append(doc_id)
        
        # Count relevant documents found
        found_relevant = len(set(retrieved_ids) & set(relevant_docs))
        
        return found_relevant / len(relevant_docs)
    
    @staticmethod
    def calculate_precision_at_k(retrieved_docs: List[Dict[str, Any]], 
                                relevant_docs: List[str], 
                                k: int = 10) -> float:
        """
        Calculate Precision@K metric
        
        Args:
            retrieved_docs: List of retrieved documents
            relevant_docs: List of relevant document IDs  
            k: Cutoff for evaluation
            
        Returns:
            Precision@K score (0.0 to 1.0)
        """
        if not retrieved_docs:
            return 0.0
        
        # Extract document IDs from retrieved results (top K)
        retrieved_ids = []
        for doc in retrieved_docs[:k]:
            doc_id = doc.get('doc_id') or doc.get('uri', '').split('/')[-1]
            if doc_id:
                retrieved_ids.append(doc_id)
        
        if not retrieved_ids:
            return 0.0
        
        # Count relevant documents found
        found_relevant = len(set(retrieved_ids) & set(relevant_docs))
        
        return found_relevant / len(retrieved_ids)
    
    @staticmethod 
    def calculate_ndcg_at_k(retrieved_docs: List[Dict[str, Any]], 
                           relevant_docs: List[str], 
                           k: int = 10) -> float:
        """
        Calculate NDCG@K (Normalized Discounted Cumulative Gain)
        
        Args:
            retrieved_docs: List of retrieved documents with scores
            relevant_docs: List of relevant document IDs
            k: Cutoff for evaluation
            
        Returns:
            NDCG@K score (0.0 to 1.0)
        """
        if not NUMPY_AVAILABLE or not retrieved_docs or not relevant_docs:
            return 0.0
        
        try:
            # Create relevance scores for retrieved docs (top K)
            relevance_scores = []
            for doc in retrieved_docs[:k]:
                doc_id = doc.get('doc_id') or doc.get('uri', '').split('/')[-1]
                # Binary relevance: 1 if relevant, 0 if not
                score = 1.0 if doc_id in relevant_docs else 0.0
                relevance_scores.append(score)
            
            if not any(relevance_scores):
                return 0.0
            
            # Calculate NDCG using sklearn
            # We need to reshape for sklearn format
            relevance_array = np.array([relevance_scores])
            ideal_relevance = np.array([sorted(relevance_scores, reverse=True)])
            
            ndcg = ndcg_score(ideal_relevance, relevance_array, k=k)
            return float(ndcg)
            
        except Exception as e:
            logger.debug(f"NDCG calculation failed: {e}")
            return 0.0
    
    @staticmethod
    def calculate_mrr(retrieved_docs: List[Dict[str, Any]], 
                     relevant_docs: List[str]) -> float:
        """
        Calculate Mean Reciprocal Rank (MRR)
        
        Args:
            retrieved_docs: List of retrieved documents
            relevant_docs: List of relevant document IDs
            
        Returns:
            MRR score (0.0 to 1.0)
        """
        if not retrieved_docs or not relevant_docs:
            return 0.0
        
        for rank, doc in enumerate(retrieved_docs, 1):
            doc_id = doc.get('doc_id') or doc.get('uri', '').split('/')[-1]
            if doc_id in relevant_docs:
                return 1.0 / rank
        
        return 0.0
    
    @staticmethod
    def calculate_success_at_k(retrieved_docs: List[Dict[str, Any]], 
                              relevant_docs: List[str], 
                              k: int = 1) -> float:
        """
        Calculate Success@K (binary success rate)
        
        Args:
            retrieved_docs: List of retrieved documents
            relevant_docs: List of relevant document IDs
            k: Cutoff for evaluation
            
        Returns:
            Success@K score (0.0 or 1.0)
        """
        if not retrieved_docs or not relevant_docs:
            return 0.0
        
        # Check if any of top K results are relevant
        for doc in retrieved_docs[:k]:
            doc_id = doc.get('doc_id') or doc.get('uri', '').split('/')[-1]
            if doc_id in relevant_docs:
                return 1.0
        
        return 0.0


class RAGEvaluator:
    """
    RAG system evaluator with goldset support and reproducible results
    """
    
    def __init__(self):
        self.metrics_calculator = RAGEvaluationMetrics()
        
    def evaluate_rag_system(self, 
                           rag_search_function,
                           goldset: List[Dict[str, Any]],
                           rag_id: str,
                           config: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Evaluate RAG system against goldset
        
        Args:
            rag_search_function: Function that takes (rag_id, query, top_k) and returns results
            goldset: List of evaluation queries with expected results
            rag_id: RAG system identifier
            config: Evaluation configuration
            
        Returns:
            Comprehensive evaluation results
        """
        config = config or {}
        top_k = config.get('top_k', 10)
        
        logger.info(f"Starting RAG evaluation for {rag_id} with {len(goldset)} queries")
        
        # Generate run metadata
        run_id = uuid.uuid4().hex
        start_time = time.time()
        
        run_metadata = {
            "run_id": run_id,
            "rag_id": rag_id,
            "timestamp": datetime.now().isoformat(),
            "goldset_size": len(goldset),
            "config": config
        }
        
        # Evaluate each query
        query_results = []
        latencies = []
        
        for i, query_item in enumerate(goldset):
            query = query_item["query"]
            expected_docs = query_item.get("relevant", [])
            
            logger.debug(f"Evaluating query {i+1}/{len(goldset)}: {query[:50]}...")
            
            # Measure search latency
            search_start = time.perf_counter()
            try:
                results = rag_search_function(rag_id, query, top_k)
                search_latency_ms = (time.perf_counter() - search_start) * 1000
                latencies.append(search_latency_ms)
                
                # Extract candidates from results
                candidates = results.get("candidates", []) if isinstance(results, dict) else results
                
                # Calculate metrics for this query
                query_metrics = self._calculate_query_metrics(candidates, expected_docs, top_k)
                query_metrics.update({
                    "query": query,
                    "query_id": query_item.get("query_id", f"q_{i}"),
                    "latency_ms": search_latency_ms,
                    "results_count": len(candidates),
                    "expected_count": len(expected_docs)
                })
                
                query_results.append(query_metrics)
                
            except Exception as e:
                logger.error(f"Query evaluation failed for: {query[:50]}... Error: {e}")
                # Record failed query
                query_results.append({
                    "query": query,
                    "query_id": query_item.get("query_id", f"q_{i}"),
                    "error": str(e),
                    "latency_ms": 0,
                    "recall@10": 0.0,
                    "precision@10": 0.0,
                    "ndcg@10": 0.0,
                    "mrr": 0.0,
                    "success@1": 0.0
                })
        
        # Aggregate metrics
        total_time = time.time() - start_time
        aggregate_metrics = self._aggregate_metrics(query_results, latencies)
        
        # Generate final report
        evaluation_report = {
            **run_metadata,
            "total_time_seconds": total_time,
            "queries_evaluated": len(query_results),
            "queries_successful": len([q for q in query_results if "error" not in q]),
            "aggregate_metrics": aggregate_metrics,
            "query_results": query_results,
            "latency_stats": self._calculate_latency_stats(latencies)
        }
        
        logger.info(f"RAG evaluation completed: {aggregate_metrics}")
        
        return evaluation_report
    
    def _calculate_query_metrics(self, 
                                candidates: List[Dict[str, Any]], 
                                expected_docs: List[str], 
                                top_k: int) -> Dict[str, float]:
        """Calculate metrics for a single query"""
        
        metrics = {}
        
        # Calculate various metrics
        metrics["recall@1"] = self.metrics_calculator.calculate_recall_at_k(candidates, expected_docs, k=1)
        metrics["recall@5"] = self.metrics_calculator.calculate_recall_at_k(candidates, expected_docs, k=5)
        metrics["recall@10"] = self.metrics_calculator.calculate_recall_at_k(candidates, expected_docs, k=top_k)
        
        metrics["precision@1"] = self.metrics_calculator.calculate_precision_at_k(candidates, expected_docs, k=1)
        metrics["precision@5"] = self.metrics_calculator.calculate_precision_at_k(candidates, expected_docs, k=5)
        metrics["precision@10"] = self.metrics_calculator.calculate_precision_at_k(candidates, expected_docs, k=top_k)
        
        metrics["ndcg@5"] = self.metrics_calculator.calculate_ndcg_at_k(candidates, expected_docs, k=5)
        metrics["ndcg@10"] = self.metrics_calculator.calculate_ndcg_at_k(candidates, expected_docs, k=top_k)
        
        metrics["mrr"] = self.metrics_calculator.calculate_mrr(candidates, expected_docs)
        
        metrics["success@1"] = self.metrics_calculator.calculate_success_at_k(candidates, expected_docs, k=1)
        metrics["success@5"] = self.metrics_calculator.calculate_success_at_k(candidates, expected_docs, k=5)
        
        return metrics
    
    def _aggregate_metrics(self, 
                          query_results: List[Dict[str, Any]], 
                          latencies: List[float]) -> Dict[str, float]:
        """Aggregate metrics across all queries"""
        
        # Filter successful queries
        successful_queries = [q for q in query_results if "error" not in q]
        
        if not successful_queries:
            return {"error": "No successful queries"}
        
        # Calculate averages for each metric
        metric_names = ["recall@1", "recall@5", "recall@10", "precision@1", "precision@5", 
                       "precision@10", "ndcg@5", "ndcg@10", "mrr", "success@1", "success@5"]
        
        aggregated = {}
        
        for metric_name in metric_names:
            values = [q.get(metric_name, 0.0) for q in successful_queries]
            if values:
                aggregated[f"avg_{metric_name}"] = statistics.mean(values)
                aggregated[f"std_{metric_name}"] = statistics.stdev(values) if len(values) > 1 else 0.0
        
        # Overall stats  
        aggregated["success_rate"] = len(successful_queries) / len(query_results)
        
        if latencies:
            aggregated["avg_latency_ms"] = statistics.mean(latencies)
            aggregated["p95_latency_ms"] = np.percentile(latencies, 95) if NUMPY_AVAILABLE else max(latencies)
        
        return aggregated
    
    def _calculate_latency_stats(self, latencies: List[float]) -> Dict[str, float]:
        """Calculate latency statistics"""
        
        if not latencies:
            return {}
        
        stats = {
            "count": len(latencies),
            "mean_ms": statistics.mean(latencies),
            "median_ms": statistics.median(latencies),
            "min_ms": min(latencies),
            "max_ms": max(latencies)
        }
        
        if len(latencies) > 1:
            stats["std_ms"] = statistics.stdev(latencies)
        
        if NUMPY_AVAILABLE:
            stats["p50_ms"] = np.percentile(latencies, 50)
            stats["p95_ms"] = np.percentile(latencies, 95) 
            stats["p99_ms"] = np.percentile(latencies, 99)
        
        return stats


def create_sample_goldset() -> List[Dict[str, Any]]:
    """Create a sample goldset for testing"""
    
    return [
        {
            "query_id": "q1",
            "query": "What is machine learning?",
            "relevant": ["doc_ml_intro.pdf", "doc_ai_basics.pdf"]
        },
        {
            "query_id": "q2", 
            "query": "How do neural networks work?",
            "relevant": ["doc_neural_nets.pdf", "doc_deep_learning.pdf"]
        },
        {
            "query_id": "q3",
            "query": "What are the applications of AI?",
            "relevant": ["doc_ai_applications.pdf", "doc_use_cases.pdf", "doc_industry_ai.pdf"]
        }
    ]


async def save_evaluation_report(run_id: str, report: Dict[str, Any], storage_backend: str = "file") -> str:
    """
    Save evaluation report to storage
    
    Args:
        run_id: Unique run identifier
        report: Evaluation report data
        storage_backend: Storage type ("file", "s3", "database")
        
    Returns:
        Storage URI for the report
    """
    if storage_backend == "file":
        # Save to local file
        filename = f"rag_evaluation_{run_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        filepath = f"evaluations/{filename}"
        
        # Create directory if needed
        import os
        os.makedirs("evaluations", exist_ok=True)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        return f"file://{filepath}"
    
    elif storage_backend == "s3":
        # TODO: Implement S3 storage
        return f"s3://rag-reports/{run_id}/report.json"
    
    elif storage_backend == "database":
        # TODO: Implement database storage
        return f"db://evaluation_reports/{run_id}"
    
    else:
        raise ValueError(f"Unsupported storage backend: {storage_backend}")