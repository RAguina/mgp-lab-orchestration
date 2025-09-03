"""
RAG (Retrieval-Augmented Generation) Module
Production-ready implementation with BGE-M3 + Milvus + MinIO
"""

__version__ = "0.1.0"

def get_optimal_device(prefer_device: str = "auto") -> str:
    """
    Get optimal device for ML models, avoiding redundant cuda checks
    
    Args:
        prefer_device: "auto", "cuda", or "cpu"
        
    Returns:
        Device string: "cuda" or "cpu"
    """
    if prefer_device == "cpu":
        return "cpu"
    elif prefer_device == "cuda":
        return "cuda"  # Let individual models handle cuda availability
    else:  # "auto"
        try:
            import torch
            return "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            return "cpu"