"""
RAG Progress Tracker Implementation
Real-time progress tracking with Redis support and resume capability
"""

import json
import time
import asyncio
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict

try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

logger = logging.getLogger("rag.progress")


@dataclass
class ProgressUpdate:
    """Progress update data structure"""
    rag_id: str
    stage: str
    percentage: float
    timestamp: str
    status: str = "running"  # running, completed, failed, paused
    current_step: Optional[str] = None
    last_ok_step: Optional[str] = None
    attempt: int = 1
    eta_seconds: Optional[float] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class RAGProgressTracker:
    """
    Enhanced progress tracker with resume capability and multiple backends
    
    Features:
    - Redis backend for real-time updates
    - Memory fallback when Redis unavailable  
    - Resume capability for failed builds
    - WebSocket broadcasting support
    - Progress estimation and ETA calculation
    """
    
    def __init__(self, 
                 redis_url: str = "redis://localhost:6379",
                 ttl_seconds: int = 7200,  # 2 hours
                 enable_persistence: bool = True):
        """
        Initialize progress tracker
        
        Args:
            redis_url: Redis connection URL
            ttl_seconds: TTL for progress data in Redis
            enable_persistence: Whether to persist to database
        """
        self.ttl_seconds = ttl_seconds
        self.enable_persistence = enable_persistence
        
        # Redis setup
        self.redis_client = None
        self.memory_store = {}  # Fallback storage
        
        if REDIS_AVAILABLE:
            try:
                self.redis_client = redis.from_url(redis_url, decode_responses=True)
                # Test connection
                self.redis_client.ping()
                logger.info("Redis connected for progress tracking")
            except Exception as e:
                logger.warning(f"Redis unavailable, using memory store: {e}")
                self.redis_client = None
        else:
            logger.warning("Redis library not available, using memory store")
        
        # WebSocket manager (to be injected)
        self.websocket_manager = None
        
        # Stage definitions for ETA calculation
        self.stage_weights = {
            "uploading": 0.05,
            "parsing": 0.15, 
            "chunking": 0.20,
            "deduplication": 0.10,
            "embedding": 0.35,
            "indexing": 0.15
        }
        
        logger.info("RAG Progress Tracker initialized")
    
    async def start_rag_build(self, rag_id: str, metadata: Dict[str, Any] = None) -> bool:
        """
        Start tracking a new RAG build
        
        Args:
            rag_id: Unique RAG identifier
            metadata: Additional build metadata
            
        Returns:
            Success status
        """
        progress = ProgressUpdate(
            rag_id=rag_id,
            stage="initializing",
            percentage=0.0,
            timestamp=datetime.now().isoformat(),
            status="running",
            current_step="Starting RAG build",
            metadata=metadata or {}
        )
        
        return await self._store_progress(progress)
    
    async def update_progress(self, 
                            rag_id: str, 
                            stage: str, 
                            percentage: float,
                            current_step: str = None,
                            last_ok_step: str = None,
                            status: str = "running",
                            metadata: Dict[str, Any] = None) -> bool:
        """
        Update RAG build progress
        
        Args:
            rag_id: RAG identifier
            stage: Current processing stage
            percentage: Completion percentage (0-100)
            current_step: Current operation description
            last_ok_step: Last successfully completed step (for resume)
            status: Build status
            metadata: Additional metadata
            
        Returns:
            Success status
        """
        # Get existing progress for attempt tracking
        existing = await self.get_progress(rag_id)
        attempt = existing.get("attempt", 1) if existing else 1
        
        # Calculate ETA
        eta_seconds = self._calculate_eta(stage, percentage, existing)
        
        progress = ProgressUpdate(
            rag_id=rag_id,
            stage=stage,
            percentage=percentage,
            timestamp=datetime.now().isoformat(),
            status=status,
            current_step=current_step,
            last_ok_step=last_ok_step or existing.get("last_ok_step") if existing else None,
            attempt=attempt,
            eta_seconds=eta_seconds,
            metadata=metadata or {}
        )
        
        success = await self._store_progress(progress)
        
        # Broadcast to WebSocket clients
        if self.websocket_manager:
            try:
                await self.websocket_manager.broadcast(
                    f"rag_progress:{rag_id}", 
                    asdict(progress)
                )
            except Exception as e:
                logger.debug(f"WebSocket broadcast failed: {e}")
        
        return success
    
    async def mark_stage_complete(self, 
                                rag_id: str, 
                                stage: str,
                                metadata: Dict[str, Any] = None) -> bool:
        """
        Mark a stage as completed and move to next
        
        Args:
            rag_id: RAG identifier
            stage: Completed stage
            metadata: Stage completion metadata
            
        Returns:
            Success status  
        """
        # Calculate stage completion percentage
        stage_pct = self._get_stage_completion_percentage(stage)
        
        return await self.update_progress(
            rag_id=rag_id,
            stage=stage,
            percentage=stage_pct,
            current_step=f"{stage} completed",
            last_ok_step=stage,
            status="running",
            metadata=metadata
        )
    
    async def mark_build_complete(self, 
                                rag_id: str, 
                                success: bool = True,
                                final_metadata: Dict[str, Any] = None) -> bool:
        """
        Mark RAG build as completed (success or failure)
        
        Args:
            rag_id: RAG identifier
            success: Whether build succeeded
            final_metadata: Final build statistics
            
        Returns:
            Success status
        """
        status = "completed" if success else "failed"
        percentage = 100.0 if success else -1
        
        return await self.update_progress(
            rag_id=rag_id,
            stage="finished" if success else "error",
            percentage=percentage,
            current_step="Build completed" if success else "Build failed",
            last_ok_step="finished" if success else None,
            status=status,
            metadata=final_metadata
        )
    
    async def get_progress(self, rag_id: str) -> Optional[Dict[str, Any]]:
        """
        Get current progress for RAG build
        
        Args:
            rag_id: RAG identifier
            
        Returns:
            Progress data or None if not found
        """
        try:
            if self.redis_client:
                progress_json = self.redis_client.get(f"rag_progress:{rag_id}")
                if progress_json:
                    return json.loads(progress_json)
            
            # Fallback to memory store
            return self.memory_store.get(f"rag_progress:{rag_id}")
            
        except Exception as e:
            logger.error(f"Failed to get progress for {rag_id}: {e}")
            return None
    
    async def get_resume_point(self, rag_id: str) -> tuple[Optional[str], int]:
        """
        Get resume point for failed/interrupted builds
        
        Args:
            rag_id: RAG identifier
            
        Returns:
            Tuple of (last_ok_step, attempt_number)
        """
        progress = await self.get_progress(rag_id)
        if progress:
            return progress.get("last_ok_step"), progress.get("attempt", 1)
        return None, 1
    
    async def increment_attempt(self, rag_id: str) -> int:
        """
        Increment attempt counter for retries
        
        Args:
            rag_id: RAG identifier
            
        Returns:
            New attempt number
        """
        progress = await self.get_progress(rag_id)
        if progress:
            new_attempt = progress.get("attempt", 1) + 1
            await self.update_progress(
                rag_id=rag_id,
                stage=progress.get("stage", "retrying"),
                percentage=0.0,
                current_step=f"Retrying build (attempt {new_attempt})",
                status="running",
                metadata={"attempt": new_attempt, "retry": True}
            )
            return new_attempt
        return 1
    
    async def list_active_builds(self) -> List[Dict[str, Any]]:
        """
        List all active RAG builds
        
        Returns:
            List of active build progress data
        """
        active_builds = []
        
        try:
            if self.redis_client:
                # Scan for RAG progress keys
                keys = self.redis_client.keys("rag_progress:*")
                for key in keys:
                    progress_json = self.redis_client.get(key)
                    if progress_json:
                        progress = json.loads(progress_json)
                        if progress.get("status") in ["running", "paused"]:
                            active_builds.append(progress)
            else:
                # Check memory store
                for key, progress in self.memory_store.items():
                    if key.startswith("rag_progress:") and progress.get("status") in ["running", "paused"]:
                        active_builds.append(progress)
        
        except Exception as e:
            logger.error(f"Failed to list active builds: {e}")
        
        return active_builds
    
    async def cleanup_old_progress(self, max_age_hours: int = 24) -> int:
        """
        Clean up old progress records
        
        Args:
            max_age_hours: Maximum age in hours before cleanup
            
        Returns:
            Number of records cleaned up
        """
        cleaned = 0
        cutoff_time = datetime.now() - timedelta(hours=max_age_hours)
        
        try:
            if self.redis_client:
                keys = self.redis_client.keys("rag_progress:*")
                for key in keys:
                    progress_json = self.redis_client.get(key)
                    if progress_json:
                        progress = json.loads(progress_json)
                        timestamp_str = progress.get("timestamp")
                        if timestamp_str:
                            timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
                            if timestamp < cutoff_time:
                                self.redis_client.delete(key)
                                cleaned += 1
            
            # Clean memory store
            to_remove = []
            for key, progress in self.memory_store.items():
                if key.startswith("rag_progress:"):
                    timestamp_str = progress.get("timestamp")
                    if timestamp_str:
                        try:
                            timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
                            if timestamp < cutoff_time:
                                to_remove.append(key)
                                cleaned += 1
                        except ValueError:
                            pass
            
            for key in to_remove:
                del self.memory_store[key]
                
        except Exception as e:
            logger.error(f"Failed to cleanup old progress: {e}")
        
        if cleaned > 0:
            logger.info(f"Cleaned up {cleaned} old progress records")
        
        return cleaned
    
    def _calculate_eta(self, 
                      current_stage: str, 
                      current_percentage: float, 
                      existing_progress: Dict[str, Any] = None) -> Optional[float]:
        """Calculate estimated time to completion"""
        
        if not existing_progress or current_percentage <= 0:
            return None
        
        try:
            # Get start time
            start_timestamp = existing_progress.get("metadata", {}).get("start_time")
            if not start_timestamp:
                return None
            
            start_time = datetime.fromisoformat(start_timestamp)
            elapsed_seconds = (datetime.now() - start_time).total_seconds()
            
            if elapsed_seconds <= 0 or current_percentage <= 0:
                return None
            
            # Calculate progress rate
            progress_rate = current_percentage / elapsed_seconds  # percent per second
            remaining_percentage = 100.0 - current_percentage
            
            if progress_rate > 0:
                eta_seconds = remaining_percentage / progress_rate
                return min(eta_seconds, 3600)  # Cap at 1 hour
            
        except Exception as e:
            logger.debug(f"ETA calculation failed: {e}")
        
        return None
    
    def _get_stage_completion_percentage(self, stage: str) -> float:
        """Get cumulative percentage for stage completion"""
        
        stage_order = ["uploading", "parsing", "chunking", "deduplication", "embedding", "indexing"]
        
        total_pct = 0.0
        for stage_name in stage_order:
            total_pct += self.stage_weights.get(stage_name, 0.1) * 100
            if stage_name == stage:
                break
        
        return min(total_pct, 95.0)  # Leave 5% for finalization
    
    async def _store_progress(self, progress: ProgressUpdate) -> bool:
        """Store progress update in backend"""
        
        key = f"rag_progress:{progress.rag_id}"
        progress_dict = asdict(progress)
        
        try:
            # Store in Redis with TTL
            if self.redis_client:
                self.redis_client.setex(
                    key,
                    self.ttl_seconds,
                    json.dumps(progress_dict)
                )
            
            # Store in memory as fallback
            self.memory_store[key] = progress_dict
            
            # Persist to database if enabled
            if self.enable_persistence:
                await self._persist_progress_to_db(progress)
            
            logger.debug(f"Progress stored for {progress.rag_id}: {progress.stage} {progress.percentage}%")
            return True
            
        except Exception as e:
            logger.error(f"Failed to store progress for {progress.rag_id}: {e}")
            return False
    
    async def _persist_progress_to_db(self, progress: ProgressUpdate):
        """Persist progress to database for auditing (placeholder)"""
        # TODO: Implement database persistence
        # This would typically save to a rag_builds or rag_progress table
        pass


# Global singleton
_progress_tracker = None

def get_progress_tracker(**kwargs) -> RAGProgressTracker:
    """Get global progress tracker instance"""
    global _progress_tracker
    if _progress_tracker is None:
        _progress_tracker = RAGProgressTracker(**kwargs)
    return _progress_tracker