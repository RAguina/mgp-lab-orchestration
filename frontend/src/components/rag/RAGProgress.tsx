import React, { useState, useEffect, useCallback } from 'react';
import { Progress, CheckCircle, AlertCircle, Clock, Loader, RefreshCw } from 'lucide-react';

interface ProgressUpdate {
  rag_id: string;
  stage: string;
  percentage: number;
  timestamp: string;
  status: 'running' | 'completed' | 'failed' | 'paused';
  current_step?: string;
  last_ok_step?: string;
  attempt: number;
  eta_seconds?: number;
  metadata?: any;
}

interface RAGProgressProps {
  ragId: string;
  onComplete?: (ragId: string, success: boolean) => void;
  onError?: (error: string) => void;
  workspaceId?: string;
}

const RAGProgress: React.FC<RAGProgressProps> = ({ 
  ragId, 
  onComplete, 
  onError, 
  workspaceId 
}) => {
  const [progress, setProgress] = useState<ProgressUpdate | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [wsConnected, setWsConnected] = useState(false);
  const [lastUpdate, setLastUpdate] = useState<Date | null>(null);
  const [retryCount, setRetryCount] = useState(0);

  // Format time remaining
  const formatETA = useCallback((seconds: number | undefined): string => {
    if (!seconds) return '';
    
    if (seconds < 60) return `${Math.round(seconds)}s`;
    if (seconds < 3600) return `${Math.round(seconds / 60)}m`;
    return `${Math.round(seconds / 3600)}h ${Math.round((seconds % 3600) / 60)}m`;
  }, []);

  // Format timestamp
  const formatTimestamp = useCallback((timestamp: string): string => {
    try {
      return new Date(timestamp).toLocaleTimeString();
    } catch {
      return timestamp;
    }
  }, []);

  // Get progress via HTTP polling (fallback)
  const fetchProgress = useCallback(async () => {
    try {
      const headers: HeadersInit = {};
      if (workspaceId) {
        headers['workspace-id'] = workspaceId;
      }

      const response = await fetch(`/api/rag/${ragId}/status`, { headers });
      
      if (!response.ok) {
        throw new Error(`Failed to fetch progress: ${response.statusText}`);
      }

      const data = await response.json();
      
      if (data.progress) {
        setProgress(data.progress);
        setLastUpdate(new Date());
        
        // Check if completed
        if (data.progress.status === 'completed') {
          onComplete?.(ragId, true);
        } else if (data.progress.status === 'failed') {
          onComplete?.(ragId, false);
        }
      }
      
      setIsLoading(false);
    } catch (error) {
      console.error('Failed to fetch progress:', error);
      onError?.(error instanceof Error ? error.message : 'Failed to fetch progress');
      setIsLoading(false);
    }
  }, [ragId, workspaceId, onComplete, onError]);

  // WebSocket connection for real-time updates
  const connectWebSocket = useCallback(() => {
    try {
      const wsProtocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
      const wsUrl = `${wsProtocol}//${window.location.host}/ws/rag/progress/${ragId}`;
      
      const ws = new WebSocket(wsUrl);
      
      ws.onopen = () => {
        console.log('WebSocket connected for RAG progress');
        setWsConnected(true);
        setRetryCount(0);
      };
      
      ws.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          if (data.type === 'rag_progress' && data.rag_id === ragId) {
            setProgress(data.progress);
            setLastUpdate(new Date());
            
            // Check if completed
            if (data.progress.status === 'completed') {
              onComplete?.(ragId, true);
              ws.close();
            } else if (data.progress.status === 'failed') {
              onComplete?.(ragId, false);
              ws.close();
            }
          }
        } catch (error) {
          console.error('Failed to parse WebSocket message:', error);
        }
      };
      
      ws.onclose = () => {
        setWsConnected(false);
        
        // Retry connection if not completed
        if (progress?.status === 'running' && retryCount < 3) {
          setTimeout(() => {
            setRetryCount(prev => prev + 1);
            connectWebSocket();
          }, 5000 * (retryCount + 1)); // Exponential backoff
        }
      };
      
      ws.onerror = (error) => {
        console.error('WebSocket error:', error);
        setWsConnected(false);
      };
      
      return ws;
    } catch (error) {
      console.error('Failed to connect WebSocket:', error);
      return null;
    }
  }, [ragId, progress?.status, retryCount, onComplete]);

  // Initialize progress tracking
  useEffect(() => {
    // Initial fetch
    fetchProgress();
    
    // Try WebSocket connection
    const ws = connectWebSocket();
    
    // Fallback: Poll if WebSocket fails
    let pollInterval: NodeJS.Timeout;
    
    if (!ws) {
      pollInterval = setInterval(fetchProgress, 5000); // Poll every 5 seconds
    }
    
    return () => {
      if (ws) ws.close();
      if (pollInterval) clearInterval(pollInterval);
    };
  }, [ragId]);

  // Stage display configuration
  const stageConfig = {
    initializing: { label: 'Initializing', color: 'bg-blue-500' },
    uploading: { label: 'Uploading Files', color: 'bg-blue-500' },
    parsing: { label: 'Parsing Documents', color: 'bg-yellow-500' },
    chunking: { label: 'Chunking Text', color: 'bg-orange-500' },
    deduplication: { label: 'Deduplication', color: 'bg-purple-500' },
    embedding: { label: 'Generating Embeddings', color: 'bg-indigo-500' },
    indexing: { label: 'Vector Indexing', color: 'bg-green-500' },
    finished: { label: 'Completed', color: 'bg-green-600' },
    error: { label: 'Failed', color: 'bg-red-500' }
  };

  const currentStageConfig = progress ? stageConfig[progress.stage as keyof typeof stageConfig] || stageConfig.initializing : stageConfig.initializing;

  if (isLoading && !progress) {
    return (
      <div className=\"flex items-center justify-center p-6\">
        <Loader className=\"animate-spin h-6 w-6 text-blue-500 mr-2\" />
        <span>Loading progress...</span>
      </div>
    );
  }

  if (!progress) {
    return (
      <div className=\"p-6 text-center text-gray-500\">
        <AlertCircle className=\"h-12 w-12 mx-auto mb-2 text-gray-400\" />
        <p>No progress data available</p>
        <button 
          onClick={fetchProgress}
          className=\"mt-2 text-blue-500 hover:text-blue-700\"
        >
          <RefreshCw className=\"h-4 w-4 inline mr-1\" />
          Retry
        </button>
      </div>
    );
  }

  return (
    <div className=\"bg-white rounded-lg shadow-lg p-6 max-w-2xl mx-auto\">
      <div className=\"flex items-center justify-between mb-4\">
        <h3 className=\"text-lg font-semibold text-gray-900\">
          RAG Build Progress
        </h3>
        <div className=\"flex items-center text-sm text-gray-500\">
          {wsConnected ? (
            <div className=\"flex items-center\">
              <div className=\"h-2 w-2 bg-green-400 rounded-full mr-1\"></div>
              <span>Live</span>
            </div>
          ) : (
            <div className=\"flex items-center\">
              <RefreshCw className=\"h-3 w-3 mr-1\" />
              <span>Polling</span>
            </div>
          )}
        </div>
      </div>

      {/* RAG ID */}
      <div className=\"mb-4 p-3 bg-gray-50 rounded-md\">
        <div className=\"text-sm text-gray-600\">RAG ID:</div>
        <div className=\"font-mono text-sm\">{ragId}</div>
      </div>

      {/* Current Stage */}
      <div className=\"mb-6\">
        <div className=\"flex items-center justify-between mb-2\">
          <div className=\"flex items-center\">
            {progress.status === 'completed' ? (
              <CheckCircle className=\"h-5 w-5 text-green-500 mr-2\" />
            ) : progress.status === 'failed' ? (
              <AlertCircle className=\"h-5 w-5 text-red-500 mr-2\" />
            ) : (
              <Loader className=\"animate-spin h-5 w-5 text-blue-500 mr-2\" />
            )}
            <span className=\"font-medium text-gray-900\">{currentStageConfig.label}</span>
          </div>
          <div className=\"text-right\">
            <div className=\"text-lg font-bold text-gray-900\">
              {progress.percentage >= 0 ? `${Math.round(progress.percentage)}%` : 'Error'}
            </div>
            {progress.eta_seconds && progress.status === 'running' && (
              <div className=\"text-sm text-gray-500 flex items-center\">
                <Clock className=\"h-3 w-3 mr-1\" />
                {formatETA(progress.eta_seconds)} remaining
              </div>
            )}
          </div>
        </div>

        {/* Progress Bar */}
        <div className=\"w-full bg-gray-200 rounded-full h-3 mb-2\">
          <div
            className={`h-3 rounded-full transition-all duration-300 ${
              progress.status === 'failed' ? 'bg-red-500' : currentStageConfig.color
            }`}
            style={{
              width: `${Math.max(0, Math.min(100, progress.percentage))}%`
            }}
          />
        </div>

        {/* Current Step */}
        {progress.current_step && (
          <div className=\"text-sm text-gray-600 mt-2\">{progress.current_step}</div>
        )}
      </div>

      {/* Status Details */}
      <div className=\"space-y-3\">
        {/* Attempt Info (for retries) */}
        {progress.attempt > 1 && (
          <div className=\"flex items-center text-sm text-orange-600\">
            <RefreshCw className=\"h-4 w-4 mr-1\" />
            <span>Attempt {progress.attempt}</span>
            {progress.last_ok_step && (
              <span className=\"ml-2 text-gray-500\">
                (resumed from: {progress.last_ok_step})
              </span>
            )}
          </div>
        )}

        {/* Metadata */}
        {progress.metadata && Object.keys(progress.metadata).length > 0 && (
          <details className=\"text-sm\">
            <summary className=\"cursor-pointer text-gray-600 hover:text-gray-800\">
              Build Details
            </summary>
            <div className=\"mt-2 p-3 bg-gray-50 rounded text-xs font-mono\">
              {JSON.stringify(progress.metadata, null, 2)}
            </div>
          </details>
        )}

        {/* Last Update */}
        <div className=\"flex items-center justify-between text-xs text-gray-500\">
          <span>Last updated: {formatTimestamp(progress.timestamp)}</span>
          <span>Status: {progress.status}</span>
        </div>
      </div>

      {/* Action Buttons */}
      {progress.status === 'failed' && (
        <div className=\"mt-4 flex space-x-2\">
          <button
            onClick={fetchProgress}
            className=\"bg-blue-500 text-white px-3 py-1 rounded text-sm hover:bg-blue-600\"
          >
            <RefreshCw className=\"h-3 w-3 inline mr-1\" />
            Refresh
          </button>
          {/* Could add retry button here if backend supports it */}
        </div>
      )}
    </div>
  );
};

export default RAGProgress;