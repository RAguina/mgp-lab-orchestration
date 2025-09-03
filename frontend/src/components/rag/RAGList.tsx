import React, { useState, useEffect, useCallback } from 'react';
import { 
  Database, 
  Search, 
  Trash2, 
  Eye, 
  Settings, 
  Clock, 
  CheckCircle, 
  AlertCircle,
  Loader,
  RefreshCw,
  BarChart3,
  Plus
} from 'lucide-react';

interface RAGItem {
  rag_id: string;
  name: string;
  description?: string;
  status: 'building' | 'completed' | 'failed' | 'unknown';
  created_at: string;
  updated_at: string;
  documents_count?: number;
  chunks_count?: number;
  build_stats?: {
    processing_time: number;
    embedding_time: number;
    indexing_time: number;
    total_time: number;
    avg_chunk_quality: number;
  };
}

interface RAGListProps {
  onSelectRAG: (ragId: string) => void;
  onCreateNew: () => void;
  workspaceId?: string;
}

const RAGList: React.FC<RAGListProps> = ({ onSelectRAG, onCreateNew, workspaceId }) => {
  const [rags, setRags] = useState<RAGItem[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [selectedRAG, setSelectedRAG] = useState<string | null>(null);
  const [deleteConfirm, setDeleteConfirm] = useState<string | null>(null);
  
  // Fetch RAG list
  const fetchRAGs = useCallback(async () => {
    try {
      setError(null);
      
      const headers: HeadersInit = {};
      if (workspaceId) {
        headers['workspace-id'] = workspaceId;
      }

      // Note: This endpoint doesn't exist yet, but would be useful
      // For now we'll simulate the list based on status checks
      const response = await fetch('/api/rag/list', { headers });
      
      if (response.ok) {
        const data = await response.json();
        setRags(data.rags || []);
      } else {
        // Fallback: empty list for now
        setRags([]);
      }
      
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load RAGs');
      setRags([]); // Fallback to empty list
    } finally {
      setIsLoading(false);
    }
  }, [workspaceId]);

  // Delete RAG
  const deleteRAG = useCallback(async (ragId: string) => {
    try {
      const headers: HeadersInit = {};
      if (workspaceId) {
        headers['workspace-id'] = workspaceId;
      }

      const response = await fetch(`/api/rag/${ragId}`, {
        method: 'DELETE',
        headers
      });

      if (!response.ok) {
        throw new Error(`Delete failed: ${response.statusText}`);
      }

      // Remove from list
      setRags(prev => prev.filter(rag => rag.rag_id !== ragId));
      setDeleteConfirm(null);
      
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Delete failed');
    }
  }, [workspaceId]);

  // Load RAGs on mount
  useEffect(() => {
    fetchRAGs();
  }, [fetchRAGs]);

  const formatDate = (dateString: string): string => {
    try {
      return new Date(dateString).toLocaleString();
    } catch {
      return dateString;
    }
  };

  const formatDuration = (seconds: number): string => {
    if (seconds < 60) return `${seconds.toFixed(1)}s`;
    if (seconds < 3600) return `${(seconds / 60).toFixed(1)}m`;
    return `${(seconds / 3600).toFixed(1)}h`;
  };

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'completed':
        return <CheckCircle className=\"h-5 w-5 text-green-500\" />;
      case 'building':
        return <Loader className=\"h-5 w-5 text-blue-500 animate-spin\" />;
      case 'failed':
        return <AlertCircle className=\"h-5 w-5 text-red-500\" />;
      default:
        return <Clock className=\"h-5 w-5 text-gray-500\" />;
    }
  };

  const getStatusColor = (status: string): string => {
    switch (status) {
      case 'completed':
        return 'bg-green-100 text-green-800';
      case 'building':
        return 'bg-blue-100 text-blue-800';
      case 'failed':
        return 'bg-red-100 text-red-800';
      default:
        return 'bg-gray-100 text-gray-800';
    }
  };

  if (isLoading) {
    return (
      <div className=\"flex items-center justify-center p-12\">
        <Loader className=\"animate-spin h-8 w-8 text-blue-500 mr-3\" />
        <span className=\"text-gray-600\">Loading RAG systems...</span>
      </div>
    );
  }

  return (
    <div className=\"max-w-6xl mx-auto p-6\">
      {/* Header */}
      <div className=\"flex items-center justify-between mb-6\">
        <div>
          <h1 className=\"text-3xl font-bold text-gray-900 flex items-center\">
            <Database className=\"h-8 w-8 mr-3\" />
            RAG Systems
          </h1>
          <p className=\"text-gray-600 mt-1\">Manage your Retrieval-Augmented Generation systems</p>
        </div>
        
        <div className=\"flex space-x-3\">
          <button
            onClick={fetchRAGs}
            className=\"bg-gray-500 text-white px-4 py-2 rounded-md hover:bg-gray-600 flex items-center\"
          >
            <RefreshCw className=\"h-4 w-4 mr-2\" />
            Refresh
          </button>
          <button
            onClick={onCreateNew}
            className=\"bg-blue-500 text-white px-4 py-2 rounded-md hover:bg-blue-600 flex items-center\"
          >
            <Plus className=\"h-4 w-4 mr-2\" />
            Create New RAG
          </button>
        </div>
      </div>

      {/* Error Message */}
      {error && (
        <div className=\"mb-6 p-4 bg-red-50 border border-red-200 rounded-md flex items-center\">
          <AlertCircle className=\"h-5 w-5 text-red-500 mr-2\" />
          <span className=\"text-red-700\">{error}</span>
          <button
            onClick={() => setError(null)}
            className=\"ml-auto text-red-500 hover:text-red-700\"
          >
            ×
          </button>
        </div>
      )}

      {/* RAG List */}
      {rags.length === 0 ? (
        <div className=\"text-center py-12\">
          <Database className=\"h-16 w-16 text-gray-300 mx-auto mb-4\" />
          <h3 className=\"text-lg font-medium text-gray-900 mb-2\">No RAG systems found</h3>
          <p className=\"text-gray-500 mb-4\">Create your first RAG system to get started</p>
          <button
            onClick={onCreateNew}
            className=\"bg-blue-500 text-white px-6 py-3 rounded-md hover:bg-blue-600 flex items-center mx-auto\"
          >
            <Plus className=\"h-5 w-5 mr-2\" />
            Create Your First RAG
          </button>
        </div>
      ) : (
        <div className=\"grid grid-cols-1 lg:grid-cols-2 xl:grid-cols-3 gap-6\">
          {rags.map((rag) => (
            <div
              key={rag.rag_id}
              className={`bg-white rounded-lg shadow-md border-2 transition-all duration-200 hover:shadow-lg ${ 
                selectedRAG === rag.rag_id ? 'border-blue-500' : 'border-gray-200 hover:border-gray-300'
              }`}
            >
              {/* RAG Card Header */}
              <div className=\"p-4 border-b border-gray-100\">
                <div className=\"flex items-start justify-between mb-2\">
                  <div className=\"flex-1 min-w-0\">
                    <h3 className=\"text-lg font-semibold text-gray-900 truncate\">
                      {rag.name || `RAG-${rag.rag_id.slice(-8)}`}
                    </h3>
                    {rag.description && (
                      <p className=\"text-sm text-gray-600 mt-1 truncate\">{rag.description}</p>
                    )}
                  </div>
                  <div className=\"flex items-center ml-2\">
                    {getStatusIcon(rag.status)}
                  </div>
                </div>
                
                <div className=\"flex items-center space-x-2\">
                  <span className={`px-2 py-1 text-xs font-medium rounded-full ${getStatusColor(rag.status)}`}>
                    {rag.status}
                  </span>
                  <span className=\"text-xs text-gray-500 font-mono\">
                    {rag.rag_id.slice(-8)}
                  </span>
                </div>
              </div>

              {/* RAG Card Body */}
              <div className=\"p-4\">
                {/* Stats */}
                {rag.build_stats && (
                  <div className=\"grid grid-cols-2 gap-3 mb-4 text-sm\">
                    <div>
                      <div className=\"text-gray-500\">Documents</div>
                      <div className=\"font-semibold\">{rag.documents_count || 0}</div>
                    </div>
                    <div>
                      <div className=\"text-gray-500\">Chunks</div>
                      <div className=\"font-semibold\">{rag.chunks_count || 0}</div>
                    </div>
                    <div>
                      <div className=\"text-gray-500\">Quality</div>
                      <div className=\"font-semibold\">
                        {(rag.build_stats.avg_chunk_quality * 100).toFixed(1)}%
                      </div>
                    </div>
                    <div>
                      <div className=\"text-gray-500\">Build Time</div>
                      <div className=\"font-semibold\">
                        {formatDuration(rag.build_stats.total_time)}
                      </div>
                    </div>
                  </div>
                )}

                {/* Timestamps */}
                <div className=\"text-xs text-gray-500 space-y-1 mb-4\">
                  <div>Created: {formatDate(rag.created_at)}</div>
                  {rag.updated_at !== rag.created_at && (
                    <div>Updated: {formatDate(rag.updated_at)}</div>
                  )}
                </div>
              </div>

              {/* RAG Card Actions */}
              <div className=\"p-4 pt-0 flex space-x-2\">
                <button
                  onClick={() => {
                    setSelectedRAG(rag.rag_id);
                    onSelectRAG(rag.rag_id);
                  }}
                  disabled={rag.status === 'building'}
                  className=\"flex-1 bg-blue-500 text-white py-2 px-3 rounded text-sm hover:bg-blue-600 disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center\"
                >
                  <Search className=\"h-3 w-3 mr-1\" />
                  Test
                </button>
                
                <button
                  onClick={() => {/* TODO: Open evaluation modal */}}
                  disabled={rag.status !== 'completed'}
                  className=\"bg-purple-500 text-white py-2 px-3 rounded text-sm hover:bg-purple-600 disabled:opacity-50 disabled:cursor-not-allowed flex items-center\"
                  title=\"Run Evaluation\"
                >
                  <BarChart3 className=\"h-3 w-3\" />
                </button>
                
                <button
                  onClick={() => {/* TODO: Open settings modal */}}
                  className=\"bg-gray-500 text-white py-2 px-3 rounded text-sm hover:bg-gray-600 flex items-center\"
                  title=\"Settings\"
                >
                  <Settings className=\"h-3 w-3\" />
                </button>
                
                <button
                  onClick={() => setDeleteConfirm(rag.rag_id)}
                  className=\"bg-red-500 text-white py-2 px-3 rounded text-sm hover:bg-red-600 flex items-center\"
                  title=\"Delete RAG\"
                >
                  <Trash2 className=\"h-3 w-3\" />
                </button>
              </div>
            </div>
          ))}
        </div>
      )}

      {/* Delete Confirmation Modal */}
      {deleteConfirm && (
        <div className=\"fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center p-4 z-50\">
          <div className=\"bg-white rounded-lg p-6 max-w-md w-full\">
            <div className=\"flex items-center mb-4\">
              <AlertCircle className=\"h-6 w-6 text-red-500 mr-2\" />
              <h3 className=\"text-lg font-semibold text-gray-900\">Delete RAG System</h3>
            </div>
            
            <p className=\"text-gray-600 mb-4\">
              Are you sure you want to delete this RAG system? This action cannot be undone and will permanently remove:
            </p>
            
            <ul className=\"text-sm text-gray-600 mb-6 list-disc list-inside space-y-1\">
              <li>All document chunks and embeddings</li>
              <li>Vector index data</li>
              <li>Search history and metadata</li>
            </ul>
            
            <div className=\"text-sm text-gray-500 mb-4 font-mono bg-gray-100 p-2 rounded\">
              RAG ID: {deleteConfirm}
            </div>
            
            <div className=\"flex space-x-3 justify-end\">
              <button
                onClick={() => setDeleteConfirm(null)}
                className=\"bg-gray-500 text-white px-4 py-2 rounded hover:bg-gray-600\"
              >
                Cancel
              </button>
              <button
                onClick={() => deleteRAG(deleteConfirm)}
                className=\"bg-red-500 text-white px-4 py-2 rounded hover:bg-red-600\"
              >
                Delete Permanently
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default RAGList;