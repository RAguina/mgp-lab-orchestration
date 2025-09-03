import React, { useState, useCallback, useRef } from 'react';
import { Upload, Settings, Loader, CheckCircle, AlertCircle } from 'lucide-react';

interface RAGCreatorProps {
  onRAGCreated: (ragId: string) => void;
  workspaceId?: string;
}

interface RAGConfig {
  chunk_size: number;
  chunk_overlap: number;
  min_chunk_size: number;
  min_quality_score: number;
  enable_deduplication: boolean;
  embedding_model: string;
  embedding_device: string;
  batch_size: number;
  collection_name: string;
  use_reranker: boolean;
}

const defaultConfig: RAGConfig = {
  chunk_size: 800,
  chunk_overlap: 100,
  min_chunk_size: 120,
  min_quality_score: 0.3,
  enable_deduplication: true,
  embedding_model: 'bge-m3',
  embedding_device: 'cuda',
  batch_size: 32,
  collection_name: 'ai_lab_chunks',
  use_reranker: true
};

const RAGCreator: React.FC<RAGCreatorProps> = ({ onRAGCreated, workspaceId }) => {
  const [files, setFiles] = useState<File[]>([]);
  const [config, setConfig] = useState<RAGConfig>(defaultConfig);
  const [ragName, setRagName] = useState('');
  const [ragDescription, setRagDescription] = useState('');
  const [isUploading, setIsUploading] = useState(false);
  const [isBuilding, setIsBuilding] = useState(false);
  const [uploadResult, setUploadResult] = useState<any>(null);
  const [buildResult, setBuildResult] = useState<any>(null);
  const [error, setError] = useState<string | null>(null);
  const [showAdvancedConfig, setShowAdvancedConfig] = useState(false);
  
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleFileChange = useCallback((event: React.ChangeEvent<HTMLInputElement>) => {
    const selectedFiles = Array.from(event.target.files || []);
    const validFiles = selectedFiles.filter(file => {
      const validTypes = ['.pdf', '.docx', '.txt', '.md'];
      const fileExtension = '.' + file.name.split('.').pop()?.toLowerCase();
      return validTypes.includes(fileExtension);
    });

    if (validFiles.length !== selectedFiles.length) {
      setError('Some files were skipped. Only PDF, DOCX, TXT, and MD files are supported.');
    }

    setFiles(prev => [...prev, ...validFiles]);
  }, []);

  const removeFile = useCallback((index: number) => {
    setFiles(prev => prev.filter((_, i) => i !== index));
  }, []);

  const formatFileSize = (bytes: number): string => {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  };

  const handleUpload = async () => {
    if (!files.length) {
      setError('Please select at least one file');
      return;
    }

    if (!ragName.trim()) {
      setError('Please enter a RAG name');
      return;
    }

    setIsUploading(true);
    setError(null);

    try {
      const formData = new FormData();
      files.forEach(file => {
        formData.append('files', file);
      });
      formData.append('rag_name', ragName);
      formData.append('rag_description', ragDescription);

      const headers: HeadersInit = {};
      if (workspaceId) {
        headers['workspace-id'] = workspaceId;
      }

      const response = await fetch('/api/rag/upload', {
        method: 'POST',
        body: formData,
        headers
      });

      if (!response.ok) {
        throw new Error(`Upload failed: ${response.statusText}`);
      }

      const result = await response.json();
      setUploadResult(result);
      console.log('Upload successful:', result);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Upload failed');
    } finally {
      setIsUploading(false);
    }
  };

  const handleBuildRAG = async () => {
    if (!uploadResult?.upload_id) {
      setError('No upload found. Please upload files first.');
      return;
    }

    setIsBuilding(true);
    setError(null);

    try {
      const buildRequest = {
        upload_id: uploadResult.upload_id,
        rag_name: ragName,
        rag_description: ragDescription,
        config: config
      };

      const headers: HeadersInit = {
        'Content-Type': 'application/json'
      };
      if (workspaceId) {
        headers['workspace-id'] = workspaceId;
      }

      const response = await fetch('/api/rag/build', {
        method: 'POST',
        headers,
        body: JSON.stringify(buildRequest)
      });

      if (!response.ok) {
        throw new Error(`Build failed: ${response.statusText}`);
      }

      const result = await response.json();
      setBuildResult(result);
      
      if (result.rag_id) {
        onRAGCreated(result.rag_id);
      }
      
      console.log('RAG build started:', result);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Build failed');
    } finally {
      setIsBuilding(false);
    }
  };

  const resetForm = () => {
    setFiles([]);
    setRagName('');
    setRagDescription('');
    setUploadResult(null);
    setBuildResult(null);
    setError(null);
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  return (
    <div className=\"max-w-4xl mx-auto p-6 bg-white rounded-lg shadow-lg\">
      <h2 className=\"text-2xl font-bold text-gray-900 mb-6\">Create RAG System</h2>

      {error && (
        <div className=\"mb-4 p-4 bg-red-50 border border-red-200 rounded-md flex items-center\">
          <AlertCircle className=\"h-5 w-5 text-red-500 mr-2\" />
          <span className=\"text-red-700\">{error}</span>
        </div>
      )}

      {/* Step 1: File Upload */}
      <div className=\"mb-8\">
        <h3 className=\"text-lg font-semibold mb-4 flex items-center\">
          <span className=\"bg-blue-500 text-white rounded-full w-6 h-6 flex items-center justify-center text-sm mr-2\">1</span>
          Upload Documents
        </h3>

        <div className=\"space-y-4\">
          <div className=\"grid grid-cols-1 md:grid-cols-2 gap-4\">
            <div>
              <label className=\"block text-sm font-medium text-gray-700 mb-1\">
                RAG Name *
              </label>
              <input
                type=\"text\"
                value={ragName}
                onChange={(e) => setRagName(e.target.value)}
                className=\"w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500\"
                placeholder=\"My Knowledge Base\"
              />
            </div>
            <div>
              <label className=\"block text-sm font-medium text-gray-700 mb-1\">
                Description (Optional)
              </label>
              <input
                type=\"text\"
                value={ragDescription}
                onChange={(e) => setRagDescription(e.target.value)}
                className=\"w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500\"
                placeholder=\"Brief description of the RAG system\"
              />
            </div>
          </div>

          <div
            className=\"border-2 border-dashed border-gray-300 rounded-lg p-8 text-center cursor-pointer hover:border-blue-400\"
            onClick={() => fileInputRef.current?.click()}
          >
            <Upload className=\"mx-auto h-12 w-12 text-gray-400 mb-4\" />
            <p className=\"text-gray-600 mb-2\">Click to select files or drag and drop</p>
            <p className=\"text-sm text-gray-500\">PDF, DOCX, TXT, MD files supported</p>
            <input
              ref={fileInputRef}
              type=\"file\"
              multiple
              accept=\".pdf,.docx,.txt,.md\"
              onChange={handleFileChange}
              className=\"hidden\"
            />
          </div>

          {files.length > 0 && (
            <div className=\"space-y-2\">
              <h4 className=\"font-medium text-gray-700\">Selected Files ({files.length}):</h4>
              {files.map((file, index) => (
                <div key={index} className=\"flex items-center justify-between bg-gray-50 p-3 rounded-md\">
                  <div>
                    <span className=\"font-medium\">{file.name}</span>
                    <span className=\"ml-2 text-sm text-gray-500\">({formatFileSize(file.size)})</span>
                  </div>
                  <button
                    onClick={() => removeFile(index)}
                    className=\"text-red-500 hover:text-red-700\"
                  >
                    ×
                  </button>
                </div>
              ))}
            </div>
          )}

          <button
            onClick={handleUpload}
            disabled={isUploading || files.length === 0 || !ragName.trim()}
            className=\"bg-blue-500 text-white px-6 py-2 rounded-md hover:bg-blue-600 disabled:opacity-50 disabled:cursor-not-allowed flex items-center\"
          >
            {isUploading ? (
              <>
                <Loader className=\"animate-spin h-4 w-4 mr-2\" />
                Uploading...
              </>
            ) : (
              <>
                <Upload className=\"h-4 w-4 mr-2\" />
                Upload Files
              </>
            )}
          </button>

          {uploadResult && (
            <div className=\"p-4 bg-green-50 border border-green-200 rounded-md flex items-center\">
              <CheckCircle className=\"h-5 w-5 text-green-500 mr-2\" />
              <span className=\"text-green-700\">
                Files uploaded successfully! {uploadResult.files_processed} files processed.
              </span>
            </div>
          )}
        </div>
      </div>

      {/* Step 2: Configuration */}
      <div className=\"mb-8\">
        <h3 className=\"text-lg font-semibold mb-4 flex items-center\">
          <span className=\"bg-blue-500 text-white rounded-full w-6 h-6 flex items-center justify-center text-sm mr-2\">2</span>
          Configuration
        </h3>

        <div className=\"space-y-4\">
          <div className=\"flex items-center justify-between\">
            <label className=\"flex items-center\">
              <input
                type=\"checkbox\"
                checked={showAdvancedConfig}
                onChange={(e) => setShowAdvancedConfig(e.target.checked)}
                className=\"mr-2\"
              />
              <span className=\"text-sm font-medium text-gray-700\">Show advanced configuration</span>
            </label>
          </div>

          {showAdvancedConfig ? (
            <div className=\"grid grid-cols-1 md:grid-cols-2 gap-4 p-4 bg-gray-50 rounded-md\">
              <div>
                <label className=\"block text-sm font-medium text-gray-700 mb-1\">Chunk Size</label>
                <input
                  type=\"number\"
                  value={config.chunk_size}
                  onChange={(e) => setConfig({...config, chunk_size: parseInt(e.target.value)})}
                  className=\"w-full px-3 py-2 border border-gray-300 rounded-md\"
                />
              </div>
              <div>
                <label className=\"block text-sm font-medium text-gray-700 mb-1\">Chunk Overlap</label>
                <input
                  type=\"number\"
                  value={config.chunk_overlap}
                  onChange={(e) => setConfig({...config, chunk_overlap: parseInt(e.target.value)})}
                  className=\"w-full px-3 py-2 border border-gray-300 rounded-md\"
                />
              </div>
              <div>
                <label className=\"block text-sm font-medium text-gray-700 mb-1\">Min Quality Score</label>
                <input
                  type=\"number\"
                  step=\"0.1\"
                  min=\"0\"
                  max=\"1\"
                  value={config.min_quality_score}
                  onChange={(e) => setConfig({...config, min_quality_score: parseFloat(e.target.value)})}
                  className=\"w-full px-3 py-2 border border-gray-300 rounded-md\"
                />
              </div>
              <div>
                <label className=\"block text-sm font-medium text-gray-700 mb-1\">Batch Size</label>
                <input
                  type=\"number\"
                  value={config.batch_size}
                  onChange={(e) => setConfig({...config, batch_size: parseInt(e.target.value)})}
                  className=\"w-full px-3 py-2 border border-gray-300 rounded-md\"
                />
              </div>
              <div className=\"flex items-center\">
                <label className=\"flex items-center\">
                  <input
                    type=\"checkbox\"
                    checked={config.enable_deduplication}
                    onChange={(e) => setConfig({...config, enable_deduplication: e.target.checked})}
                    className=\"mr-2\"
                  />
                  <span className=\"text-sm font-medium text-gray-700\">Enable Deduplication</span>
                </label>
              </div>
              <div className=\"flex items-center\">
                <label className=\"flex items-center\">
                  <input
                    type=\"checkbox\"
                    checked={config.use_reranker}
                    onChange={(e) => setConfig({...config, use_reranker: e.target.checked})}
                    className=\"mr-2\"
                  />
                  <span className=\"text-sm font-medium text-gray-700\">Use Reranker</span>
                </label>
              </div>
            </div>
          ) : (
            <div className=\"text-sm text-gray-600 p-4 bg-gray-50 rounded-md\">
              Using optimized default settings:
              <ul className=\"mt-2 list-disc list-inside\">
                <li>Chunk size: {config.chunk_size} tokens</li>
                <li>Overlap: {config.chunk_overlap} tokens</li>
                <li>BGE-M3 embeddings with reranking</li>
                <li>Deduplication enabled</li>
              </ul>
            </div>
          )}
        </div>
      </div>

      {/* Step 3: Build RAG */}
      <div className=\"mb-8\">
        <h3 className=\"text-lg font-semibold mb-4 flex items-center\">
          <span className=\"bg-blue-500 text-white rounded-full w-6 h-6 flex items-center justify-center text-sm mr-2\">3</span>
          Build RAG System
        </h3>

        <div className=\"space-y-4\">
          <button
            onClick={handleBuildRAG}
            disabled={isBuilding || !uploadResult}
            className=\"bg-green-500 text-white px-6 py-2 rounded-md hover:bg-green-600 disabled:opacity-50 disabled:cursor-not-allowed flex items-center\"
          >
            {isBuilding ? (
              <>
                <Loader className=\"animate-spin h-4 w-4 mr-2\" />
                Building RAG...
              </>
            ) : (
              <>
                <Settings className=\"h-4 w-4 mr-2\" />
                Build RAG System
              </>
            )}
          </button>

          {buildResult && (
            <div className=\"p-4 bg-blue-50 border border-blue-200 rounded-md\">
              <div className=\"flex items-center mb-2\">
                <CheckCircle className=\"h-5 w-5 text-blue-500 mr-2\" />
                <span className=\"font-medium text-blue-700\">RAG Build Started!</span>
              </div>
              <div className=\"text-sm text-blue-600\">
                <p>RAG ID: <code className=\"bg-blue-100 px-2 py-1 rounded\">{buildResult.rag_id}</code></p>
                <p>Status: {buildResult.status}</p>
                <p>You can track progress in the RAG list.</p>
              </div>
            </div>
          )}

          {buildResult && (
            <div className=\"flex space-x-4\">
              <button
                onClick={resetForm}
                className=\"bg-gray-500 text-white px-4 py-2 rounded-md hover:bg-gray-600\"
              >
                Create Another RAG
              </button>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default RAGCreator;