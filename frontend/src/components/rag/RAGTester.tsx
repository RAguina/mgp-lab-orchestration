import React, { useState, useCallback } from 'react';
import { Search, Loader, BookOpen, Clock, Target, BarChart3 } from 'lucide-react';

interface RAGTesterProps {
  ragId: string;
  workspaceId?: string;
}

interface SearchResult {
  uri: string;
  excerpt: string;
  metadata: any;
  quality_score: number;
  similarity_score: number;
  content?: string;
}

interface SearchResponse {
  rag_id: string;
  query: string;
  params: {
    top_k: number;
    ef_search: number;
    include_full_content: boolean;
    use_reranker: boolean;
    rerank_applied: boolean;
  };
  candidates: SearchResult[];
  total_found: number;
  returned_count: number;
}

interface EvaluationResult {
  run_id: string;
  rag_id: string;
  evaluation_summary: {
    queries_evaluated: number;
    queries_successful: number;
    success_rate: number;
    avg_recall: number;
    avg_precision: number;
    avg_ndcg: number;
    avg_latency_ms: number;
    p95_latency_ms: number;
  };
  success_criteria: {
    recall_target_met: boolean;
    latency_target_met: boolean;
    overall_success: boolean;
  };
}

const RAGTester: React.FC<RAGTesterProps> = ({ ragId, workspaceId }) => {
  const [query, setQuery] = useState('');
  const [searchResults, setSearchResults] = useState<SearchResponse | null>(null);
  const [isSearching, setIsSearching] = useState(false);
  const [searchError, setSearchError] = useState<string | null>(null);
  
  // Search parameters
  const [topK, setTopK] = useState(5);
  const [efSearch, setEfSearch] = useState(96);
  const [useReranker, setUseReranker] = useState(true);
  const [includeFullContent, setIncludeFullContent] = useState(false);
  
  // Evaluation
  const [isEvaluating, setIsEvaluating] = useState(false);
  const [evaluationResult, setEvaluationResult] = useState<EvaluationResult | null>(null);
  const [evaluationError, setEvaluationError] = useState<string | null>(null);
  
  const [searchHistory, setSearchHistory] = useState<Array<{query: string, timestamp: Date, latency: number}>>([]);

  const handleSearch = useCallback(async () => {
    if (!query.trim()) return;
    
    setIsSearching(true);
    setSearchError(null);
    
    const startTime = Date.now();
    
    try {
      const searchParams = new URLSearchParams({
        top_k: topK.toString(),
        ef_search: efSearch.toString(),
        use_reranker: useReranker.toString(),
        include_full_content: includeFullContent.toString()
      });

      const headers: HeadersInit = {
        'Content-Type': 'application/json'
      };
      if (workspaceId) {
        headers['workspace-id'] = workspaceId;
      }

      const response = await fetch(`/api/rag/${ragId}/search?${searchParams}`, {
        method: 'POST',
        headers,
        body: JSON.stringify({ query })
      });

      if (!response.ok) {
        throw new Error(`Search failed: ${response.statusText}`);
      }

      const result: SearchResponse = await response.json();
      setSearchResults(result);
      
      // Add to history
      const latency = Date.now() - startTime;
      setSearchHistory(prev => [
        { query, timestamp: new Date(), latency },
        ...prev.slice(0, 9) // Keep last 10 searches
      ]);

    } catch (error) {
      setSearchError(error instanceof Error ? error.message : 'Search failed');
    } finally {
      setIsSearching(false);
    }
  }, [query, ragId, topK, efSearch, useReranker, includeFullContent, workspaceId]);

  const handleEvaluation = useCallback(async () => {
    setIsEvaluating(true);
    setEvaluationError(null);
    
    try {
      const headers: HeadersInit = {
        'Content-Type': 'application/json'
      };
      if (workspaceId) {
        headers['workspace-id'] = workspaceId;
      }

      const response = await fetch(`/api/rag/${ragId}/eval?use_sample_goldset=true&top_k=${topK}`, {
        method: 'POST',
        headers
      });

      if (!response.ok) {
        throw new Error(`Evaluation failed: ${response.statusText}`);
      }

      const result: EvaluationResult = await response.json();
      setEvaluationResult(result);

    } catch (error) {
      setEvaluationError(error instanceof Error ? error.message : 'Evaluation failed');
    } finally {
      setIsEvaluating(false);
    }
  }, [ragId, topK, workspaceId]);

  const formatScore = (score: number): string => {
    return (score * 100).toFixed(1) + '%';
  };

  const formatLatency = (ms: number): string => {
    return ms.toFixed(0) + 'ms';
  };

  return (
    <div className=\"max-w-6xl mx-auto p-6 space-y-6\">
      <div className=\"bg-white rounded-lg shadow-lg p-6\">
        <h2 className=\"text-2xl font-bold text-gray-900 mb-6 flex items-center\">
          <BookOpen className=\"h-6 w-6 mr-2\" />
          RAG System Tester
        </h2>

        {/* RAG Info */}
        <div className=\"mb-6 p-4 bg-gray-50 rounded-md\">
          <div className=\"text-sm text-gray-600\">RAG ID:</div>
          <div className=\"font-mono text-sm\">{ragId}</div>
        </div>

        {/* Search Parameters */}
        <div className=\"mb-6\">
          <h3 className=\"text-lg font-semibold mb-4\">Search Parameters</h3>
          <div className=\"grid grid-cols-2 md:grid-cols-4 gap-4\">
            <div>
              <label className=\"block text-sm font-medium text-gray-700 mb-1\">Top K</label>
              <input
                type=\"number\"
                min=\"1\"
                max=\"50\"
                value={topK}
                onChange={(e) => setTopK(parseInt(e.target.value) || 5)}
                className=\"w-full px-3 py-2 border border-gray-300 rounded-md\"
              />
            </div>
            <div>
              <label className=\"block text-sm font-medium text-gray-700 mb-1\">EF Search</label>
              <input
                type=\"number\"
                min=\"16\"
                max=\"512\"
                value={efSearch}
                onChange={(e) => setEfSearch(parseInt(e.target.value) || 96)}
                className=\"w-full px-3 py-2 border border-gray-300 rounded-md\"
              />
            </div>
            <div className=\"flex items-center\">
              <label className=\"flex items-center\">
                <input
                  type=\"checkbox\"
                  checked={useReranker}
                  onChange={(e) => setUseReranker(e.target.checked)}
                  className=\"mr-2\"
                />
                <span className=\"text-sm font-medium text-gray-700\">Use Reranker</span>
              </label>
            </div>
            <div className=\"flex items-center\">
              <label className=\"flex items-center\">
                <input
                  type=\"checkbox\"
                  checked={includeFullContent}
                  onChange={(e) => setIncludeFullContent(e.target.checked)}
                  className=\"mr-2\"
                />
                <span className=\"text-sm font-medium text-gray-700\">Full Content</span>
              </label>
            </div>
          </div>
        </div>

        {/* Search Interface */}
        <div className=\"mb-6\">
          <h3 className=\"text-lg font-semibold mb-4\">Search Query</h3>
          <div className=\"flex space-x-4\">
            <input
              type=\"text\"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              onKeyPress={(e) => e.key === 'Enter' && handleSearch()}
              placeholder=\"Enter your search query...\"
              className=\"flex-1 px-4 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500\"
            />
            <button
              onClick={handleSearch}
              disabled={isSearching || !query.trim()}
              className=\"bg-blue-500 text-white px-6 py-2 rounded-md hover:bg-blue-600 disabled:opacity-50 flex items-center\"
            >
              {isSearching ? (
                <Loader className=\"animate-spin h-4 w-4 mr-2\" />
              ) : (
                <Search className=\"h-4 w-4 mr-2\" />
              )}
              Search
            </button>
          </div>
        </div>

        {/* Search Error */}
        {searchError && (
          <div className=\"mb-6 p-4 bg-red-50 border border-red-200 rounded-md text-red-700\">
            {searchError}
          </div>
        )}

        {/* Search Results */}
        {searchResults && (
          <div className=\"mb-6\">
            <h3 className=\"text-lg font-semibold mb-4\">
              Search Results ({searchResults.returned_count} of {searchResults.total_found} found)
            </h3>
            
            {/* Search Metadata */}
            <div className=\"mb-4 p-3 bg-blue-50 rounded-md text-sm\">
              <div className=\"grid grid-cols-2 md:grid-cols-4 gap-4\">
                <div>Query: <span className=\"font-medium\">\"{searchResults.query}\"</span></div>
                <div>Top K: <span className=\"font-medium\">{searchResults.params.top_k}</span></div>
                <div>EF Search: <span className=\"font-medium\">{searchResults.params.ef_search}</span></div>
                <div>Reranked: <span className=\"font-medium\">{searchResults.params.rerank_applied ? 'Yes' : 'No'}</span></div>
              </div>
            </div>

            {/* Results List */}
            <div className=\"space-y-4\">
              {searchResults.candidates.map((result, index) => (
                <div key={index} className=\"border border-gray-200 rounded-lg p-4\">
                  <div className=\"flex items-start justify-between mb-2\">
                    <div className=\"text-sm text-gray-600\">
                      Result #{index + 1}
                    </div>
                    <div className=\"text-right text-sm\">
                      <div className=\"font-medium\">Similarity: {formatScore(result.similarity_score)}</div>
                      <div className=\"text-gray-600\">Quality: {formatScore(result.quality_score)}</div>
                    </div>
                  </div>
                  
                  <div className=\"mb-2\">
                    <div className=\"text-sm text-gray-600 mb-1\">Source:</div>
                    <div className=\"font-mono text-xs text-blue-600\">{result.uri}</div>
                  </div>
                  
                  <div className=\"mb-2\">
                    <div className=\"text-sm text-gray-600 mb-1\">Content:</div>
                    <div className=\"text-sm bg-gray-50 p-3 rounded border-l-4 border-blue-200\">
                      {result.content || result.excerpt}
                    </div>
                  </div>
                  
                  {result.metadata && Object.keys(result.metadata).length > 0 && (
                    <details className=\"mt-2\">
                      <summary className=\"text-sm text-gray-600 cursor-pointer\">Metadata</summary>
                      <pre className=\"text-xs bg-gray-100 p-2 rounded mt-1 overflow-auto\">
                        {JSON.stringify(result.metadata, null, 2)}
                      </pre>
                    </details>
                  )}
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Search History */}
        {searchHistory.length > 0 && (
          <div className=\"mb-6\">
            <h3 className=\"text-lg font-semibold mb-4 flex items-center\">
              <Clock className=\"h-5 w-5 mr-2\" />
              Recent Searches
            </h3>
            <div className=\"space-y-2\">
              {searchHistory.slice(0, 5).map((item, index) => (
                <div key={index} className=\"flex items-center justify-between p-2 bg-gray-50 rounded text-sm\">
                  <span className=\"flex-1 truncate\">
                    \"{item.query}\"
                  </span>
                  <span className=\"text-gray-500 ml-4\">
                    {formatLatency(item.latency)} - {item.timestamp.toLocaleTimeString()}
                  </span>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>

      {/* Evaluation Section */}
      <div className=\"bg-white rounded-lg shadow-lg p-6\">
        <h3 className=\"text-lg font-semibold mb-4 flex items-center\">
          <BarChart3 className=\"h-5 w-5 mr-2\" />
          Quality Evaluation
        </h3>
        
        <div className=\"mb-4\">
          <p className=\"text-sm text-gray-600 mb-4\">
            Run standardized evaluation with sample queries to assess RAG system quality.
          </p>
          
          <button
            onClick={handleEvaluation}
            disabled={isEvaluating}
            className=\"bg-purple-500 text-white px-6 py-2 rounded-md hover:bg-purple-600 disabled:opacity-50 flex items-center\"
          >
            {isEvaluating ? (
              <Loader className=\"animate-spin h-4 w-4 mr-2\" />
            ) : (
              <Target className=\"h-4 w-4 mr-2\" />
            )}
            {isEvaluating ? 'Evaluating...' : 'Run Evaluation'}
          </button>
        </div>

        {evaluationError && (
          <div className=\"mb-4 p-4 bg-red-50 border border-red-200 rounded-md text-red-700\">
            {evaluationError}
          </div>
        )}

        {evaluationResult && (
          <div className=\"space-y-4\">
            <div className=\"p-4 bg-green-50 border border-green-200 rounded-md\">
              <h4 className=\"font-semibold text-green-800 mb-2\">Evaluation Complete</h4>
              <div className=\"text-sm text-green-700\">
                Run ID: <code>{evaluationResult.run_id}</code>
              </div>
            </div>
            
            {/* Success Criteria */}
            <div className=\"grid grid-cols-1 md:grid-cols-3 gap-4\">
              <div className={`p-4 rounded-md border ${
                evaluationResult.success_criteria.recall_target_met
                  ? 'bg-green-50 border-green-200'
                  : 'bg-red-50 border-red-200'
              }`}>
                <div className=\"text-sm font-medium\">Recall Target</div>
                <div className={`text-lg font-bold ${
                  evaluationResult.success_criteria.recall_target_met
                    ? 'text-green-600'
                    : 'text-red-600'
                }`}>
                  {evaluationResult.success_criteria.recall_target_met ? 'PASS' : 'FAIL'}
                </div>
              </div>
              
              <div className={`p-4 rounded-md border ${
                evaluationResult.success_criteria.latency_target_met
                  ? 'bg-green-50 border-green-200'
                  : 'bg-red-50 border-red-200'
              }`}>
                <div className=\"text-sm font-medium\">Latency Target</div>
                <div className={`text-lg font-bold ${
                  evaluationResult.success_criteria.latency_target_met
                    ? 'text-green-600'
                    : 'text-red-600'
                }`}>
                  {evaluationResult.success_criteria.latency_target_met ? 'PASS' : 'FAIL'}
                </div>
              </div>
              
              <div className={`p-4 rounded-md border ${
                evaluationResult.success_criteria.overall_success
                  ? 'bg-green-50 border-green-200'
                  : 'bg-red-50 border-red-200'
              }`}>
                <div className=\"text-sm font-medium\">Overall</div>
                <div className={`text-lg font-bold ${
                  evaluationResult.success_criteria.overall_success
                    ? 'text-green-600'
                    : 'text-red-600'
                }`}>
                  {evaluationResult.success_criteria.overall_success ? 'PASS' : 'FAIL'}
                </div>
              </div>
            </div>
            
            {/* Detailed Metrics */}
            <div className=\"grid grid-cols-2 md:grid-cols-4 gap-4\">
              <div className=\"p-3 bg-blue-50 rounded-md\">
                <div className=\"text-sm text-blue-600 font-medium\">Recall@10</div>
                <div className=\"text-lg font-bold text-blue-800\">
                  {formatScore(evaluationResult.evaluation_summary.avg_recall)}
                </div>
              </div>
              
              <div className=\"p-3 bg-purple-50 rounded-md\">
                <div className=\"text-sm text-purple-600 font-medium\">Precision@10</div>
                <div className=\"text-lg font-bold text-purple-800\">
                  {formatScore(evaluationResult.evaluation_summary.avg_precision)}
                </div>
              </div>
              
              <div className=\"p-3 bg-orange-50 rounded-md\">
                <div className=\"text-sm text-orange-600 font-medium\">NDCG@10</div>
                <div className=\"text-lg font-bold text-orange-800\">
                  {formatScore(evaluationResult.evaluation_summary.avg_ndcg)}
                </div>
              </div>
              
              <div className=\"p-3 bg-gray-50 rounded-md\">
                <div className=\"text-sm text-gray-600 font-medium\">Avg Latency</div>
                <div className=\"text-lg font-bold text-gray-800\">
                  {formatLatency(evaluationResult.evaluation_summary.avg_latency_ms)}
                </div>
              </div>
            </div>
            
            <div className=\"text-sm text-gray-600\">
              Evaluated {evaluationResult.evaluation_summary.queries_evaluated} queries 
              ({evaluationResult.evaluation_summary.queries_successful} successful,{' '}
              {formatScore(evaluationResult.evaluation_summary.success_rate)} success rate)
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default RAGTester;