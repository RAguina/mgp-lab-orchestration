#!/usr/bin/env python3
"""
RAG Document Processing Pipeline Test
Test complete processing pipeline with sample documents
"""

import sys
import os
import tempfile
from pathlib import Path

# UTF-8 handling for Windows
os.environ['PYTHONIOENCODING'] = 'utf-8'
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.stderr.reconfigure(encoding='utf-8', errors='replace')

# Add lab path
LAB_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, LAB_ROOT)

def create_test_documents():
    """Create sample documents for testing"""
    test_docs = {}
    
    # Sample markdown document
    test_docs['sample.md'] = """# Machine Learning Guide

## Introduction
Machine learning is a subset of artificial intelligence (AI) that enables computers to learn and make decisions from data without being explicitly programmed for every task.

## Key Concepts

### Supervised Learning
In supervised learning, algorithms learn from labeled training data to make predictions on new, unseen data.

- **Classification**: Predicting categories (e.g., spam vs. not spam)
- **Regression**: Predicting continuous values (e.g., house prices)

### Unsupervised Learning  
Unsupervised learning finds patterns in data without labeled examples.

- **Clustering**: Grouping similar data points
- **Dimensionality Reduction**: Simplifying data while preserving important features

## Popular Algorithms

1. **Linear Regression**: Simple but effective for many prediction tasks
2. **Random Forest**: Ensemble method that combines multiple decision trees
3. **Neural Networks**: Inspired by biological neural networks, powerful for complex patterns

## Applications

Machine learning is used in:
- Image recognition and computer vision
- Natural language processing
- Recommendation systems
- Autonomous vehicles
- Medical diagnosis

## Conclusion
Machine learning continues to evolve rapidly, with new techniques and applications emerging regularly. Understanding the fundamentals is crucial for anyone working in data science or AI.
"""

    # Sample text document with some repetition (for deduplication testing)
    test_docs['concepts.txt'] = """Core Concepts in Artificial Intelligence

Artificial Intelligence (AI) is a broad field that encompasses machine learning, deep learning, and natural language processing.

Machine learning is a subset of artificial intelligence that enables computers to learn from data. This is the same concept mentioned in many AI textbooks.

Deep learning uses neural networks with multiple layers to process complex patterns in data. Neural networks are inspired by biological neural networks in the brain.

Natural language processing (NLP) helps computers understand and generate human language. NLP is essential for chatbots, translation services, and text analysis.

Key applications include:
- Computer vision for image analysis
- Speech recognition systems  
- Predictive analytics
- Autonomous systems
- Medical AI diagnostics

Machine learning algorithms can be categorized into supervised and unsupervised learning approaches. Supervised learning uses labeled data, while unsupervised learning finds patterns without labels.

The field of artificial intelligence continues to grow rapidly with new breakthroughs in machine learning and deep learning techniques.
"""

    # Sample technical document
    test_docs['embeddings.txt'] = """Vector Embeddings in Machine Learning

Vector embeddings are dense numerical representations of discrete objects like words, documents, or images. They capture semantic relationships in a continuous vector space.

## Word Embeddings

Word embeddings map words to high-dimensional vectors where semantically similar words are located close to each other.

Popular models include:
- Word2Vec (CBOW and Skip-gram)
- GloVe (Global Vectors)
- FastText

## Document Embeddings

Document embeddings represent entire documents as vectors, enabling document similarity calculations and clustering.

Techniques include:
- Doc2Vec
- Universal Sentence Encoder  
- BERT-based embeddings
- BGE-M3 (BAAI General Embedding)

## Applications

Vector embeddings enable:
- Semantic search and retrieval
- Document clustering and classification
- Recommendation systems
- Language translation
- Question answering systems

## BGE-M3 Features

BGE-M3 (BAAI General Embedding Model 3) offers:
- Multi-lingual support (100+ languages)
- Multi-granularity (sentence, passage, document)
- Multi-functionality (dense, sparse, colbert retrieval)
- High performance on MTEB benchmarks

The model produces 1024-dimensional embeddings with L2 normalization for cosine similarity calculations.
"""

    return test_docs

def test_smart_chunker():
    """Test smart chunking component"""
    print("Testing Smart Chunker...")
    
    try:
        from langchain_integration.rag.processing.smart_chunker import SmartChunker
        
        # Create chunker
        chunker = SmartChunker(chunk_size=200, overlap=50)
        
        # Test with sample text
        test_text = create_test_documents()['sample.md']
        
        # Chunk the document
        chunks = chunker.chunk_document(
            text=test_text,
            doc_id="test_doc",
            doc_metadata={"source": "test"}
        )
        
        print(f"OK Smart chunker created {len(chunks)} chunks")
        
        # Show chunk details
        for i, chunk in enumerate(chunks[:3]):  # Show first 3
            print(f"   Chunk {i+1}: {chunk.metrics.token_count} tokens, quality: {chunk.metrics.quality_score:.2f}")
            print(f"   Content: {chunk.content[:100]}...")
        
        if len(chunks) > 3:
            print(f"   ... and {len(chunks)-3} more chunks")
        
        return True
        
    except Exception as e:
        print(f"FAIL Smart chunker test: {e}")
        return False

def test_document_parser():
    """Test document parser component"""
    print("\nTesting Document Parser...")
    
    try:
        from langchain_integration.rag.processing.document_parser import DocumentParser
        
        # Create parser
        parser = DocumentParser()
        
        # Test with temporary files
        test_docs = create_test_documents()
        
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            parsed_docs = []
            
            for filename, content in test_docs.items():
                # Write test file
                file_path = temp_path / filename
                file_path.write_text(content, encoding='utf-8')
                
                # Parse file
                parsed_doc = parser.parse_file(str(file_path))
                parsed_docs.append(parsed_doc)
                
                print(f"OK Parsed {filename}: {parsed_doc.word_count} words, type: {parsed_doc.file_type}")
        
        print(f"OK Document parser processed {len(parsed_docs)} files")
        return True
        
    except Exception as e:
        print(f"FAIL Document parser test: {e}")
        return False

def test_deduplicator():
    """Test semantic deduplicator"""
    print("\nTesting Semantic Deduplicator...")
    
    try:
        from langchain_integration.rag.processing.deduplicator import SemanticDeduplicator
        from langchain_integration.rag.processing.smart_chunker import SmartChunker
        
        # Create components
        dedup = SemanticDeduplicator()
        chunker = SmartChunker(chunk_size=150)
        
        # Create test chunks with some duplicates
        test_texts = [
            "Machine learning is a subset of artificial intelligence.",
            "Machine learning is a subset of AI technology.",  # Similar
            "Deep learning uses neural networks for pattern recognition.",
            "Neural networks are used in deep learning for patterns.", # Similar
            "Natural language processing helps computers understand text.",
            "Machine learning is a subset of artificial intelligence."  # Exact duplicate
        ]
        
        chunks = []
        for i, text in enumerate(test_texts):
            doc_chunks = chunker.chunk_document(text, f"test_doc_{i}")
            chunks.extend(doc_chunks)
        
        print(f"   Created {len(chunks)} test chunks")
        
        # Deduplicate
        unique_chunks, duplicate_groups = dedup.deduplicate_chunks(chunks)
        
        print(f"OK Deduplication: {len(chunks)} -> {len(unique_chunks)} chunks")
        print(f"   Found {len(duplicate_groups)} duplicate groups")
        
        # Show deduplication details
        for i, group in enumerate(duplicate_groups[:2]):  # Show first 2 groups
            print(f"   Group {i+1}: {len(group.duplicates)} duplicates, strategy: {group.merge_strategy}")
        
        return True
        
    except Exception as e:
        print(f"FAIL Deduplicator test: {e}")
        return False

def test_complete_pipeline():
    """Test complete document processing pipeline"""
    print("\nTesting Complete Pipeline...")
    
    try:
        from langchain_integration.rag.processing.document_pipeline import DocumentProcessingPipeline, PipelineConfig
        
        # Create pipeline with test config
        config = PipelineConfig(
            chunk_size=300,
            chunk_overlap=50,
            min_quality_score=0.2,
            enable_deduplication=True
        )
        
        pipeline = DocumentProcessingPipeline(config)
        
        # Create temporary test files
        test_docs = create_test_documents()
        
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Write test files
            file_paths = []
            for filename, content in test_docs.items():
                file_path = temp_path / filename
                file_path.write_text(content, encoding='utf-8')
                file_paths.append(str(file_path))
            
            # Process files through pipeline
            result = pipeline.process_files(file_paths)
            
            print(f"OK Pipeline processed {len(file_paths)} files")
            print(f"   Success: {result.success}")
            print(f"   Chunks created: {len(result.chunks)}")
            print(f"   Errors: {len(result.errors)}")
            
            # Show stats
            stats = result.processing_stats
            doc_stats = stats.get('documents', {})
            chunk_stats = stats.get('chunks', {})
            
            print(f"   Total words: {doc_stats.get('total_words', 0)}")
            print(f"   Avg tokens per chunk: {chunk_stats.get('avg_tokens_per_chunk', 0):.1f}")
            print(f"   Avg quality score: {chunk_stats.get('avg_quality_score', 0):.2f}")
            
            # Show sample chunks
            print("\n   Sample chunks:")
            for i, chunk in enumerate(result.chunks[:3]):
                print(f"   [{i+1}] {chunk.section_type}: {chunk.content[:80]}...")
            
            return result.success
        
    except Exception as e:
        print(f"FAIL Pipeline test: {e}")
        return False

def test_pipeline_config():
    """Test pipeline configuration validation"""
    print("\nTesting Pipeline Configuration...")
    
    try:
        from langchain_integration.rag.processing.document_pipeline import PipelineConfig, DocumentProcessingPipeline
        
        # Test valid config
        valid_config = PipelineConfig(chunk_size=400, chunk_overlap=50)
        pipeline = DocumentProcessingPipeline(valid_config)
        
        issues = pipeline.validate_config()
        if not issues:
            print("OK Valid config passed validation")
        else:
            print(f"WARN Valid config had issues: {issues}")
        
        # Test invalid config
        invalid_config = PipelineConfig(chunk_size=100, chunk_overlap=150)  # overlap > size
        pipeline2 = DocumentProcessingPipeline(invalid_config)
        
        issues2 = pipeline2.validate_config()
        if issues2:
            print(f"OK Invalid config caught: {len(issues2)} issues")
        else:
            print("WARN Invalid config not caught")
        
        return True
        
    except Exception as e:
        print(f"FAIL Config test: {e}")
        return False

if __name__ == "__main__":
    print("RAG Document Processing Pipeline Tests")
    print("=" * 50)
    
    success_count = 0
    total_tests = 5
    
    # Run tests
    tests = [
        test_smart_chunker,
        test_document_parser,
        test_deduplicator,
        test_complete_pipeline,
        test_pipeline_config
    ]
    
    for test_func in tests:
        try:
            if test_func():
                success_count += 1
        except Exception as e:
            print(f"FAIL {test_func.__name__}: {e}")
    
    print("\n" + "=" * 50)
    print("Tests completed: {success_count}/{total_tests}".format(
        success_count=success_count, total_tests=total_tests))
    
    if success_count == total_tests:
        print("SUCCESS All tests passed! Document processing pipeline ready.")
    else:
        failed = total_tests - success_count
        print("WARN {failed} test(s) failed. Check logs above.".format(failed=failed))
    
    print("\nNext steps:")
    print("   1. Test with real PDF/DOCX files")
    print("   2. Integrate with vector store (Milvus)")
    print("   3. Add progress tracking for large documents")
    print("   4. Test with production-scale document collections")