#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RAG Implementation Status Check
Simple test without external dependencies or unicode issues
"""

import sys
import os
sys.path.append(os.path.dirname(__file__))

def check_file_exists(file_path, description):
    """Check if a file exists and print result"""
    full_path = os.path.join(os.path.dirname(__file__), file_path)
    if os.path.exists(full_path):
        print(f"[OK] {description}")
        return True
    else:
        print(f"[MISSING] {description}")
        return False

def check_implementation_status():
    """Check RAG implementation file structure"""
    print("RAG Implementation Status Check")
    print("=" * 50)
    
    files_to_check = [
        # Day 1-2: Core Infrastructure
        ("langchain_integration/rag/__init__.py", "RAG Module Init"),
        ("langchain_integration/rag/embeddings/bge_m3_provider.py", "BGE-M3 Provider"),
        ("langchain_integration/rag/embeddings/embedding_manager.py", "Embedding Manager"),
        ("langchain_integration/rag/storage/milvus_store.py", "Milvus Store"),
        ("api/endpoints/rag.py", "RAG API Endpoints"),
        
        # Day 3: Document Processing
        ("langchain_integration/rag/processing/smart_chunker.py", "Smart Chunker"),
        ("langchain_integration/rag/processing/document_parser.py", "Document Parser"),
        ("langchain_integration/rag/processing/deduplicator.py", "Deduplicator"),
        ("langchain_integration/rag/processing/document_pipeline.py", "Processing Pipeline"),
        
        # Day 4: RAG Builder  
        ("langchain_integration/rag/rag_builder.py", "RAG Builder Orchestrator"),
    ]
    
    total = len(files_to_check)
    existing = 0
    
    for file_path, description in files_to_check:
        if check_file_exists(file_path, description):
            existing += 1
    
    print(f"\nStatus: {existing}/{total} files exist ({existing/total*100:.1f}% complete)")
    
    return existing, total

def check_basic_syntax():
    """Check if Python files have valid syntax"""
    print("\nSyntax Check:")
    print("-" * 30)
    
    rag_files = [
        "langchain_integration/rag/rag_builder.py",
        "langchain_integration/rag/processing/document_pipeline.py",
        "api/endpoints/rag.py"
    ]
    
    syntax_ok = 0
    total_checked = 0
    
    for file_path in rag_files:
        full_path = os.path.join(os.path.dirname(__file__), file_path)
        if os.path.exists(full_path):
            total_checked += 1
            try:
                with open(full_path, 'r', encoding='utf-8') as f:
                    compile(f.read(), full_path, 'exec')
                print(f"[OK] {os.path.basename(file_path)} - Valid syntax")
                syntax_ok += 1
            except SyntaxError as e:
                print(f"[ERROR] {os.path.basename(file_path)} - Syntax error: {e}")
            except Exception as e:
                print(f"[ERROR] {os.path.basename(file_path)} - Error: {e}")
    
    print(f"Syntax check: {syntax_ok}/{total_checked} files OK")
    return syntax_ok, total_checked

def check_dependencies():
    """Check which dependencies are missing"""
    print("\nDependency Check:")
    print("-" * 30)
    
    dependencies = [
        ("sentence_transformers", "Sentence Transformers (BGE-M3)"),
        ("pymilvus", "Milvus Client"),
        ("PyPDF2", "PDF Processing"),
        ("docx", "DOCX Processing"),
        ("psutil", "System utilities"),
        ("redis", "Redis (optional)"),
        ("minio", "MinIO S3 client"),
        ("transformers", "HuggingFace Transformers")
    ]
    
    available = 0
    total_deps = len(dependencies)
    
    for module_name, description in dependencies:
        try:
            __import__(module_name)
            print(f"[OK] {description}")
            available += 1
        except ImportError:
            print(f"[MISSING] {description}")
    
    print(f"Dependencies: {available}/{total_deps} available")
    return available, total_deps

if __name__ == "__main__":
    # Check file structure
    existing_files, total_files = check_implementation_status()
    
    # Check syntax
    valid_syntax, total_syntax = check_basic_syntax()
    
    # Check dependencies
    available_deps, total_deps = check_dependencies()
    
    # Summary
    print("\n" + "=" * 50)
    print("SUMMARY:")
    print(f"Files: {existing_files}/{total_files} ({existing_files/total_files*100:.1f}%)")
    print(f"Syntax: {valid_syntax}/{total_syntax} ({valid_syntax/total_syntax*100:.1f}% if files exist)")
    print(f"Dependencies: {available_deps}/{total_deps} ({available_deps/total_deps*100:.1f}%)")
    
    # Overall status
    if existing_files == total_files and valid_syntax == total_syntax:
        if available_deps >= total_deps * 0.7:  # At least 70% of deps
            print("STATUS: Ready for integration testing")
            sys.exit(0)
        else:
            print("STATUS: Need to install dependencies")
            print("Run: pip install sentence-transformers pymilvus PyPDF2 python-docx psutil minio")
            sys.exit(1)
    else:
        print("STATUS: Implementation incomplete")
        sys.exit(1)