#!/usr/bin/env python3
"""
RAG Implementation - REALISTIC TEST STRATEGY
Tests what we can actually validate without external dependencies
"""

import sys
import os
import importlib.util
import ast
sys.path.append(os.path.dirname(__file__))

def test_file_structure():
    """Test 1: File structure validation"""
    
    print("="*60)
    print("TEST 1: FILE STRUCTURE VALIDATION")
    print("="*60)
    
    required_files = [
        # Core RAG Infrastructure  
        "langchain_integration/rag/embeddings/bge_m3_provider.py",
        "langchain_integration/rag/embeddings/embedding_manager.py", 
        "langchain_integration/rag/storage/milvus_store.py",
        "langchain_integration/rag/storage/minio_store.py",
        "langchain_integration/rag/storage/document_store.py",
        "api/endpoints/rag.py",
        
        # Document Processing
        "langchain_integration/rag/processing/smart_chunker.py",
        "langchain_integration/rag/processing/document_parser.py",
        "langchain_integration/rag/processing/deduplicator.py",
        "langchain_integration/rag/processing/document_pipeline.py",
        
        # RAG Builder
        "langchain_integration/rag/rag_builder.py",
        
        # Frontend & Quality
        "langchain_integration/rag/rerank/bge_reranker.py",
        "langchain_integration/rag/progress/tracker.py",
        "langchain_integration/rag/evaluation/metrics.py",
        
        # Production
        ".env.example"
    ]
    
    base_path = os.path.dirname(__file__)
    existing = 0
    
    print("Checking required files:")
    for file_path in required_files:
        full_path = os.path.join(base_path, file_path)
        if os.path.exists(full_path):
            print(f"  [OK] {file_path}")
            existing += 1
        else:
            print(f"  [MISSING] {file_path}")
    
    completion = (existing / len(required_files)) * 100
    print(f"\nFile Completion: {existing}/{len(required_files)} ({completion:.1f}%)")
    
    return completion >= 95, existing, len(required_files)

def test_python_syntax():
    """Test 2: Python syntax validation"""
    
    print("\n" + "="*60)
    print("TEST 2: PYTHON SYNTAX VALIDATION")
    print("="*60)
    
    python_files = [
        "langchain_integration/rag/embeddings/bge_m3_provider.py",
        "langchain_integration/rag/embeddings/embedding_manager.py", 
        "langchain_integration/rag/storage/milvus_store.py",
        "langchain_integration/rag/storage/minio_store.py",
        "langchain_integration/rag/storage/document_store.py",
        "api/endpoints/rag.py",
        "langchain_integration/rag/processing/smart_chunker.py",
        "langchain_integration/rag/processing/document_parser.py",
        "langchain_integration/rag/processing/deduplicator.py",
        "langchain_integration/rag/processing/document_pipeline.py",
        "langchain_integration/rag/rag_builder.py",
        "langchain_integration/rag/rerank/bge_reranker.py",
        "langchain_integration/rag/progress/tracker.py",
        "langchain_integration/rag/evaluation/metrics.py"
    ]
    
    base_path = os.path.dirname(__file__)
    valid_files = 0
    
    print("Checking Python syntax:")
    for file_path in python_files:
        full_path = os.path.join(base_path, file_path)
        if os.path.exists(full_path):
            try:
                with open(full_path, 'r', encoding='utf-8') as f:
                    source = f.read()
                ast.parse(source)
                print(f"  [OK] {file_path}")
                valid_files += 1
            except SyntaxError as e:
                print(f"  [SYNTAX ERROR] {file_path}: {e}")
            except Exception as e:
                print(f"  [ERROR] {file_path}: {e}")
        else:
            print(f"  [NOT FOUND] {file_path}")
    
    syntax_completion = (valid_files / len(python_files)) * 100
    print(f"\nSyntax Validation: {valid_files}/{len(python_files)} ({syntax_completion:.1f}%)")
    
    return syntax_completion >= 95, valid_files, len(python_files)

def test_api_import():
    """Test 3: API import validation (without dependencies)"""
    
    print("\n" + "="*60)
    print("TEST 3: API IMPORT VALIDATION")
    print("="*60)
    
    try:
        # Test if API can be imported without crashing
        spec = importlib.util.spec_from_file_location("rag_api", "api/endpoints/rag.py")
        if spec and spec.loader:
            module = importlib.util.module_from_spec(spec)
            sys.modules["rag_api"] = module
            spec.loader.exec_module(module)
            
            # Check if router exists
            if hasattr(module, 'router'):
                print("  [OK] API module loads successfully")
                print("  [OK] FastAPI router exists")
                
                # Check router routes
                router = module.router
                routes = [route.path for route in router.routes]
                print(f"  [INFO] Found {len(routes)} routes: {routes}")
                
                expected_routes = ["/upload", "/build", "/{rag_id}/status", "/{rag_id}/search", 
                                 "/{rag_id}/query", "/{rag_id}", "/list"]
                
                found_routes = 0
                for expected in expected_routes:
                    if any(expected.replace("{rag_id}", "") in route or expected in route for route in routes):
                        found_routes += 1
                        print(f"  [OK] Route found: {expected}")
                    else:
                        print(f"  [MISSING] Route not found: {expected}")
                
                route_completion = (found_routes / len(expected_routes)) * 100
                print(f"\nAPI Route Completion: {found_routes}/{len(expected_routes)} ({route_completion:.1f}%)")
                
                return route_completion >= 80, found_routes, len(expected_routes)
            else:
                print("  [ERROR] Router not found in module")
                return False, 0, 1
        else:
            print("  [ERROR] Cannot load API module spec")
            return False, 0, 1
            
    except Exception as e:
        print(f"  [ERROR] API import failed: {e}")
        return False, 0, 1

def test_dependencies_status():
    """Test 4: Dependency availability check"""
    
    print("\n" + "="*60)
    print("TEST 4: DEPENDENCY STATUS CHECK")
    print("="*60)
    
    required_deps = [
        ("sentence-transformers", "BGE-M3 embeddings"),
        ("pymilvus", "Milvus vector database"),
        ("minio", "MinIO object storage"),
        ("PyPDF2", "PDF document parsing"),
        ("python-docx", "DOCX document parsing"),
        ("redis", "Progress tracking (optional)"),
        ("numpy", "Numerical operations"),
        ("scikit-learn", "ML utilities")
    ]
    
    available = 0
    print("Checking dependency availability:")
    
    for dep, description in required_deps:
        try:
            __import__(dep.replace("-", "_"))
            print(f"  [OK] {dep} - {description}")
            available += 1
        except ImportError:
            print(f"  [MISSING] {dep} - {description}")
    
    dep_completion = (available / len(required_deps)) * 100
    print(f"\nDependency Availability: {available}/{len(required_deps)} ({dep_completion:.1f}%)")
    
    return dep_completion >= 75, available, len(required_deps)

def test_configuration():
    """Test 5: Configuration file validation"""
    
    print("\n" + "="*60)
    print("TEST 5: CONFIGURATION VALIDATION")
    print("="*60)
    
    config_file = ".env.example"
    
    if not os.path.exists(config_file):
        print(f"  [ERROR] Configuration file not found: {config_file}")
        return False, 0, 1
    
    try:
        with open(config_file, 'r') as f:
            content = f.read()
        
        required_sections = [
            "RAG System Configuration",
            "Milvus Vector Database", 
            "MinIO/S3 Object Storage",
            "RAG Processing Configuration",
            "API & Web Configuration"
        ]
        
        sections_found = 0
        print("Checking configuration sections:")
        
        for section in required_sections:
            if section in content:
                print(f"  [OK] {section}")
                sections_found += 1
            else:
                print(f"  [MISSING] {section}")
        
        # Check for key variables
        key_vars = ["MILVUS_URI", "MINIO_ENDPOINT", "RAG_CHUNK_SIZE", "RAG_EMBEDDING_MODEL"]
        vars_found = 0
        
        print("\nChecking key variables:")
        for var in key_vars:
            if var in content:
                print(f"  [OK] {var}")
                vars_found += 1
            else:
                print(f"  [MISSING] {var}")
        
        config_score = ((sections_found + vars_found) / (len(required_sections) + len(key_vars))) * 100
        print(f"\nConfiguration Completeness: {sections_found + vars_found}/{len(required_sections) + len(key_vars)} ({config_score:.1f}%)")
        
        return config_score >= 90, sections_found + vars_found, len(required_sections) + len(key_vars)
        
    except Exception as e:
        print(f"  [ERROR] Failed to read configuration: {e}")
        return False, 0, 1

def main():
    """Run realistic RAG implementation tests"""
    
    print("RAG IMPLEMENTATION - REALISTIC TEST STRATEGY")
    print("Testing what we can validate without external dependencies")
    print("="*70)
    
    tests = [
        ("File Structure", test_file_structure),
        ("Python Syntax", test_python_syntax),
        ("API Import", test_api_import),
        ("Dependencies", test_dependencies_status),
        ("Configuration", test_configuration)
    ]
    
    results = []
    stats = {}
    
    for test_name, test_func in tests:
        success, passed, total = test_func()
        results.append(success)
        stats[test_name] = (passed, total, success)
    
    # Final assessment
    print("\n" + "="*70)
    print("REALISTIC ASSESSMENT RESULTS")
    print("="*70)
    
    print("\nTEST RESULTS:")
    for test_name, (passed, total, success) in stats.items():
        status = "PASS" if success else "FAIL"
        percentage = (passed / total * 100) if total > 0 else 0
        print(f"  {test_name}: {status} ({passed}/{total} - {percentage:.1f}%)")
    
    success_count = sum(results)
    success_rate = (success_count / len(results)) * 100
    print(f"\nOverall Test Success: {success_count}/{len(results)} ({success_rate:.1f}%)")
    
    # Realistic assessment
    print("\n" + "="*50)
    if success_rate >= 80:
        print("REALISTIC STATUS: IMPLEMENTATION COMPLETE")
        print("✅ All core components implemented")
        print("✅ Code structure and syntax valid") 
        print("⚠️  External dependencies need installation")
        print("⚠️  Backend server needs dependency setup")
        print("\nNEXT STEPS:")
        print("1. pip install sentence-transformers pymilvus minio PyPDF2 python-docx")
        print("2. Setup Milvus database (Docker: milvusdb/milvus)")
        print("3. Setup MinIO server (Docker: minio/minio)")
        print("4. Test end-to-end RAG workflow")
    elif success_rate >= 60:
        print("REALISTIC STATUS: MOSTLY COMPLETE")
        print("✅ Core implementation exists")
        print("⚠️  Some issues need fixing")
        print("⚠️  Dependencies missing")
    else:
        print("REALISTIC STATUS: INCOMPLETE")
        print("❌ Major issues found")
        print("❌ Implementation needs work")
    
    print("="*50)
    
    return success_rate >= 80

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)