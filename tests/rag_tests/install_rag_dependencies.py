#!/usr/bin/env python3
"""
RAG Dependencies Installation Script
Step-by-step installation with validation
"""

import subprocess
import sys
import importlib

def run_pip_install(package, description=""):
    """Install package with pip and handle errors"""
    print(f"\n{'='*50}")
    print(f"Installing {package}")
    if description:
        print(f"Purpose: {description}")
    print('='*50)
    
    try:
        result = subprocess.run([
            sys.executable, "-m", "pip", "install", package
        ], capture_output=True, text=True, check=True)
        
        print(f"✅ Successfully installed {package}")
        if result.stdout:
            print("Output:", result.stdout.strip())
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install {package}")
        print(f"Error: {e}")
        if e.stderr:
            print("Error details:", e.stderr.strip())
        return False

def test_import(package, import_name=None):
    """Test if package can be imported"""
    import_name = import_name or package.replace("-", "_")
    try:
        __import__(import_name)
        print(f"✅ {package} imports successfully")
        return True
    except ImportError as e:
        print(f"❌ {package} import failed: {e}")
        return False

def install_rag_dependencies():
    """Install all RAG dependencies in order"""
    
    print("RAG DEPENDENCIES INSTALLATION")
    print("=" * 70)
    print("Installing required packages for RAG system...")
    
    # Core dependencies in installation order
    dependencies = [
        # Basic ML/API dependencies
        ("python-multipart", "FastAPI file uploads"),
        ("numpy", "Numerical operations"),
        ("scikit-learn", "ML utilities"),
        
        # Document processing
        ("PyPDF2", "PDF document parsing"),
        ("python-docx", "DOCX document parsing"),
        
        # RAG core components  
        ("sentence-transformers", "BGE-M3 embeddings"),
        ("pymilvus>=2.4.0", "Milvus vector database client"),
        ("minio", "MinIO/S3 object storage"),
        
        # Optional but recommended
        ("redis", "Progress tracking (optional)"),
        ("psutil", "System monitoring")
    ]
    
    installed = 0
    failed = []
    
    for package, description in dependencies:
        if run_pip_install(package, description):
            installed += 1
        else:
            failed.append(package)
    
    print(f"\n{'='*70}")
    print("INSTALLATION SUMMARY")
    print('='*70)
    print(f"Successfully installed: {installed}/{len(dependencies)}")
    
    if failed:
        print(f"Failed installations: {failed}")
    
    # Test imports
    print(f"\n{'='*50}")
    print("TESTING IMPORTS")
    print('='*50)
    
    test_packages = [
        ("multipart", "python-multipart"),
        ("numpy", "numpy"), 
        ("sklearn", "scikit-learn"),
        ("PyPDF2", "PyPDF2"),
        ("docx", "python-docx"),
        ("sentence_transformers", "sentence-transformers"),
        ("pymilvus", "pymilvus"),
        ("minio", "minio"),
        ("redis", "redis"),
        ("psutil", "psutil")
    ]
    
    working_imports = 0
    for import_name, package in test_packages:
        if test_import(package, import_name):
            working_imports += 1
    
    print(f"\nWorking imports: {working_imports}/{len(test_packages)}")
    
    # Final assessment
    print(f"\n{'='*70}")
    print("FINAL ASSESSMENT")
    print('='*70)
    
    success_rate = (working_imports / len(test_packages)) * 100
    
    if success_rate >= 90:
        print("🎉 INSTALLATION COMPLETE!")
        print("✅ All dependencies installed successfully")
        print("✅ RAG system is ready for testing")
        print("\nNext steps:")
        print("1. Setup Milvus database (docker run -p 19530:19530 milvusdb/milvus:latest)")
        print("2. Setup MinIO server (docker run -p 9000:9000 minio/minio server /data)")
        print("3. Run: python test_rag_realistic.py")
        return True
        
    elif success_rate >= 70:
        print("⚠️  MOSTLY INSTALLED")
        print(f"✅ {working_imports}/{len(test_packages)} dependencies working")
        print("⚠️  Some packages may need manual installation")
        print("⚠️  Check failed installations above")
        return False
        
    else:
        print("❌ INSTALLATION INCOMPLETE") 
        print(f"❌ Only {working_imports}/{len(test_packages)} dependencies working")
        print("❌ Major installation issues detected")
        print("\nTroubleshooting:")
        print("- Check Python version (requires 3.8+)")
        print("- Update pip: python -m pip install --upgrade pip")
        print("- Check internet connection")
        return False

if __name__ == "__main__":
    success = install_rag_dependencies()
    sys.exit(0 if success else 1)