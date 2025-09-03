#!/usr/bin/env python3
"""
Browse MinIO stored RAG content
Access and view stored chunks programmatically
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

def browse_minio_content():
    """Browse and display MinIO stored content"""
    
    print("=" * 60)
    print("MINIO CONTENT BROWSER")
    print("=" * 60)
    
    try:
        from langchain_integration.rag.storage.minio_store import MinIODocumentStore
        import json
        
        # Initialize MinIO store
        minio_store = MinIODocumentStore()
        
        print("1. MINIO CONNECTION STATUS")
        print("-" * 30)
        health = minio_store.health_check()
        print(f"Health Status: {'✓ HEALTHY' if health.get('healthy') else '✗ UNHEALTHY'}")
        print(f"Backend: {health.get('backend', 'unknown')}")
        
        # List RAG systems
        print("\n2. RAG SYSTEMS STATISTICS")
        print("-" * 30)
        
        test_rag_stats = minio_store.get_rag_stats("test_rag_001")
        print(f"RAG ID: test_rag_001")
        print(f"  Objects: {test_rag_stats.get('object_count', 0)}")
        print(f"  Total size: {test_rag_stats.get('total_size_mb', 0):.2f} MB")
        print(f"  Storage: {test_rag_stats.get('storage_backend', 'unknown')}")
        
        if test_rag_stats.get('object_count', 0) > 0:
            print("\n3. SAMPLE CONTENT ACCESS")
            print("-" * 30)
            
            # Try to access a few stored chunks
            sample_uris = [
                "minio://rag-storage/chunks/test_rag_001/test_doc_001/chunk_000.json",
                "minio://rag-storage/chunks/test_rag_001/test_doc_001/chunk_001.json",
                "minio://rag-storage/chunks/test_rag_001/test_doc_001/chunk_002.json"
            ]
            
            for i, uri in enumerate(sample_uris):
                print(f"\nAccessing: {uri}")
                try:
                    content = minio_store.get_chunk_content(uri)
                    if content:
                        print(f"  ✓ SUCCESS")
                        print(f"  Content preview: {content['content'][:100]}...")
                        print(f"  Metadata: {content.get('metadata', {})}")
                        print(f"  Stored at: {content.get('stored_at', 'unknown')}")
                        
                        # Save readable version to file
                        output_file = f"chunk_{i:03d}_readable.json"
                        with open(output_file, 'w', encoding='utf-8') as f:
                            json.dump(content, f, indent=2, ensure_ascii=False)
                        print(f"  Saved readable version: {output_file}")
                    else:
                        print(f"  ✗ CONTENT NOT FOUND")
                except Exception as e:
                    print(f"  ✗ ERROR: {e}")
                    
                if i >= 2:  # Only show first 3
                    break
            
            print("\n4. DIRECT MINIO CLIENT ACCESS")
            print("-" * 30)
            
            # Show how to use MinIO client directly
            if minio_store.client:
                print("Using MinIO client directly:")
                
                try:
                    # List objects in bucket
                    objects = list(minio_store.client.list_objects(
                        minio_store.bucket_name, 
                        prefix="chunks/test_rag_001/",
                        recursive=True
                    ))
                    
                    print(f"Found {len(objects)} objects in bucket:")
                    for obj in objects[:5]:  # Show first 5
                        print(f"  - {obj.object_name} ({obj.size} bytes)")
                    
                    if len(objects) > 5:
                        print(f"  ... and {len(objects) - 5} more")
                    
                except Exception as e:
                    print(f"Direct listing error: {e}")
            else:
                print("MinIO client not available (using local fallback)")
                
                # Show local fallback location
                local_dir = minio_store.local_fallback_dir / "chunks" / "test_rag_001"
                if local_dir.exists():
                    files = list(local_dir.rglob("*.json"))
                    print(f"Local fallback files: {len(files)}")
                    for file_path in files[:3]:
                        print(f"  - {file_path}")
                        
        else:
            print("\nNo stored content found. Run the pipeline test first:")
            print("  python test_real_rag_pipeline.py")
        
    except Exception as e:
        print(f"Error browsing MinIO content: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    browse_minio_content()