# test_quick_fix.py
import time
from langchain_integration.langgraph.routing_agent import run_orchestrator

print('🚀 Testing orchestrator with QUICK FIX (forced optimized)...')
start = time.time()

try:
    result = run_orchestrator('Write a Python function to sort a list')
    elapsed = time.time() - start
    
    print(f'⏱️ Total time: {elapsed:.2f}s (should be ~60-90s now)')
    print(f'📊 Success: {result.get("output", "") != ""}')
    print(f'📝 Output length: {len(result.get("output", ""))} chars')
    print(f'🎯 Model used: {result.get("selected_model")}')
    
    if elapsed < 120:  # Under 2 minutes
        print('✅ QUICK FIX WORKING - Much faster!')
    else:
        print('❌ Still slow, may need more fixes')
        
except Exception as e:
    elapsed = time.time() - start
    print(f'❌ Failed after {elapsed:.2f}s: {e}')
