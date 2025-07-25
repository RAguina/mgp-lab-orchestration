from langchain_integration.langgraph.routing_agent import run_orchestrator
import time

print('🎬 Testing FULL orchestrator with working model...')
start_time = time.time()

result = run_orchestrator('Write a Python function to sort a list')

total_time = time.time() - start_time
print(f'⏱️ Total orchestrator time: {total_time:.2f}s')
print(f'📊 Output length: {len(result.get("output", ""))} chars')
print(f'📝 First 200 chars: {result.get("output", "")[:200]}...')
print(f'🎯 Final model used: {result.get("selected_model")}')
print(f'📈 Metrics: {result.get("metrics", {})}')
