# test_all_evolved_nodes.py
import time

# Test todos los nodos evolucionados
from langchain_integration.langgraph.nodes.history_reader_node import history_reader_node
from langchain_integration.langgraph.nodes.comparison_node import comparison_node  
from langchain_integration.langgraph.nodes.rubric_generator_node import rubric_generator_node
from langchain_integration.langgraph.nodes.rubric_validator_node import rubric_validator_node
from langchain_integration.langgraph.nodes.summary_node import summary_node

print('🚀 Testing ALL evolved nodes...')

# Estado base del orchestrator exitoso
base_state = {
    'input': 'Write a Python function to sort a list',
    'output': 'def sort_list(lst): return sorted(lst)  # Simple sorting function',
    'task_type': 'code',
    'selected_model': 'deepseek7b',
    'strategy': 'optimized',
    'messages': ['Orchestrator completed successfully'],
    'execution_metrics': {'total_time': 89.05, 'cache_hit': True, 'inference_time': 52.8}
}

# 1. Test History Reader
print('1️⃣ Testing History Reader...')
state = history_reader_node(base_state)
print(f'   ✅ History loaded: {len(state.get("last_output", ""))} chars')

# 2. Test Rubric Generator  
print('2️⃣ Testing Rubric Generator...')
state = rubric_generator_node(state)
print(f'   ✅ Rubrics generated: {len(state.get("analysis_result", ""))} chars')

# 3. Test Rubric Validator
print('3️⃣ Testing Rubric Validator...')
state = rubric_validator_node(state)
validation_meta = state.get('rubric_validation_metadata', {})
print(f'   ✅ Validation completed: {validation_meta.get("criteria_detected", 0)} criterios')
print(f'   📊 Average score: {validation_meta.get("scores_extracted", {}).get("average_score", "N/A")}')

# 4. Test Comparison (usando output vs historial)
print('4️⃣ Testing Comparison...')
state = comparison_node(state)
print(f'   ✅ Comparison done: {len(state.get("comparison_result", ""))} chars')

# 5. Test Summary
print('5️⃣ Testing Summary Generator...')
state = summary_node(state)
print(f'   ✅ Summary: {state.get("final_summary", "")}')

print('🎉 ALL EVOLVED NODES TEST COMPLETED!')
print(f'📋 Final state keys: {list(state.keys())}')
