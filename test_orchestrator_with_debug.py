# test_orchestrator_with_debug.py
import signal
import time
from orchestrator_debug_system import start_execution_debug, complete_execution_debug, analyze_latest_execution

def timeout_handler(signum, frame):
    print('⏰ TIMEOUT! Execution taking too long')
    raise TimeoutError('Execution timeout')

def test_orchestrator_with_timeout():
    # Set 3 minute timeout
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(180)  # 3 minutes
    
    try:
        exec_id = start_execution_debug('Write a Python function to sort a list')
        print(f'🚀 Started execution: {exec_id}')
        
        from langchain_integration.langgraph.routing_agent import run_orchestrator
        result = run_orchestrator('Write a Python function to sort a list')
        
        complete_execution_debug(result.get('output', ''), True, None, result.get('metrics', {}))
        
        # Analyze
        analysis = analyze_latest_execution()
        print(f'📊 Analysis: {analysis}')
        
        signal.alarm(0)  # Cancel timeout
        return result
        
    except TimeoutError:
        complete_execution_debug('', False, 'Execution timeout after 3 minutes')
        print('❌ Execution timed out - checking what failed')
        analysis = analyze_latest_execution()
        print(f'📊 Timeout analysis: {analysis}')
        return None
    except Exception as e:
        complete_execution_debug('', False, str(e))
        print(f'❌ Execution failed: {e}')
        return None

if __name__ == '__main__':
    test_orchestrator_with_timeout()
