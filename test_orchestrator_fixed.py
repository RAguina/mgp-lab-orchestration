from langchain_integration.langgraph.routing_agent import run_orchestrator
import time
import os

# Fix encoding para Windows (opcional, pero no molesta)
os.environ["PYTHONIOENCODING"] = "utf-8"

print("🎬 Testing FULL orchestrator with encoding fix...")
start_time = time.time()

try:
    result = run_orchestrator("Write a Python function to sort a list")

    elapsed = time.time() - start_time
    print(f"⏱️ Total orchestrator time: {elapsed:.2f}s")

    output = result.get("output", "")
    print(f"📊 Output length: {len(output)} chars")
    print(f"📝 First 200 chars: {output[:200]}...")

    print(f"🎯 Final model used: {result.get('selected_model')}")
    print(f"📈 Workers executed: {result.get('metrics', {}).get('workersExecuted', 0)}")

    if "Error" not in output:
        print("✅ ORCHESTRATOR WORKING!")
    else:
        print("❌ Still has errors")

except Exception as e:
    print(f"❌ Exception: {e}")
