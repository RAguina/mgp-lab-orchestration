from langchain_integration.langgraph.nodes.resource_monitor_node import resource_monitor_node

print("🔧 Testing fixed resource monitor...")

state = {
    "input": "Write a Python function",
    "task_type": "code",
    "selected_model": "deepseek7b",
    "strategy": "optimized",
    "messages": [],
}

try:
    result = resource_monitor_node(state)
    print(f"✅ Success! Strategy: {result.get('strategy')}")
    print(f"📊 VRAM Status: {result.get('vram_status', '')[:100]}...")
    print(f"📝 Messages: {len(result.get('messages', []))}")
except Exception as e:
    print(f"❌ Still failing: {e}")
