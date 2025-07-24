# utils/analysis/output_comparator.py

import os
import json
from difflib import SequenceMatcher

def get_latest_outputs(model_key, n=5):
    logs_path = f"logs/{model_key}"
    logs = []
    files = [f for f in os.listdir(logs_path) if f.endswith(".json")]
    files.sort(reverse=True)
    for filename in files[:n]:
        with open(os.path.join(logs_path, filename), encoding="utf-8") as f:
            data = json.load(f)
            logs.append(data)
    return logs

llama_logs = get_latest_outputs("llama3", n=3)
mistral_logs = get_latest_outputs("mistral7b", n=3)

for llog in llama_logs:
    for mlog in mistral_logs:
        if llog["prompt"] == mlog["prompt"]:
            print(f"=== Comparando prompt: {llog['prompt'][:80]}... ===")
            print(f"[llama3]\n{llog['output'][:300]}...\n")
            print(f"[mistral7b]\n{mlog['output'][:300]}...\n")
            sim = SequenceMatcher(None, llog["output"], mlog["output"]).ratio()
            print(f"Similitud de outputs: {sim:.2%}")
            print("="*80)
