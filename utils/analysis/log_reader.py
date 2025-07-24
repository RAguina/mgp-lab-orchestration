# utils/analysis/log_reader.py

import os
import json

logs_path = "logs/llama3"

for filename in os.listdir(logs_path):
    if filename.endswith(".json"):
        with open(os.path.join(logs_path, filename), encoding="utf-8") as f:
            data = json.load(f)
            print(f"Prompt: {data['prompt']}")
            print(f"Output sample: {data['output'][:100]}...")
            print(f"Tiempo de inferencia: {data['infer_time_sec']:.2f}s")
            print("-" * 40)
