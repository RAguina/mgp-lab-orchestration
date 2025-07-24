# utils/analysis/metrics_reader.py

import csv

metrics_file = "metrics/llama3/llama3_metrics.csv"
metrics = []

with open(metrics_file, encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        row["infer_time_sec"] = float(row["infer_time_sec"])
        metrics.append(row)

metrics.sort(key=lambda x: x["infer_time_sec"], reverse=True)

for m in metrics[:5]:
    print(f"Prompt: {m['prompt']}")
    print(f"Inferencia: {m['infer_time_sec']}s, Modelo: {m['model_key']}")
    print("-" * 30)
