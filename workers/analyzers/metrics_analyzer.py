# workers/analyzers/metrics_analyzer.py

import os
import json
import csv
from difflib import SequenceMatcher

class MetricsAnalyzer:
    def __init__(self, models=("llama3", "mistral7b")):
        self.models = models

    def read_metrics(self, model_key):
        metrics_file = f"metrics/{model_key}/{model_key}_metrics.csv"
        metrics = []
        if not os.path.exists(metrics_file):
            print(f"[WARN] No existe el archivo: {metrics_file}")
            return metrics
        with open(metrics_file, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                row["infer_time_sec"] = float(row["infer_time_sec"])
                metrics.append(row)
        return metrics

    def average_inference_times(self):
        summary = {}
        for model in self.models:
            metrics = self.read_metrics(model)
            if not metrics:
                continue
            avg_time = sum(m["infer_time_sec"] for m in metrics) / len(metrics)
            summary[model] = {"avg_infer_time": avg_time, "runs": len(metrics)}
            print(f"[{model}] Promedio de inferencia: {avg_time:.2f}s, Total runs: {len(metrics)}")
        return summary

    def compare_outputs_for_same_prompt(self, n=5):
        # Lee los últimos n logs y compara outputs para el mismo prompt
        logs = {model: self.get_latest_outputs(model, n) for model in self.models}
        comparisons = []
        for log1 in logs[self.models[0]]:
            for log2 in logs[self.models[1]]:
                if log1["prompt"] == log2["prompt"]:
                    sim = SequenceMatcher(None, log1["output"], log2["output"]).ratio()
                    print(f"Prompt:\n{log1['prompt'][:120]}...\n")
                    print(f"[{self.models[0]}]:\n{log1['output'][:200]}...\n")
                    print(f"[{self.models[1]}]:\n{log2['output'][:200]}...\n")
                    print(f"Similitud de outputs: {sim:.2%}")
                    print("-" * 40)
                    comparisons.append({
                        "prompt": log1["prompt"],
                        f"{self.models[0]}_output": log1["output"],
                        f"{self.models[1]}_output": log2["output"],
                        "similarity": sim
                    })
        return comparisons

    def get_latest_outputs(self, model_key, n=5):
        logs_path = f"logs/{model_key}"
        logs = []
        if not os.path.exists(logs_path):
            return logs
        files = [f for f in os.listdir(logs_path) if f.endswith(".json")]
        files.sort(reverse=True)
        for filename in files[:n]:
            with open(os.path.join(logs_path, filename), encoding="utf-8") as f:
                data = json.load(f)
                logs.append(data)
        return logs

    def analyze(self, params=None):
        # Analiza todo y devuelve un dict resumen (expandible a futuro)
        summary = self.average_inference_times()
        comparisons = self.compare_outputs_for_same_prompt(n=5)
        return {
            "summary": summary,
            "comparisons": comparisons
        }

if __name__ == "__main__":
    analyzer = MetricsAnalyzer(models=("llama3", "mistral7b"))
    analyzer.analyze()
