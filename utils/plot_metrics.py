import pandas as pd, matplotlib.pyplot as plt, pathlib as p

csv_path = p.Path("metrics/mistral7b/mistral7b_metrics.csv")
df = pd.read_csv(csv_path)
df = df.sort_values("timestamp")

# Throughput (tokens / segundo)
df["tok_per_s"] = df["tokens_generated"] / df["infer_time_sec"]

plt.figure()
plt.plot(df["timestamp"], df["tok_per_s"], marker="o")
plt.xticks(rotation=45, ha="right")
plt.xlabel("timestamp"); plt.ylabel("tokens/seg"); plt.title("Mistral-7B throughput")
plt.tight_layout()
plt.show()
