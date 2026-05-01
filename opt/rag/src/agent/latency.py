import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from agent_router import run_agent_timed

GOLDEN_PATH = ROOT / "data" / "sets" / "all_golden_set.jsonl"
OUTPUT_CSV = ROOT / "data" / "agent_latency_stats.csv"

def calc_percentile(data, p):
    if not data: return 0.0
    s = sorted(data)
    k = (len(s) - 1) * (p / 100.0)
    f = int(k)
    c = min(f + 1, len(s) - 1)
    d = k - f
    return round(s[f] + d * (s[c] - s[f]), 2)

def run_benchmark():
    latencies_ms = []
    with open(GOLDEN_PATH, "r", encoding="utf-8") as f:
        queries = [json.loads(line)["text"] for line in f if line.strip()]

    print(f"Running {len(queries)} queries from {GOLDEN_PATH.name}")
    for i, q in enumerate(queries, 1):
        try:
            state, latency_s = run_agent_timed(q)
            latencies_ms.append(round(latency_s * 1000, 2))
            answer = state.answer  # ✅ FIX: доступ через атрибут
            print(f"[{i}/{len(queries)}] Latency: {latency_s:.3f}s | Q: {q[:70]}...")
        except Exception as e:
            print(f"[{i}/{len(queries)}] FAILED: {e}")
            continue

    if not latencies_ms:
        print("No data collected.")
        return

    mean = round(sum(latencies_ms) / len(latencies_ms), 2)
    p95 = calc_percentile(latencies_ms, 95)
    p99 = calc_percentile(latencies_ms, 99)
    min_val = round(min(latencies_ms), 2)
    max_val = round(max(latencies_ms), 2)

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_CSV, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value_ms"])
        writer.writerow(["mean", mean])
        writer.writerow(["p95", p95])
        writer.writerow(["p99", p99])
        writer.writerow(["min", min_val])
        writer.writerow(["max", max_val])
        writer.writerow(["count", len(latencies_ms)])

    print(f"\n=== AGENT LATENCY STATS (ms) ===")
    print(f"Mean: {mean} | P95: {p95} | P99: {p99} | Min: {min_val} | Max: {max_val} | N: {len(latencies_ms)}")
    print(f"Saved to {OUTPUT_CSV}")

if __name__ == "__main__":
    run_benchmark()