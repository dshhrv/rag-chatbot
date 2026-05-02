import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from agent_router import run_agent_timed

GOLDEN_PATH = ROOT / "data" / "sets" / "all_golden_set.jsonl"
OUTPUT_CSV = ROOT / "data" / "agent_latency_stats.csv"
OUTPUT_BY_INTENT_CSV = ROOT / "data" / "agent_latency_by_intent.csv"

def calc_percentile(data, p):
    if not data:
        return 0.0
    s = sorted(data)
    k = (len(s) - 1) * (p / 100.0)
    f = int(k)
    c = min(f + 1, len(s) - 1)
    d = k - f
    return round(s[f] + d * (s[c] - s[f]), 2)

def run_benchmark():
    records = []

    with open(GOLDEN_PATH, "r", encoding="utf-8") as f:
        queries = [json.loads(line)["text"] for line in f if line.strip()]

    print(f"Running {len(queries)} queries from {GOLDEN_PATH.name}")
    for i, q in enumerate(queries, 1):
        try:
            state, latency_s = run_agent_timed(q)
            latency_ms = round(latency_s * 1000, 2)
            intent = state.get("intent", "UNKNOWN")
            records.append((intent, latency_ms))
            answer = state.get("answer", "")
            print(f"[{i}/{len(queries)}] Intent: {intent} | Latency: {latency_s:.3f}s | Q: {q[:70]}...")
        except Exception as e:
            print(f"[{i}/{len(queries)}] FAILED: {e}")
            continue

    if not records:
        print("No data collected.")
        return

    all_latencies = [ms for _, ms in records]
    mean_all = round(sum(all_latencies) / len(all_latencies), 2)
    p95_all = calc_percentile(all_latencies, 95)
    p99_all = calc_percentile(all_latencies, 99)
    min_all = round(min(all_latencies), 2)
    max_all = round(max(all_latencies), 2)

    intent_data = defaultdict(list)
    for intent, ms in records:
        intent_data[intent].append(ms)

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_CSV, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value_ms"])
        writer.writerow(["mean", mean_all])
        writer.writerow(["p95", p95_all])
        writer.writerow(["p99", p99_all])
        writer.writerow(["min", min_all])
        writer.writerow(["max", max_all])
        writer.writerow(["count", len(all_latencies)])

    with open(OUTPUT_BY_INTENT_CSV, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["intent", "count", "mean_ms", "p95_ms", "p99_ms", "min_ms", "max_ms"])
        for intent in sorted(intent_data.keys()):
            latencies = intent_data[intent]
            mean = round(sum(latencies) / len(latencies), 2)
            p95 = calc_percentile(latencies, 95)
            p99 = calc_percentile(latencies, 99)
            min_ms = round(min(latencies), 2)
            max_ms = round(max(latencies), 2)
            writer.writerow([intent, len(latencies), mean, p95, p99, min_ms, max_ms])

    print("\n=== TOTAL LATENCY STATS (ms) ===")
    print(f"Mean: {mean_all} | P95: {p95_all} | P99: {p99_all} | Min: {min_all} | Max: {max_all} | N: {len(all_latencies)}")

    print("\n=== LATENCY BY INTENT (ms) ===")
    for intent in sorted(intent_data.keys()):
        latencies = intent_data[intent]
        mean = round(sum(latencies) / len(latencies), 2)
        p95 = calc_percentile(latencies, 95)
        p99 = calc_percentile(latencies, 99)
        print(f"{intent:12} | count={len(latencies):3} | mean={mean:6.2f} | p95={p95:6.2f} | p99={p99:6.2f}")

    print(f"\nSaved overall stats to {OUTPUT_CSV}")
    print(f"Saved per-intent stats to {OUTPUT_BY_INTENT_CSV}")

if __name__ == "__main__":
    run_benchmark()