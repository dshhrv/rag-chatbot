import json
import csv
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.llm.nli import verify_answer_simple, verify_answer_with_threshold, verify_answer_decomposed
from src.llm.client import load_chunks_map

GOLDEN_PATH = ROOT / "data" / "sets" / "all_golden_set.jsonl"
POPATKUS_PATH = ROOT / "data" / "popatkus_all_v5.jsonl"
OUTPUT_CSV = ROOT / "data" / "nli_latency_stats.csv"

chunks_map = load_chunks_map(POPATKUS_PATH)
METHODS = {
    "simple": verify_answer_simple,
    "threshold": verify_answer_with_threshold,
    "decomposed": verify_answer_decomposed,
}

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
    all_latencies = {m: [] for m in METHODS}

    with open(GOLDEN_PATH, "r", encoding="utf-8") as in_f:
        for line in in_f:
            item = json.loads(line)
            query = item.get("text", "")
            rel_ids = item.get("rel", [])
            answer = item.get("answer", "Тестовый ответ")
            premise_parts = []
            for cid in rel_ids[:2]:
                chunk = chunks_map.get(cid)
                if chunk and chunk.get("text"):
                    premise_parts.append(chunk["text"])
            premise = " ".join(premise_parts)[:1000]
            if not premise.strip() or not answer.strip():
                continue
            for method_name, method_fn in METHODS.items():
                result = method_fn(premise, answer)
                all_latencies[method_name].append(result["latency_ms"])

    rows = []
    for method_name in METHODS:
        lats = all_latencies[method_name]
        if not lats:
            continue
        mean = round(sum(lats) / len(lats), 2)
        rows.append({
            "method": method_name,
            "mean_ms": mean,
            "p95_ms": calc_percentile(lats, 95),
            "p99_ms": calc_percentile(lats, 99),
            "min_ms": round(min(lats), 2),
            "max_ms": round(max(lats), 2),
            "count": len(lats)
        })

    with open(OUTPUT_CSV, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["method", "mean_ms", "p95_ms", "p99_ms", "min_ms", "max_ms", "count"])
        writer.writeheader()
        writer.writerows(rows)

if __name__ == "__main__":
    run_benchmark()