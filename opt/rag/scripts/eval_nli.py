import sys
import json
import csv
import time
from pathlib import Path
from collections import defaultdict

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.llm.nli import verify_answer_simple, verify_answer_with_threshold, verify_answer_decomposed
from src.retrieval.bm25 import load_index, INDEX_PATH
from src.retrieval.retrieve import retrieve_top
from src.llm.promts import PROMT_BASE
from src.retrieval.encoder import load_chunks_map
from llama_cpp import Llama

ONLY_ENGLISH = False
GOLDEN_PATH = ROOT / "data" / "sets" / "golden_50.jsonl"
CHUNKS_PATH = ROOT / "data" / "popatkus_all_v5.jsonl"
OUTPUT_CSV = ROOT / "data" / "nli_latency_stats.csv"
OUTPUT_SCORES_ALL = ROOT / "data" / "nli_scores_all.csv"
OUTPUT_SCORES_NEUTRAL = ROOT / "data" / "nli_scores_neutral.csv"
OUTPUT_SCORES_CONTRADICTION = ROOT / "data" / "nli_scores_contradiction.csv"
MODEL_PATH = ROOT / "models" / "qwen2_5_3b_q4_k_m.gguf"

bm25, ids, meta = load_index(INDEX_PATH)
chunks_map = load_chunks_map(CHUNKS_PATH) if Path(CHUNKS_PATH).exists() else {}

llm_model = Llama(
    model_path=str(MODEL_PATH),
    n_ctx=3072,
    n_threads=4,
    n_batch=512,
    n_ubatch=512,
    verbose=False,
    use_mlock=True,
)

def get_chunk_text(chunk):
    if isinstance(chunk, dict):
        return chunk.get("text", "")
    return str(chunk)

def build_premise(ctx_ids):
    parts = []
    for cid in ctx_ids:
        chunk = chunks_map.get(cid)
        if not chunk:
            continue
        txt = get_chunk_text(chunk)
        if txt:
            parts.append(txt)
    return " ".join(parts)[:1000]

def generate_answer(query, lang, ctx_ids):
    ctx_ids = ctx_ids[:3]
    premise = build_premise(ctx_ids)
    if not premise.strip():
        return ""
    user_content = f"ВОПРОС:\n{query}\nКОНТЕКСТ:\n{premise}"
    messages = [{"role": "system", "content": PROMT_BASE}, {"role": "user", "content": user_content}]
    res = llm_model.create_chat_completion(messages=messages, temperature=0.0, max_tokens=256, stream=False)
    return res["choices"][0]["message"]["content"].strip()

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
    all_latencies = {"simple": [], "threshold": [], "decomposed": []}
    scores_all = {"simple": [], "threshold": [], "decomposed": []}
    scores_neutral = {"simple": [], "threshold": [], "decomposed": []}
    scores_contradiction = {"simple": [], "threshold": [], "decomposed": []}
    label_counts = {"simple": defaultdict(int), "threshold": defaultdict(int), "decomposed": defaultdict(int)}
    total_count = 0

    with open(GOLDEN_PATH, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            query = item.get("text", "")
            rel_ids = item.get("rel") or item.get("rel_ids") or []
            lang = item.get("lang", "ru")

            if not query or not rel_ids:
                continue

            final_ids, _ = retrieve_top(query, lang, bm25, ids, meta, top_dense=80, top_bm25=10, top_final=10, only_english=ONLY_ENGLISH)
            ctx_ids = final_ids[:3]

            premise = build_premise(ctx_ids)
            if not premise.strip():
                continue

            answer = generate_answer(query, lang, ctx_ids)
            if not answer.strip():
                continue

            r_simple = verify_answer_simple(premise, answer)
            r_thresh = verify_answer_with_threshold(premise, answer)
            r_decomp = verify_answer_decomposed(premise, answer)

            all_latencies["simple"].append(r_simple["latency_ms"])
            all_latencies["threshold"].append(r_thresh["latency_ms"])
            all_latencies["decomposed"].append(r_decomp["latency_ms"])

            for method, res in [("simple", r_simple), ("threshold", r_thresh), ("decomposed", r_decomp)]:
                score = res.get("score")
                label = res["label"]
                if score is not None:
                    scores_all[method].append(score)
                    if label == "neutral":
                        scores_neutral[method].append(score)
                    elif label == "contradiction":
                        scores_contradiction[method].append(score)
                label_counts[method][label] += 1

            total_count += 1

    rows = []
    for method in ["simple", "threshold", "decomposed"]:
        lats = all_latencies[method]
        if not lats:
            continue
        
        neutral_rate = label_counts[method].get("neutral", 0) / total_count * 100 if total_count > 0 else 0
        contradiction_rate = label_counts[method].get("contradiction", 0) / total_count * 100 if total_count > 0 else 0
        entailment_rate = label_counts[method].get("entailment", 0) / total_count * 100 if total_count > 0 else 0
        
        scores = scores_all[method]
        mean_score = sum(scores) / len(scores) if scores else 0
        score_std = (sum((s - mean_score) ** 2 for s in scores) / len(scores)) ** 0.5 if scores else 0
        
        rows.append({
            "method": method,
            "mean_latency_ms": round(sum(lats) / len(lats), 2),
            "p95_latency_ms": calc_percentile(lats, 95),
            "p99_latency_ms": calc_percentile(lats, 99),
            "min_latency_ms": round(min(lats), 2),
            "max_latency_ms": round(max(lats), 2),
            "count": len(lats),
            "neutral_rate_pct": round(neutral_rate, 2),
            "contradiction_rate_pct": round(contradiction_rate, 2),
            "entailment_rate_pct": round(entailment_rate, 2),
            "mean_score": round(mean_score, 3),
            "score_std": round(score_std, 3),
            "min_score": round(min(scores), 3) if scores else 0,
            "max_score": round(max(scores), 3) if scores else 0
        })

    with open(OUTPUT_CSV, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "method", "mean_latency_ms", "p95_latency_ms", "p99_latency_ms",
            "min_latency_ms", "max_latency_ms", "count",
            "neutral_rate_pct", "contradiction_rate_pct", "entailment_rate_pct",
            "mean_score", "score_std", "min_score", "max_score"
        ])
        writer.writeheader()
        writer.writerows(rows)

    def save_scores(filepath, scores_dict):
        rows = []
        for method in ["simple", "threshold", "decomposed"]:
            for score in scores_dict[method]:
                rows.append({"method": method, "score": round(score, 4)})
        with open(filepath, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["method", "score"])
            writer.writeheader()
            writer.writerows(rows)

    save_scores(OUTPUT_SCORES_ALL, scores_all)
    save_scores(OUTPUT_SCORES_NEUTRAL, scores_neutral)
    save_scores(OUTPUT_SCORES_CONTRADICTION, scores_contradiction)

if __name__ == "__main__":
    run_benchmark()