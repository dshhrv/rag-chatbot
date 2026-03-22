import optuna
import json
import os
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import math
from src.retrieval.dense import dense_search, COLLECTION_NAME, MODEL_NAME
from src.retrieval.bm25 import bm25_search, load_index, INDEX_PATH, N_GRAM_SIZE
from src.retrieval.rrf import rrf_fuse
from src.retrieval.cc import cc_fuse
from transformers import MarianTokenizer, MarianMTModel
from scripts.translate import en2ru, MODEL_NAME_TRANSLATE

import torch




bm25, ids, meta = load_index(INDEX_PATH)
MAX_DENSE = 100
MAX_BM25 = 100

IN_PATH = "data/golden_set.jsonl"
OUT_PATH = str(ROOT / "data" / "cache" / "cache.jsonl")

TOP_DENSE_CHOICES = [10, 20, 30, 50, 80, 100]
TOP_BM25_CHOICES  = [10, 20, 30, 50, 80, 100]
TOP_FINAL_CHOICES = [10, 20, 30]

RRF_K_CHOICES = [1, 2, 3, 3.5, 4, 4.5, 5, 8, 10, 15, 20, 25, 28, 30, 35, 36, 40, 50, 60, 80, 100]
W_DENSE = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.5, 2.0, 2.5]
W_BM25  = [0.1, 0.2, 0.3, 0.4, 0.5, 1.0, 1.5, 2.0]

OPTIMIZE_METRIC = "both"
OPTIMIZE_K = 10
HIT_MIN = 0.95
MODE = "maximin"

def mrr_at_k(ranked, rel_set, k):
    for i, d in enumerate(ranked[:k], start=1):
        if d in rel_set:
            return 1.0/i
    return 0.0


def ndcg_at_k(ranked, rel_set, k):
    m = min(k, len(rel_set))
    if m == 0:
        return 0.0
    dcg = 0.0
    for i, d in enumerate(ranked[:k], start=1):
        if d in rel_set:
            dcg += 1.0 / math.log2(i + 1)
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, m + 1))
    return dcg / idcg if idcg > 0 else 0.0


def hit_at_k(ranked, rel_set, k):
    return 1.0 if any(d in rel_set for d in ranked[:k]) else 0.0

def recall_at_k(ranked, rel_set, k):
    if not rel_set:
        return 0.0
    return sum(1 for d in ranked[:k] if d in rel_set) / len(rel_set)


def dump_line(f, obj):
    f.write(json.dumps(obj, ensure_ascii=False) + "\n")
    

def load_golden():
    data = []
    with open(IN_PATH, "r", encoding="utf-8") as fin:
        for line in fin:
            obj = json.loads(line)
            q = obj["text"]
            rel = obj.get("rel", [])
            lang = obj.get("lang", "ru")
            data.append((q, rel, lang))
    return data


def build_cache(golden, out_path, only_english=False, tok=None, model=None):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fout:
        for q, rel, lang in golden:
            if only_english and lang == "en":
                q = en2ru(q, tok=tok, model=model, device="cpu", max_new_tokens=128)
                lang = "ru"

            dense_ids = dense_search(text=q, lang=lang, coll_name=COLLECTION_NAME, limit=MAX_DENSE, debug=False)
            bm25_ids  = bm25_search(bm25=bm25, ids=ids, query_text=q, lang=lang, top_k=MAX_BM25, ngram_n=N_GRAM_SIZE)

            dump_line(fout, {
                "q": q,
                "rel": rel,
                "dense": dense_ids,
                "bm25": bm25_ids,
            })



def load_cache(path):
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))
    return data
        
        
def score(cache, top_dense, top_bm25, top_final, alpha, k_metric):
    # weights = {"dense": w_dense, "bm25": w_bm25}
    weights = {"dense": alpha, "bm25": 1.0 - alpha}
    mrr_sum = 0.0
    ndcg_sum = 0.0
    recall_sum = 0.0
    hit_sum = 0.0
    n = 0
    for ex in cache:
        rel_set = set(ex["rel"])
        # dense = [{"id": x} for x in ex["dense"][:top_dense]]
        # bm25l = [{"id": x} for x in ex["bm25"][:top_bm25]]
        dense_items = ex["dense"][:top_dense]
        bm25_items = ex["bm25"][:top_bm25]
        fused = rrf_fuse(
            ranked_lists={"dense": dense, "bm25": bm25l},
            weights=weights,
            k=rrf_k,
            top=top_final,
        )
        # fused = cc_fuse(
        #     ranked_lists={"dense": dense_items, "bm25": bm25_items},
        #     weights=weights,
        #     top=top_final,
        # )
        ranked = [x["id"] for x in fused]
        mrr_sum += mrr_at_k(ranked, rel_set, k_metric)
        ndcg_sum += ndcg_at_k(ranked, rel_set, k_metric)
        recall_sum += recall_at_k(ranked, rel_set, k_metric)
        hit_sum += hit_at_k(ranked, rel_set, k_metric)
        n += 1
    mrr_mean = mrr_sum / n
    ndcg_mean = ndcg_sum / n
    recall_mean = recall_sum / n
    hit_mean = hit_sum / n
    return mrr_mean, ndcg_mean, recall_mean, hit_mean


def objective(trial, cache):
    top_dense = trial.suggest_categorical("top_dense", TOP_DENSE_CHOICES)
    top_bm25  = trial.suggest_categorical("top_bm25", TOP_BM25_CHOICES)
    top_final = trial.suggest_categorical("top_final", TOP_FINAL_CHOICES)
    # rrf_k = trial.suggest_categorical("rrf_k", RRF_K_CHOICES)
    # w_dense = trial.suggest_categorical("w_dense", W_DENSE)
    # w_bm25  = trial.suggest_categorical("w_bm25", W_BM25)
    alpha = trial.suggest_float("alpha", 0.0, 1.0, step=0.01)

    mrr_mean, ndcg_mean, recall_mean, hit_mean = score(
        cache=cache,
        top_dense=top_dense,
        top_bm25=top_bm25,
        top_final=top_final,
        alpha=alpha,
        k_metric=OPTIMIZE_K,
    )

    trial.set_user_attr(f"mrr@{OPTIMIZE_K}", mrr_mean)
    trial.set_user_attr(f"ndcg@{OPTIMIZE_K}", ndcg_mean)
    trial.set_user_attr(f"recall@{OPTIMIZE_K}", recall_mean)
    trial.set_user_attr(f"hit@{OPTIMIZE_K}", hit_mean)
    if OPTIMIZE_METRIC=="both":
        return hit_mean, recall_mean


def combined_score(hit, recall, mode="maximin"):
        if mode == "maximin":
            return min(hit, recall)
        elif mode == "f1":
            return (2 * hit * recall) / (hit + recall + 1e-12)

        
def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--only-english", action="store_true")
    parser.add_argument("--trials", type=int, default=500)
    args = parser.parse_args()
    golden = load_golden()
    cache_path = OUT_PATH
    if args.only_english:
        cache_path = OUT_PATH.replace(".jsonl", "_en2ru.jsonl")
    tok = None
    model = None
    if args.only_english:
        tok = MarianTokenizer.from_pretrained(MODEL_NAME_TRANSLATE)
        model = MarianMTModel.from_pretrained(MODEL_NAME_TRANSLATE).to("cpu")
        model.eval()
    if not os.path.exists(cache_path):
        build_cache(golden, cache_path, only_english=args.only_english, tok=tok, model=model)
    cache = load_cache(cache_path)
    
    
    study = optuna.create_study(
        directions=["maximize", "maximize"],
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=10),
        storage="sqlite:///hpo_rag.db",
        study_name="hpo_rarht",
        load_if_exists=True,
    )
    study.optimize(lambda t: objective(t, cache), n_trials=args.trials)

    pareto = study.best_trials
    feasible = [t for t in pareto if t.values[0] >= HIT_MIN]
    pool = feasible if feasible else pareto
    best = max(pool, key=lambda t: combined_score(t.values[0], t.values[1], MODE))
    hit10, recall10 = best.values
    
    
    print(f"best_values, {COLLECTION_NAME}, model={MODEL_NAME}, only_english={args.only_english}, (hit@{OPTIMIZE_K}, recall@{OPTIMIZE_K}): ({hit10:.6f}, {recall10:.6f})")
    print("best_params:", best.params)

if __name__ == "__main__":
    main()
