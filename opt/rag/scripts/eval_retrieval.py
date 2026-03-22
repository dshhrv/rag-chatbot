import argparse
import sys
import json
import math
from pathlib import Path
import re
import numpy as np

import pandas as pd
import matplotlib.pyplot as plt
import torch
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.retrieval.retrieve import ONLY_ENGLISH
from src.retrieval.rrf import rrf_fuse
from scripts.translate import en2ru, MODEL_NAME_TRANSLATE
from time import perf_counter
from src.retrieval.dense import dense_search, COLLECTION_NAME, MODEL_NAME
from src.retrieval.bm25 import bm25_search, load_index, INDEX_PATH, N_GRAM_SIZE
from src.retrieval.encoder import rerank_one, MODEL_RERANK, load_chunks_map
from src.retrieval.cc import cc_fuse

def dump_line(f, obj):
    f.write(json.dumps(obj, ensure_ascii=False) + "\n")

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


def plot(agg, out_png, metric, title):
    out_png = re.sub("intfloat/", "", str(out_png))
    plt.figure()
    plt.plot(agg["k"], agg[metric], marker="o")
    plt.xlabel("K")
    plt.ylabel(metric)
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_png)


def run_retrieval(mode, golden_path, runs_path, bm25, ids, meta, top_dense, top_bm25,
                  top_final, weights, k0, alpha, mt_tok, mt_model, batch_size, reranker, chunks_map, translate_en=True, debug_translate=False):
    runs_path = Path(runs_path)
    runs_path.parent.mkdir(parents=True, exist_ok=True)
    times_ms = []

    with open(golden_path, "r", encoding="utf-8") as fin, open(runs_path, "w", encoding="utf-8") as fout:
        for line in fin:
            t0 = perf_counter()
            obj = json.loads(line)
            qid = obj["id"]
            lang = obj.get("lang", "ru")
            text = obj["text"]
            if translate_en and lang == "en":
                ttr0 = perf_counter()
                text = en2ru(text, tok=mt_tok, model=mt_model, device="cpu", max_new_tokens=128)
                ttr1 = perf_counter()
                if debug_translate:
                    print("translate_ms", (ttr1 - ttr0) * 1000)
                lang = "ru"
            if mode == "dense":
                ranked = dense_search(coll_name=COLLECTION_NAME, lang=lang, text=text, limit=top_dense, debug=True)
            elif mode == "bm25":
                ngram_n = meta.get("ngram_n", N_GRAM_SIZE)
                ranked = bm25_search(bm25=bm25, ids=ids, query_text=text, lang=lang, top_k=top_bm25, ngram_n=ngram_n)
            elif mode == "rrf":
                ngram_n = meta.get("ngram_n", N_GRAM_SIZE)
                bm25_ranked = bm25_search(bm25=bm25, ids=ids, query_text=text, lang=lang, top_k=top_bm25, ngram_n=ngram_n)
                dense_ranked = dense_search(coll_name=COLLECTION_NAME, lang=lang, text=text, limit=top_dense, debug=False)
                ranked = rrf_fuse(
                    ranked_lists={"dense": dense_ranked, "bm25": bm25_ranked},
                    weights=weights,
                    k=k0,
                    key=lambda r: r["id"],
                    top=top_final,
                )
            elif mode == "cc":
                ngram_n = meta.get("ngram_n", N_GRAM_SIZE)
                bm25_ranked = bm25_search(bm25=bm25, ids=ids, query_text=text, lang=lang, top_k=top_bm25, ngram_n=ngram_n)
                dense_ranked = dense_search(coll_name=COLLECTION_NAME, lang=lang, text=text, limit=top_dense, debug=False)
                weights = {"dense": alpha, "bm25": 1.0 - alpha}
                ranked = cc_fuse(
                    ranked_lists={"dense": dense_ranked, "bm25": bm25_ranked},
                    weights=weights,
                    top=top_final,
                )
            elif mode == "encoder":
                ngram_n = meta.get("ngram_n", N_GRAM_SIZE)
                bm25_ranked = bm25_search(bm25=bm25, ids=ids, query_text=text, lang=lang, top_k=top_bm25, ngram_n=ngram_n)
                dense_ranked = dense_search(coll_name=COLLECTION_NAME, lang=lang, text=text, limit=top_dense, debug=False)
                rrf_fused = rrf_fuse(
                    ranked_lists={"dense": dense_ranked, "bm25": bm25_ranked},
                    weights=weights,
                    k=k0,
                    key=lambda r: r["id"],
                    top=top_final,
                )
                ranked = rerank_one(
                    reranker = reranker, query=text, final_ids=rrf_fused, chunks_map=chunks_map, batch_size=batch_size
                )
            dump_line(fout, {"id": qid, "retrieved": ranked})
            t1 = perf_counter()
            times_ms.append((t1 - t0) * 1000)

    return times_ms


def eval_metrics(golden_path, runs_path, ks):
    rel_by_id = {}
    with open(golden_path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            rel_by_id[obj["id"]] = set(obj.get("rel", []))
    rows = []
    with open(runs_path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            qid = obj["id"]
            ranked_raw = obj["retrieved"]
            if ranked_raw and isinstance(ranked_raw[0], dict):
                ranked = [str(x["id"]) for x in ranked_raw]
            else:
                ranked = [str(x) for x in ranked_raw]

            rel = rel_by_id.get(qid, set())
            for k in ks:
                topk = ranked[:k]
                hit_cnt = sum(1 for d in topk if d in rel)
                rows.append(
                    {
                        "id": qid,
                        "k": k,
                        "recall": hit_cnt / len(rel) if len(rel) != 0 else 0.0,
                        "precision": hit_cnt / k,
                        "hit": 1.0 if hit_cnt > 0 else 0.0,
                        "mrr": mrr_at_k(ranked, rel, k),
                        "ndcg": ndcg_at_k(ranked, rel, k),
                    }
                )
    df = pd.DataFrame(rows)
    agg = df.groupby("k", as_index=False).mean(numeric_only=True)
    return df, agg


def has_en_queries(golden_path):
    with open(golden_path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            if obj.get("lang", "ru") == "en":
                return True
    return False



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--golden", default="data/golden_set.jsonl")
    parser.add_argument("--mode", choices=["dense", "bm25", "rrf", "encoder", "cc"], required=True)
    parser.add_argument("--top-dense", type=int, default=80)
    parser.add_argument("--top-bm25", type=int, default=10)
    parser.add_argument("--top-final", type=int, default=10)
    parser.add_argument("--rrf-k0", type=int, default=15)
    parser.add_argument("--dense-w", type=float, default=2.5)
    parser.add_argument("--bm25-w", type=float, default=0.3)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--ks", default="1,3,5,10,15")
    parser.add_argument("--batch_size", type=int, default=512)
    
    parser.add_argument("--runs", default=None)
    parser.add_argument("--out-csv", default=None)
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--only-english", action="store_true")

    BASE_DIR = Path(__file__).resolve().parents[1]
    POPATKUS  = BASE_DIR / "data" / "popatkus_all_v5.jsonl"
    
    args = parser.parse_args()
    if args.runs is None:
        args.runs = f"data/runs/runs_{args.mode}_{COLLECTION_NAME}_{args.golden}_dense_{args.dense_w}_bm25_{args.bm25_w}_rrfk_{args.rrf_k0}_{args.only_english}.jsonl"
    if args.out_csv is None:
        args.out_csv = f"graph/{args.mode}_{COLLECTION_NAME}_{args.golden}_dense_{args.dense_w}_bm25_{args.bm25_w}_rrfk_{args.rrf_k0}_{args.only_english}/data.csv"
    if args.out_dir is None:
        args.out_dir = f"graph/{args.mode}_{COLLECTION_NAME}_{args.golden}_dense_{args.dense_w}_bm25_{args.bm25_w}_rrfk_{args.rrf_k0}_{args.only_english}"
    
    translate_en = args.only_english and has_en_queries(args.golden)

    mt_tok = None
    mt_model = None
    if translate_en:
        from transformers import MarianTokenizer, MarianMTModel
        mt_tok = MarianTokenizer.from_pretrained(MODEL_NAME_TRANSLATE)
        mt_model = MarianMTModel.from_pretrained(MODEL_NAME_TRANSLATE).to("cpu")
        mt_model.eval()

    
    reranker = None
    chunks_map = None
    if args.mode == "encoder":
        chunks_map = load_chunks_map(POPATKUS)
        from sentence_transformers import CrossEncoder
        reranker = CrossEncoder(MODEL_RERANK, trust_remote_code=True, max_length=512)
    
    bm25, ids, meta = load_index(INDEX_PATH)
    weights = {"dense": args.dense_w, "bm25": args.bm25_w}
    ks = [int(x) for x in args.ks.split(",")]
    t0 = perf_counter()
    times_ms = run_retrieval(
        mode=args.mode,
        golden_path=args.golden,
        runs_path=args.runs,
        bm25=bm25,
        ids=ids,
        meta=meta,
        top_dense=args.top_dense,
        top_bm25=args.top_bm25,
        top_final=args.top_final,
        weights=weights,
        k0=args.rrf_k0,
        alpha = args.alpha,
        mt_tok=mt_tok,
        mt_model=mt_model,
        batch_size=args.batch_size,
        reranker=reranker,
        chunks_map=chunks_map,
        translate_en=translate_en,
        debug_translate=False,
    )
    total_ms = (perf_counter() - t0) * 1000
    print(f"TOTAL_TIME_MS({args.mode}): {total_ms:.3f}")
    p95 = np.percentile(times_ms, 95)
    
    
    _, agg = eval_metrics(args.golden, args.runs, ks)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    with open(f"{args.out_dir}/timing.json", "w", encoding="utf-8") as f:
        nq = sum(1 for _ in open(args.golden, "r", encoding="utf-8"))
        json.dump({
            "mode": args.mode,
            "model": MODEL_NAME,
            "collection": COLLECTION_NAME,
            "n_queries": nq,
            "total_time_ms": total_ms,
            "ms_per_query": total_ms / nq if nq else None,
            "p95": p95,
            "top_dense": args.top_dense,
            "top_bm25": args.top_bm25,
            "top_final": args.top_final,
            "rrf_k0": args.rrf_k0,
            "dense_w": args.dense_w,
            "bm25_w": args.bm25_w,  
        }, f, ensure_ascii=False, indent=2)
    
    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    agg.to_csv(args.out_csv, index=False)
    plot(agg, str(out_dir / f"{MODEL_NAME}_{args.mode}_{N_GRAM_SIZE}_ndcg.png"), "ndcg", f"{MODEL_NAME} - {args.mode}_only: nDCG@K")
    plot(agg, str(out_dir / f"{MODEL_NAME}_{args.mode}_{N_GRAM_SIZE}_recall.png"), "recall", f"{MODEL_NAME} - {args.mode}_only: Recall@K")
    plot(agg, str(out_dir / f"{MODEL_NAME}_{args.mode}_{N_GRAM_SIZE}_mrr.png"), "mrr", f"{MODEL_NAME} - {args.mode}_only: MRR@K")
    plot(agg, str(out_dir / f"{MODEL_NAME}_{args.mode}_{N_GRAM_SIZE}_precision.png"), "precision", f"{MODEL_NAME} - {args.mode}_only: Precision@K")
    plot(agg, str(out_dir / f"{MODEL_NAME}_{args.mode}_{N_GRAM_SIZE}_hit.png"), "hit", f"{MODEL_NAME} - {args.mode}_only: Hit@K")


if __name__ == "__main__":
    main()
