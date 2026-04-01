import json
import argparse
import numpy as np
import pandas as pd
import re
import sys
from pathlib import Path
import joblib

from pymorphy3 import MorphAnalyzer
from sentence_transformers import CrossEncoder
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.retrieval.bm25 import load_index, INDEX_PATH
from src.retrieval.retrieve import retrieve_top
from src.retrieval.encoder import rerank_one, MODEL_RERANK, load_chunks_map
from src.retrieval.triggers import (
    TRIGGER_GROUPS_EN,
    TRIGGER_GROUPS_RU,
    PHRASE_TRIGGER_GROUPS_EN,
    PHRASE_TRIGGER_GROUPS_RU,
)


TRIGGERS_RU = set().union(*map(set, TRIGGER_GROUPS_RU.values()))
TRIGGERS_EN = set().union(*map(set, TRIGGER_GROUPS_EN.values()))

PHRASE_TRIGGERS_RU = [
    p for phrases in PHRASE_TRIGGER_GROUPS_RU.values() for p in phrases
]
PHRASE_TRIGGERS_EN = [
    p for phrases in PHRASE_TRIGGER_GROUPS_EN.values() for p in phrases
]

morph = MorphAnalyzer()



REFUSE_VECTORIZER = None
REFUSE_MODEL = None
REFUSE_THRESHOLD = 0.5
REFUSE_VECTORIZER = None
REFUSE_MODEL = None
REFUSE_THRESHOLD = 0.75

def load_refuse_model(path):
    global REFUSE_VECTORIZER, REFUSE_MODEL, REFUSE_THRESHOLD
    artifact = joblib.load(path)
    REFUSE_VECTORIZER = artifact["vectorizer"]
    REFUSE_MODEL = artifact["model"]
    REFUSE_THRESHOLD = max(float(artifact.get("threshold", 0.75)), 0.75)


def normalize_query(query):
    q = query.lower().replace("ё", "е")
    q = re.sub(r"[^a-zа-я0-9\s]", " ", q)
    tokens = q.split()
    return [morph.parse(tok)[0].normal_form for tok in tokens]


def extract_retrieval_features(reranked_items, k=5):
    scores = [float(d["ce_score"]) for d in reranked_items[:k]]
    while len(scores) < k:
        scores.append(0.0)

    s1, s2, s3, s4, s5 = scores[:5]
    return {
        "top1": s1,
        "top2": s2,
        "top3": s3,
        "top4": s4,
        "top5": s5,
        "mean_top3": float(np.mean([s1, s2, s3])),
        "mean_top5": float(np.mean([s1, s2, s3, s4, s5])),
        "std_top3": float(np.std([s1, s2, s3])),
        "min_top3": float(np.min([s1, s2, s3])),
        "sum_top3": float(s1 + s2 + s3),
        "gap12": float(s1 - s2),
        "gap13": float(s1 - s3),
    }


def refuse_rules(query, lang):
    lemmas = normalize_query(query)
    lemma_set = set(lemmas)
    lang = lang.lower()

    if lang == "ru":
        if lemma_set & TRIGGERS_RU:
            return True
        joined = " ".join(lemmas)
        return any(p in joined for p in PHRASE_TRIGGERS_RU)
    if lemma_set & TRIGGERS_EN:
        return True
    joined = " ".join(lemmas)
    return any(p in joined for p in PHRASE_TRIGGERS_EN)


def refuse(query, lang):
    if REFUSE_MODEL is None or REFUSE_VECTORIZER is None:
        return refuse_rules(query, lang)

    text = f"[{lang.lower()}] {query}"
    x = REFUSE_VECTORIZER.transform([text])
    proba = REFUSE_MODEL.predict_proba(x)[0, 1]
    return (proba >= REFUSE_THRESHOLD) or refuse_rules(query, lang)


def fit_confidence_stats(records):
    good_rows = []
    all_rows = []
    for rec in records:
        reranked_items = rec["reranked_items"]
        feats = extract_retrieval_features(reranked_items)
        all_rows.append(feats)
        rel = rec.get("rel", [])
        if isinstance(rel, str):
            rel = [rel]

        top_ids = [d["id"] for d in reranked_items[:5]]
        hit = any(r in top_ids for r in rel)
        if hit:
            good_rows.append(feats)

    base_rows = good_rows if good_rows else all_rows
    df = pd.DataFrame(base_rows)

    if df.empty:
        return {
            "top1_q20": 0.0,
            "mean_top3_q20": 0.0,
            "gap12_q20": 0.0,
        }
    return {
        "top1_q20": float(df["top1"].quantile(0.20)),
        "mean_top3_q20": float(df["mean_top3"].quantile(0.20)),
        "gap12_q20": float(df["gap12"].quantile(0.20)),
    }


def load_or_fit_confidence_stats(stats_path, calibration_path, bm25, ids, meta, chunks_map, reranker, args):
    stats_path = Path(stats_path)
    if stats_path.exists():
        with open(stats_path, "r", encoding="utf-8") as fin:
            return json.load(fin)
    calibration_records = build_records(
        path=calibration_path,
        bm25=bm25,
        ids=ids,
        meta=meta,
        chunks_map=chunks_map,
        reranker=reranker,
        args=args,
        include_expected_action=False,
    )

    stats = fit_confidence_stats(calibration_records)
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    with open(stats_path, "w", encoding="utf-8") as fout:
        json.dump(stats, fout, ensure_ascii=False, indent=2)
    return stats


def retrieve_again(reranked_items, stats):
    feats = extract_retrieval_features(reranked_items)
    bad_votes = 0
    if feats["top1"] < stats["top1_q20"]:
        bad_votes += 1
    if feats["mean_top3"] < stats["mean_top3_q20"]:
        bad_votes += 1
    if feats["gap12"] < stats["gap12_q20"]:
        bad_votes += 1

    return bad_votes >= 2


def build_records(path, bm25, ids, meta, chunks_map, reranker, args, include_expected_action=False):
    with open(path, "r", encoding="utf-8") as fin:
        items = [json.loads(line) for line in fin]
    records = []
    for obj in items:
        qid = obj["id"]
        query = obj.get("text") or obj.get("query") or obj.get("q")
        lang = obj.get("lang", "ru")
        rel = obj.get("rel", [])
        
        if isinstance(rel, str):
            rel = [rel]
        final_ids, _ = retrieve_top(
            query=query,
            lang=lang,
            bm25=bm25,
            ids=ids,
            meta=meta,
            top_dense=args.top_dense,
            top_bm25=args.top_bm25,
            top_final=args.top_final,
        )
        reranked = rerank_one(
            reranker=reranker,
            query=query,
            final_ids=final_ids,
            chunks_map=chunks_map,
            batch_size=args.batch_size,
        )

        rec = {
            "id": qid,
            "query": query,
            "lang": lang,
            "rel": rel,
            "reranked_items": reranked,
        }

        if include_expected_action:
            rec["expected_action"] = obj.get("expected_action")

        records.append(rec)

    return records



def crag_retrieved(reranked_items, query, lang, stats, bm25, ids, meta, chunks_map, reranker, args):
    if refuse(query, lang):
        return [], "REFUSE", "PROVOCATIVE"

    if retrieve_again(reranked_items, stats):
        return [], "RETRIEVE_AGAIN", "LOW_CONFIDENCE"
    return reranked_items, "ANSWER", "OK"



def dump_line(f, obj):
    f.write(json.dumps(obj, ensure_ascii=False) + "\n")

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--calibration", default="/home/dshhrv666/server1/popatkus-rag-bot/opt/rag/data/sets/all_golden_set.jsonl")
    p.add_argument("--evaluation", default="/home/dshhrv666/server1/popatkus-rag-bot/opt/rag/data/sets/evaluation_golden_set.jsonl")
    p.add_argument("--stats_json", default="/home/dshhrv666/server1/popatkus-rag-bot/opt/rag/data/sets/crag/confidence_stats.json")
    
    # p.add_argument("--refuse_model", default="/home/dshhrv666/server1/popatkus-rag-bot/opt/rag/data/crag/action_eval/refuse-logreg.joblib")
    p.add_argument("--refuse_model", default=None)
    p.add_argument("--top_dense_retry", type=int, default=100)
    p.add_argument("--top_bm25_retry", type=int, default=20)
    p.add_argument("--top_final_retry", type=int, default=20)
    
    p.add_argument("--out", default="/home/dshhrv666/server1/popatkus-rag-bot/opt/rag/data/crag/crag_out.jsonl")
    p.add_argument("--popatkus", default="/home/dshhrv666/server1/popatkus-rag-bot/opt/rag/data/popatkus_all_v5.jsonl")
    p.add_argument("--max_length", type=int, default=512)
    p.add_argument("--batch_size", type=int, default=10)
    p.add_argument("--top_dense", type=int, default=80)
    p.add_argument("--top_bm25", type=int, default=10)
    p.add_argument("--top_final", type=int, default=10)
    args = p.parse_args()
    if args.refuse_model:
        load_refuse_model(args.refuse_model)

    bm25, ids, meta = load_index(INDEX_PATH)
    chunks_map = load_chunks_map(args.popatkus)
    reranker = CrossEncoder(MODEL_RERANK, trust_remote_code=True, max_length=args.max_length)

    stats = load_or_fit_confidence_stats(
        stats_path=args.stats_json,
        calibration_path=args.calibration,
        bm25=bm25,
        ids=ids,
        meta=meta,
        chunks_map=chunks_map,
        reranker=reranker,
        args=args,
    )

    evaluation_records = build_records(
        path=args.evaluation,
        bm25=bm25,
        ids=ids,
        meta=meta,
        chunks_map=chunks_map,
        reranker=reranker,
        args=args,
        include_expected_action=True,
    )

    with open(args.out, "w", encoding="utf-8") as fout:
        for rec in evaluation_records:
            _, action, reason = crag_retrieved(
                reranked_items=rec["reranked_items"],
                query=rec["query"],
                lang=rec["lang"],
                stats=stats,
                bm25=bm25,
                ids=ids,
                meta=meta,
                chunks_map=chunks_map,
                reranker=reranker,
                args=args,
            )

            dump_line(fout, {
                "id": rec["id"],
                "lang": rec["lang"],
                "query": rec["query"],
                "expected_action": rec.get("expected_action"),
                "action": action,
                "reason": reason,
            })


if __name__ == "__main__":
    main()
