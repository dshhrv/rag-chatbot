import json
import argparse
from sentence_transformers import CrossEncoder
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.retrieval.bm25 import load_index, INDEX_PATH
from src.retrieval.retrieve import retrieve_top

MODEL_RERANK = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"

def dump_line(f, obj):
    f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def load_chunks_map(jsonl_path):
    m = {}
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            o = json.loads(line)
            m[o["id"]] = o.get("text", "")
    return m


def rerank_one(reranker, query, final_ids, chunks_map=512, batch_size=10):
    candidate_ids = [x["id"] if isinstance(x, dict) else x for x in final_ids]
    cand_texts = [chunks_map.get(i, "") for i in candidate_ids]
    pairs = [(query, t) for t in cand_texts]
    scores = reranker.predict(pairs, batch_size=batch_size, convert_to_numpy=False)
    scores = [float(s) for s in scores]
    items = [{"id": cid, "ce_score": s} for cid, s in zip(candidate_ids, scores)]
    items.sort(key=lambda x: x["ce_score"], reverse=True)
    return items


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--golden", default="/server1/popatkus-rag-bot/opt/rag/data/golden_set.jsonl")
    p.add_argument("--out", default="/server1/popatkus-rag-bot/opt/rag/data/rerank/rerank_out.jsonl")
    p.add_argument("--popatkus", default="/server1/popatkus-rag-bot/opt/rag/data/popatkus_all_v5.jsonl")

    p.add_argument("--max_length", type=int, default=512)
    p.add_argument("--batch_size", type=int, default=10)
    p.add_argument("--top_dense", type=int, default=80)
    p.add_argument("--top_bm25", type=int, default=10)
    p.add_argument("--top_final", type=int, default=10)
    args = p.parse_args()
    bm25, ids, meta = load_index(INDEX_PATH)
    
    
    chunks_map = load_chunks_map(args.popatkus)
    reranker = CrossEncoder(MODEL_RERANK, trust_remote_code=True, max_length=args.max_length)
    
    with open(args.golden, "r", encoding="utf-8") as fin, open(args.out, "w", encoding="utf-8") as fout:
        for line in fin:
            obj = json.loads(line)
            qid = obj["id"]
            query = obj.get("text") or obj.get("query") or obj.get("q")
            lang = obj.get("lang", "ru")
            rel = obj.get("rel", [])

            final_ids, definitions = retrieve_top(
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

            dump_line(fout, {
                "id": qid,
                "lang": lang,
                "query": query,
                "rel": rel,
                "candidates": final_ids,
                "reranked": reranked,
                "definitions": definitions,
            })

if __name__ == "__main__":
    main()
