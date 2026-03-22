from src.retrieval.bm25 import bm25_search, load_index, INDEX_PATH
from src.retrieval.dense import dense_search, COLLECTION_NAME


def cc_fuse(ranked_lists, weights, top=10, key=lambda x: x["id"], score_key=lambda x: x["score"]):
    scores = {}
    best = {}

    for name, items in ranked_lists.items():
        w = weights.get(name, 0.0)
        vals = [score_key(it) for it in items]
        min_q, max_q = min(vals), max(vals)
        diff = max_q - min_q
            
        for it in items:
            doc_id = key(it)
            norm = (score_key(it) - min_q) / diff
            scores[doc_id] = scores.get(doc_id, 0.0) + w * norm
            if doc_id not in best:
                best[doc_id] = it
                
    fused = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    return [best[doc_id] for doc_id, _ in fused[:top]]
    
    
if __name__ == "__main__":
    cc_fuse()