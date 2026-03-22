def rrf_fuse(ranked_lists, weights, k=15, key=lambda x: x["id"], top=10):
    scores = {}
    best = {}
    for name, items in ranked_lists.items():
        w = weights.get(name, 1.0)
        for rank, it in enumerate(items, start=1):
            doc_id = key(it)
            scores[doc_id] = scores.get(doc_id, 0.0) + w / (k + rank)
            if doc_id not in best:
                best[doc_id] = it
    fused = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    return [best[doc_id] for doc_id, _ in fused[:top]]
    
if __name__ == "__main__":
    rrf_fuse()