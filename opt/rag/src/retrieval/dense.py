from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from qdrant_client.models import Filter, FieldCondition, MatchValue
import json
from src.embedder import embed_query
import pickle
from time import perf_counter


URL = "http://localhost:6333"
COLLECTION_NAME = "popatkus-base"
# GOLDEN_SET_PATH = "/opt/rag/data/golden_set.jsonl"
MODEL_NAME = "intfloat/multilingual-e5-base"
client = QdrantClient(URL)
model = SentenceTransformer(MODEL_NAME)


def make_lang_filter(lang):
    return Filter(
        must=[FieldCondition(key="lang", match=MatchValue(value=lang))]
    )
    
def search_one(text, lang, coll_name, limit=30):
    flt = make_lang_filter(lang)

    vec = embed_query(model, text)
    
    res = client.query_points(
        collection_name=coll_name,
        query=vec,
        query_filter=flt,
        limit=limit,
    )
    hits = res.points
    
    for h in hits:
        print(h.id, h.score, h.payload.get("text"))
        


# def run_golden_set(coll_name, limit=30, n_questions=5):
#     with open(GOLDEN_SET_PATH, "r", encoding="utf-8") as f:
#         i = 0
#         for line in f:
#             obj = json.loads(line)
#             lang = obj.get("lang")
#             text = obj.get("text")
#             print("\nQ:", text, "| lang:", lang)
#             search_one(text=text, lang=lang, coll_name=coll_name, limit=limit)

#             i += 1
#             if i >= n_questions:
#                 break

def dense_search(coll_name, lang, text, limit=100, debug=False):
    flt = make_lang_filter(lang)
    t0 = perf_counter()
    vec = embed_query(model, text)
    t1 = perf_counter()
    res = client.query_points(
        collection_name=coll_name,
        query=vec,
        query_filter=flt,
        limit=limit,
    )
    hits = res.points
    t2 = perf_counter()
    if debug:
        print("encode_ms", (t1 - t0) * 1000, "search_ms", (t2 - t1) * 1000, "n_hits", len(hits))
    # return [h.id for h in hits]
    return [{"id": h.id, "score": float(h.score)} for h in hits]
        


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--text")
    parser.add_argument("--coll-name", dest="coll_name", default=COLLECTION_NAME)
    parser.add_argument("--golden-set-flag", action="store_true")
    parser.add_argument("--lang", choices=["ru", "en"], default="ru")
    parser.add_argument("--limit", type=int, default=30)
    parser.add_argument("--n-questions", type=int, default=5)
    args = parser.parse_args()
    if args.text:
        search_one(args.text, args.lang, args.coll_name, args.limit)
    else:
        run_golden_set(args.coll_name, args.limit, args.n_questions)
