import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import pickle
import argparse
from src.retrieval.glossary import make_dict
from src.retrieval.bm25 import load_index
from src.retrieval.retrieve import retrieve_top


BM25_INDEX_PATH = "/opt/rag/data/bm25index.pkl"
POPATKUS_PATH = "/opt/rag/data/popatkus_all_v5.jsonl"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", required=True)
    parser.add_argument("--lang", choices=["ru", "en"], default="ru")
    parser.add_argument("--index-path", default=BM25_INDEX_PATH)
    parser.add_argument("--in-path", default=POPATKUS_PATH)
    args = parser.parse_args()
    make_dict(args.in_path)
    bm25, ids, meta = load_index(args.index_path)
    final_ids, definitions = retrieve_top(
        query=args.query,
        lang=args.lang,
        bm25=bm25,
        ids=ids,
        meta=meta,
        top_each=50,
        top_final=30,
    )
    print("DEFINITIONS:")
    for d in definitions[:10]:
        print("-", d)

    print("\nTOP IDS:")
    for x in final_ids:
        print(x)


if __name__ == "__main__":
    main()