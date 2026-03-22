import json
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct

BATCH = 20
URL = "http://localhost:6333"
IN_PATH = "../opt/rag/data/popatkus_points_5_base.jsonl"
COLLECTION_NAME = "popatkus-base"
VEC_LEN = 768


def batched(iterable, n):
    buf = []
    for x in iterable:
        buf.append(x)
        if len(buf) >= n:
            yield buf
            buf = []
    if buf:
        yield buf


def qdrant_upsert(collection_name, in_file, vec_len=VEC_LEN, wait=True, batch=BATCH, url=URL):
    client = QdrantClient(url=url)
    with open(in_file, "r", encoding="utf-8") as f:
        for lines in batched(f, batch):
            objs = [json.loads(l) for l in lines if l.strip()]
            points = []
            for o in objs:
                pid = o["id"]
                vec = o["vector"]
                payload = o.get("payload", {})
                points.append(PointStruct(id=pid, vector=vec, payload=payload))
            client.upsert(collection_name=collection_name, points=points, wait=wait)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--coll-name", dest="collection_name", default=COLLECTION_NAME)
    parser.add_argument("--in-path", dest="in_file", default=IN_PATH)
    parser.add_argument("--vec-len", dest="vec_len", type=int, default=VEC_LEN)
    parser.add_argument("--batch", dest="batch", type=int, default=BATCH)
    parser.add_argument("--url", dest="url", default=URL)
    args = parser.parse_args()

    qdrant_upsert(
        collection_name=args.collection_name,
        in_file=args.in_file,
        vec_len=args.vec_len,
        wait=True,
        batch=args.batch,
        url=args.url,
    )
