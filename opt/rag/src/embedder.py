import json
from sentence_transformers import SentenceTransformer

IN_PATH = "popatkus_all_v5.jsonl"
OUT_PATH = "popatkus_points_5.jsonl"

MODEL_NAME = "intfloat/multilingual-e5-small"
BATCH_SIZE = 8



def batched(iterable, n):
    batch = []
    for x in iterable:
        batch.append(x)
        if len(batch) == n:
            yield batch
            batch = []
    if batch:
        yield batch
     

def query_file(model, in_path: str, out_path: str, batch_size: int):
    with open(in_path, "r", encoding="utf-8") as fin, open(out_path, "w", encoding="utf-8") as fout:
        for lines in batched(fin, batch_size):
            objs = [json.loads(l) for l in lines]
            texts = ["query: " + o["text"] for o in objs]

            vecs = model.encode(
                texts,
                batch_size=batch_size,
                normalize_embeddings=True,
                show_progress_bar=False,
            )

            for o, v in zip(objs, vecs):
                point = {
                    "id": o["id"],
                    "vector": v.tolist(),
                    "payload": {k: o[k] for k in o.keys() if k != "text"} | {"text": o["text"]},
                }
                fout.write(json.dumps(point, ensure_ascii=False) + "\n")
        

def embed_query(model, text: str):
    vec = model.encode(
        ["query: " + text],
        batch_size=1,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    return vec[0]


def embed_passages_file(model, in_path: str, out_path: str, batch_size: int):
    with open(in_path, "r", encoding="utf-8") as fin, open(out_path, "w", encoding="utf-8") as fout:
        for lines in batched(fin, batch_size):
            objs = [json.loads(l) for l in lines]
            texts = ["passage: " + o["text"] for o in objs]

            vecs = model.encode(
                texts,
                batch_size=batch_size,
                normalize_embeddings=True,
                show_progress_bar=False,
            )

            for o, v in zip(objs, vecs):
                point = {
                    "id": o["id"],
                    "vector": v.tolist(),
                    "payload": {k: o[k] for k in o.keys() if k != "text"} | {"text": o["text"]},
                }
                fout.write(json.dumps(point, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["passage", "query"])
    parser.add_argument("--text")
    parser.add_argument("--in", dest="in_path", default=IN_PATH)
    parser.add_argument("--out", dest="out_path", default=OUT_PATH)
    parser.add_argument("--batch", type=int, default=BATCH_SIZE)
    args = parser.parse_args()

    model = SentenceTransformer(MODEL_NAME)

    if args.mode == "passage":
        embed_passages_file(model, args.in_path, args.out_path, args.batch)
    else:
        if args.text:
            v = embed_query(model, args.text)
        else:
            query_file(model, args.in_path, args.out_path, args.batch)


