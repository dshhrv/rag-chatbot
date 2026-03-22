import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import os
import time
import json
import requests

from src.retrieval.bm25 import load_index, INDEX_PATH
from src.retrieval.retrieve import retrieve_top
from src.llm.promts import PROMT_BASE, PROMT_COMPARISON


ONLY_ENGLISH = False
DATA_DIR = ROOT / "data"
RUNS_DIR = DATA_DIR / "llm" / "runs_llm"
OUT_PATH_ALL = DATA_DIR / "popatkus_all_v5.jsonl"
IN_PATH = DATA_DIR / "sets" / "refuse_50.jsonl"

# MODEL = "qwen3:4b-instruct-2507-q4_K_M"
MODEL = "gemma3:4b-it-q4_K_M"
# MODEL = "phi4-mini:3.8b-q4_K_M"
# MODEL = "qwen2.5:3b-instruct-q4_K_M"
# MODEL = "qwen2.5:1.5b-instruct-q4_K_M"

MODEL_TAG = MODEL.replace("/", "_").replace(":", "_")
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434/api/chat")
OUT_PATH = RUNS_DIR / f"llm_refuse_{MODEL_TAG}.jsonl"


bm25, ids, meta = load_index(INDEX_PATH)


def citation_id(chunk):
    cl = chunk.get("clause_id")
    if cl is not None:
        cl = str(cl).strip()
        if cl:
            return cl
    hp = chunk.get("heading_path") or []
    hp = ", ".join(str(x).strip() for x in hp if str(x).strip())
    return hp


def load_chunks_map(jsonl_path):
    m = {}
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            o = json.loads(line)
            m[o["id"]] = o
    return m


def dump_line(f, obj):
    f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def build_clauses_text(ctx_ids):
    clauses_text = ""
    for chunk_id in ctx_ids:
        chunk = chunks_map.get(chunk_id)
        if not chunk:
            continue
        cid_txt = citation_id(chunk)
        clauses_text += f"[{cid_txt}] {chunk.get('text', '')}\n"
    return clauses_text.strip()


def generate_answer(query, lang, ctx_ids=None, promt=PROMT_BASE, top_ctx=3):
    if ctx_ids is None:
        final_ids, _ = retrieve_top(
            query=query,
            lang=lang,
            bm25=bm25,
            ids=ids,
            meta=meta,
            top_dense=80,
            top_bm25=10,
            top_final=10,
            only_english=ONLY_ENGLISH,
        )
        ctx_ids = final_ids[:top_ctx]
    else:
        ctx_ids = ctx_ids[:top_ctx]

    clauses_text = build_clauses_text(ctx_ids)
    if not clauses_text:
        return "В документе нет прямого подтверждения"

    user_content = f"ВОПРОС:\n{query}\nКОНТЕКСТ:\n{clauses_text}"
    messages = [
        {"role": "system", "content": promt},
        {"role": "user", "content": user_content},
    ]

    flush = {
        "model": MODEL,
        "messages": messages,
        "stream": False,
        "options": {
            "temperature": 0.0,
            "num_ctx": 1024,
            "num_predict": 256,
        },
        "keep_alive": "5m",
    }

    resp = requests.post(OLLAMA_URL, json=flush, timeout=600)
    resp.raise_for_status()
    data = resp.json()
    return data["message"]["content"]


if OUT_PATH_ALL.exists():
    chunks_map = load_chunks_map(OUT_PATH_ALL)
else:
    chunks_map = {}


def initialize(in_path=IN_PATH, out_path=OUT_PATH, promt=PROMT_BASE, top_ctx=3):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(in_path, "r", encoding="utf-8") as fin, open(out_path, "a", encoding="utf-8") as fout:
        for line in fin:
            t0 = time.perf_counter()
            obj = json.loads(line)
            query = obj["text"]
            cid = obj["id"]
            rel = obj["rel"]
            lang = obj["lang"]

            final_ids, _ = retrieve_top(
                query,
                lang,
                bm25,
                ids,
                meta,
                top_dense=80,
                top_bm25=10,
                top_final=10,
                only_english=ONLY_ENGLISH,
            )

            ctx_ids = final_ids[:top_ctx]
            answer = generate_answer(
                query=query,
                lang=lang,
                ctx_ids=ctx_ids,
                promt=promt,
                top_ctx=top_ctx,
            )

            elapsed_s = round((time.perf_counter() - t0), 2)

            rec = {
                "id": cid,
                "lang": lang,
                "query": query,
                "rel": rel,
                "ctx_ids": ctx_ids,
                "answer": answer,
                "latency_s": elapsed_s,
            }
            dump_line(fout, rec)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--golden", default=IN_PATH)
    parser.add_argument("--runs", default=None)
    parser.add_argument("--comparison", action="store_true")

    args = parser.parse_args()

    if args.runs is None:
        args.runs = OUT_PATH

    promt = PROMT_COMPARISON if args.comparison else PROMT_BASE
    initialize(in_path=args.golden, out_path=args.runs, promt=promt)