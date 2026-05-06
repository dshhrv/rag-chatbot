import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import os
import time
import json
from llama_cpp import Llama, LlamaGrammar

from src.retrieval.bm25 import load_index, INDEX_PATH
from src.retrieval.retrieve import retrieve_top
from src.llm.promts import PROMT_BASE, PROMT_COMPARISON
from src.retrieval.encoder import load_chunks_map

ONLY_ENGLISH = False
DATA_DIR = ROOT / "data"
RUNS_DIR = DATA_DIR / "llm" / "runs_llm"
OUT_PATH_ALL = DATA_DIR / "popatkus_all_v5.jsonl"
IN_PATH = DATA_DIR / "sets" / "sgr_50_set.jsonl"

MODEL_PATH = ROOT / "models" / "qwen2_5_3b_q4_k_m.gguf"
GRAMMAR_PATH = Path(__file__).parent / "json_schema.gbnf"
OUT_PATH = RUNS_DIR / "llm_refuse_qwen2_5_1_5b_q4_k_m.jsonl"

bm25, ids, meta = load_index(INDEX_PATH)
chunks_map = load_chunks_map(OUT_PATH_ALL) if OUT_PATH_ALL.exists() else {}

llm_model = Llama(
    model_path=str(MODEL_PATH),
    n_ctx=3072,
    n_threads=4,
    n_batch=512,
    n_ubatch=512,
    verbose=False,
    use_mlock=True,
)

def citation_id(chunk):
    if isinstance(chunk, str):
        return chunk[:50]
    if not isinstance(chunk, dict):
        return str(chunk)[:50]
    cl = chunk.get("clause_id")
    if cl is not None:
        cl = str(cl).strip()
        if cl:
            return cl
    hp = chunk.get("heading_path") or []
    hp = ", ".join(str(x).strip() for x in hp if str(x).strip())
    return hp

def dump_line(f, obj):
    f.write(json.dumps(obj, ensure_ascii=False) + "\n")

def build_clauses_text(ctx_ids):
    clauses_text = ""
    for chunk_id in ctx_ids:
        chunk = chunks_map.get(chunk_id)
        if not chunk:
            continue
        text = chunk.get("text", "") if isinstance(chunk, dict) else str(chunk)
        cid_txt = citation_id(chunk)
        clauses_text += f"[{cid_txt}] {text}\n"
    return clauses_text.strip()

def call_ollama(messages, temperature=0.0, num_ctx=2048, num_predict=300, format=None):
    if format == "json" and not hasattr(call_ollama, "grammar"):
        from llama_cpp import LlamaGrammar
        call_ollama.grammar = LlamaGrammar.from_file(str(GRAMMAR_PATH))
        print(f"[DEBUG] Grammar loaded: {GRAMMAR_PATH.exists()}")
    
    return llm_model.create_chat_completion(
        messages=messages,
        temperature=temperature,
        max_tokens=num_predict,
        stream=False,
        grammar=call_ollama.grammar if format == "json" else None,
    )["choices"][0]["message"]["content"]

def generate_answer(query, lang, ctx_ids=None, promt=PROMT_BASE, top_ctx=3):
    if ctx_ids is None:
        final_ids, _ = retrieve_top(query, lang, bm25, ids, meta, top_dense=80, top_bm25=10, top_final=10, only_english=ONLY_ENGLISH)
        ctx_ids = final_ids[:top_ctx]
    else:
        ctx_ids = ctx_ids[:top_ctx]
    clauses_text = build_clauses_text(ctx_ids)
    if not clauses_text:
        return "В документе нет прямого подтверждения"
    user_content = f"ВОПРОС:\n{query}\nКОНТЕКСТ:\n{clauses_text}"
    messages = [{"role": "system", "content": promt}, {"role": "user", "content": user_content}]
    return call_ollama(messages, temperature=0.0, num_ctx=1024, num_predict=256)

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
            final_ids, _ = retrieve_top(query, lang, bm25, ids, meta, top_dense=80, top_bm25=10, top_final=10, only_english=ONLY_ENGLISH)
            ctx_ids = final_ids[:top_ctx]
            answer = generate_answer(query, lang, ctx_ids, promt, top_ctx)
            elapsed_s = round((time.perf_counter() - t0), 2)
            rec = {"id": cid, "lang": lang, "query": query, "rel": rel, "ctx_ids": ctx_ids, "answer": answer, "latency_s": elapsed_s}
            dump_line(fout, rec)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--golden", default=IN_PATH)
    parser.add_argument("--runs", default=None)
    parser.add_argument("--comparison", action="store_true")
    args = parser.parse_args()
    if args.runs is None: args.runs = OUT_PATH
    promt = PROMT_COMPARISON if args.comparison else PROMT_BASE
    initialize(in_path=args.golden, out_path=args.runs, promt=promt)