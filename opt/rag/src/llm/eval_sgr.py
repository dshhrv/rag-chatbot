import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import argparse
import csv
import json
import re
import time
from collections import defaultdict
from statistics import mean

from src.llm.sgr import generate_sgr, is_definition_query
from src.llm.client import citation_id
from src.retrieval.bm25 import load_index, INDEX_PATH
from src.retrieval.retrieve import retrieve_top
from src.retrieval.encoder import load_chunks_map

ONLY_ENGLISH = False
DATA_DIR = ROOT / "data"
OUT_DIR = DATA_DIR / "sgr"
RUNS_DIR = DATA_DIR / "sgr"
DEFAULT_RUNS_PATH = RUNS_DIR / "runs_sgr_gemma_gguf.jsonl"
CHUNKS_PATH = DATA_DIR / "popatkus_all_v5.jsonl"

DEFAULT_INPUTS = [
    DATA_DIR / "sets" / "all_golden_set.jsonl",
]

bm25, ids, meta = load_index(INDEX_PATH)
chunks_map = load_chunks_map(CHUNKS_PATH) if CHUNKS_PATH.exists() else {}


def existing_default_input():
    for p in DEFAULT_INPUTS:
        if p.exists():
            return p
    return DEFAULT_INPUTS[0]


def percentile(values, p):
    if not values:
        return 0.0
    s = sorted(values)
    k = (len(s) - 1) * (p / 100.0)
    f = int(k)
    c = min(f + 1, len(s) - 1)
    d = k - f
    return round(s[f] + d * (s[c] - s[f]), 2)


def stats(values, prefix):
    values = [float(v) for v in values if v is not None]
    if not values:
        return {
            f"{prefix}_mean_ms": 0.0,
            f"{prefix}_p50_ms": 0.0,
            f"{prefix}_p95_ms": 0.0,
            f"{prefix}_p99_ms": 0.0,
            f"{prefix}_min_ms": 0.0,
            f"{prefix}_max_ms": 0.0,
        }
    return {
        f"{prefix}_mean_ms": round(mean(values), 2),
        f"{prefix}_p50_ms": percentile(values, 50),
        f"{prefix}_p95_ms": percentile(values, 95),
        f"{prefix}_p99_ms": percentile(values, 99),
        f"{prefix}_min_ms": round(min(values), 2),
        f"{prefix}_max_ms": round(max(values), 2),
    }


def norm(x):
    return re.sub(r"\s+", " ", str(x or "").lower().replace("ё", "е").strip())


def chunk_citation_ids(ctx_ids):
    out = set()
    for cid in ctx_ids:
        chunk = chunks_map.get(cid)
        if not chunk:
            continue
        out.add(norm(citation_id(chunk)))
        out.add(norm(cid))
    return out


def normalize_citation(c):
    c = str(c or "").strip()
    c = re.sub(r"^\[|\]$", "", c)
    c = c.split(".")[0].strip()
    return norm(c)


def get_rel_ids(obj):
    rel = obj.get("rel") or obj.get("rel_ids") or obj.get("relevant_ids") or []
    if isinstance(rel, str):
        return [rel]
    if isinstance(rel, list):
        return [str(x) for x in rel]
    return []


def get_query(obj):
    return obj.get("text") or obj.get("q") or obj.get("question") or obj.get("query") or ""


def load_items(path, limit=None):
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            q = get_query(obj)
            if not q:
                continue
            items.append(obj)
            if limit and len(items) >= limit:
                break
    return items


def choose_context(query, lang, rel_ids, context_source, top_ctx):
    retrieval_ms = 0.0
    retrieved_ids = []
    if context_source == "gold":
        return rel_ids[:top_ctx], [], retrieval_ms

    t0 = time.perf_counter()
    final_ids, _ = retrieve_top(
        query=query,
        lang=lang,
        bm25=bm25,
        ids=ids,
        meta=meta,
        top_dense=80,
        top_bm25=10,
        top_final=max(10, top_ctx),
        only_english=ONLY_ENGLISH,
    )
    retrieval_ms = (time.perf_counter() - t0) * 1000
    retrieved_ids = [str(x) for x in final_ids]
    return retrieved_ids[:top_ctx], retrieved_ids, retrieval_ms


def evaluate_one(obj, idx, args):
    query = get_query(obj)
    lang = obj.get("lang") or ("en" if re.search(r"[a-zA-Z]", query) and not re.search(r"[а-яА-ЯёЁ]", query) else "ru")
    rel_ids = get_rel_ids(obj)

    ctx_ids, retrieved_ids, retrieval_ms = choose_context(query, lang, rel_ids, args.context_source, args.top_ctx)

    t0 = time.perf_counter()
    res = generate_sgr(query, lang, ctx_ids, top_ctx=args.top_ctx)
    generation_ms = (time.perf_counter() - t0) * 1000

    citations = [str(c) for c in res.get("citations", []) or []]
    allowed_ctx_citations = chunk_citation_ids(ctx_ids)
    rel_citations = chunk_citation_ids(rel_ids)
    normalized_citations = [normalize_citation(c) for c in citations]

    valid_citations = 0
    gold_citation_hits = 0
    glossary_citations = 0
    for c in normalized_citations:
        if c.startswith("glossary"):
            glossary_citations += 1
            valid_citations += 1
            continue
        if c in allowed_ctx_citations:
            valid_citations += 1
        if c in rel_citations:
            gold_citation_hits += 1

    found = bool(res.get("found", False))
    answer = str(res.get("answer", "") or "")
    source = str(res.get("answer_source", "unknown"))
    is_glossary = source == "glossary_fast" or (res.get("defs") and is_definition_query(query))

    retrieval_hit = None
    if rel_ids and retrieved_ids:
        rel_set = set(map(str, rel_ids))
        retrieval_hit = 1 if any(x in rel_set for x in retrieved_ids[:args.top_ctx]) else 0

    return {
        "query_id": idx,
        "query": query,
        "answer": answer,
        "lang": lang,
        "context_source": args.context_source,
        "answer_source": source,
        "is_glossary": int(bool(is_glossary)),
        "llm_called": int(bool(res.get("llm_called", False))),
        "found": int(found),
        "answer_len_chars": len(answer),
        "citations_count": len(citations),
        "valid_citations_count": valid_citations,
        "gold_citation_hits": gold_citation_hits,
        "glossary_citations_count": glossary_citations,
        "has_citation": int(len(citations) > 0),
        "all_citations_valid": int(len(citations) > 0 and valid_citations == len(citations)),
        "has_gold_citation_hit": int(gold_citation_hits > 0),
        "retrieval_hit_at_top_ctx": retrieval_hit,
        "retrieval_latency_ms": round(retrieval_ms, 2),
        "generation_latency_ms": round(generation_ms, 2),
        "llm_latency_ms": round(float(res.get("llm_latency_ms", 0.0) or 0.0), 2),
        "total_latency_ms": round(retrieval_ms + generation_ms, 2),
        "ctx_ids": "|".join(map(str, ctx_ids)),
        "rel_ids": "|".join(map(str, rel_ids)),
        "citations": "|".join(citations),
    }


def aggregate(rows, group_name):
    count = len(rows)
    if count == 0:
        return None

    def rate(field):
        vals = [r.get(field) for r in rows if r.get(field) is not None]
        if not vals:
            return 0.0
        return round(sum(float(v) for v in vals) / len(vals) * 100, 2)

    citation_total = sum(int(r["citations_count"]) for r in rows)
    valid_citation_total = sum(int(r["valid_citations_count"]) for r in rows)
    gold_citation_total = sum(int(r["gold_citation_hits"]) for r in rows)

    result = {
        "group": group_name,
        "count": count,
        "found_rate_pct": rate("found"),
        "citation_rate_pct": rate("has_citation"),
        "all_citations_valid_rate_pct": rate("all_citations_valid"),
        "citation_validity_pct": round(valid_citation_total / citation_total * 100, 2) if citation_total else 0.0,
        "gold_citation_hit_rate_pct": rate("has_gold_citation_hit"),
        "gold_citation_share_pct": round(gold_citation_total / citation_total * 100, 2) if citation_total else 0.0,
        "retrieval_hit_at_top_ctx_pct": rate("retrieval_hit_at_top_ctx"),
        "llm_called_rate_pct": rate("llm_called"),
        "answer_len_mean_chars": round(mean([r["answer_len_chars"] for r in rows]), 2),
        "answer_len_min_chars": min(r["answer_len_chars"] for r in rows),
        "answer_len_max_chars": max(r["answer_len_chars"] for r in rows),
    }
    result.update(stats([r["retrieval_latency_ms"] for r in rows], "retrieval"))
    result.update(stats([r["generation_latency_ms"] for r in rows], "generation"))
    result.update(stats([r["llm_latency_ms"] for r in rows if r["llm_called"]], "llm_only"))
    result.update(stats([r["total_latency_ms"] for r in rows], "total"))
    return result


def save_csv(path, rows, fieldnames=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    if fieldnames is None:
        fieldnames = list(rows[0].keys())
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def save_runs_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            obj = {
                "id": r.get("query_id"),
                "question": r.get("query", ""),
                "answer": r.get("answer", ""),
                "ctx_ids": str(r.get("ctx_ids", "")).split("|") if r.get("ctx_ids") else [],
                "rel": str(r.get("rel_ids", "")).split("|") if r.get("rel_ids") else [],
                "citations": str(r.get("citations", "")).split("|") if r.get("citations") else [],
                "found": bool(int(r.get("found", 0) or 0)),
                "answer_source": r.get("answer_source", ""),
                "llm_called": bool(int(r.get("llm_called", 0) or 0)),
                "latency_s": round(float(r.get("total_latency_ms", 0.0) or 0.0) / 1000.0, 3),
                "generation_latency_s": round(float(r.get("generation_latency_ms", 0.0) or 0.0) / 1000.0, 3),
                "retrieval_latency_s": round(float(r.get("retrieval_latency_ms", 0.0) or 0.0) / 1000.0, 3),
                "valid_citations_count": int(r.get("valid_citations_count", 0) or 0),
                "citations_count": int(r.get("citations_count", 0) or 0),
            }
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def run(args):
    input_path = Path(args.input) if args.input else existing_default_input()
    items = load_items(input_path, args.limit)
    if args.warmup and items:
        print("Warmup on 1 query. Warmup is not saved.")
        _ = evaluate_one(items[0], 0, args)

    print(f"Running SGR benchmark: {len(items)} rows | input={input_path.name} | context={args.context_source}")
    rows = []
    for idx, obj in enumerate(items, 1):
        try:
            r = evaluate_one(obj, idx, args)
            rows.append(r)
            print(
                f"[{idx}/{len(items)}] source={r['answer_source']} | "
                f"gen={r['generation_latency_ms']:.2f}ms | total={r['total_latency_ms']:.2f}ms | "
                f"cites={r['citations_count']} | valid={r['valid_citations_count']} | {r['query'][:70]}..."
            )
        except Exception as e:
            print(f"[{idx}/{len(items)}] FAILED: {e}")

    if not rows:
        print("No rows collected.")
        return

    groups = {
        "all": rows,
        "llm_only": [r for r in rows if r["llm_called"]],
        "non_glossary": [r for r in rows if not r["is_glossary"]],
        "glossary_fast": [r for r in rows if r["answer_source"] == "glossary_fast"],
        "with_citations": [r for r in rows if r["citations_count"] > 0],
        "without_citations": [r for r in rows if r["citations_count"] == 0],
    }

    summary = []
    for name, group_rows in groups.items():
        agg = aggregate(group_rows, name)
        if agg:
            summary.append(agg)

    by_source = []
    for source in sorted(set(r["answer_source"] for r in rows)):
        agg = aggregate([r for r in rows if r["answer_source"] == source], f"source:{source}")
        if agg:
            by_source.append(agg)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    save_csv(OUT_DIR / "gemma_sgr_summary.csv", summary)
    save_csv(OUT_DIR / "gemma_sgr_by_source.csv", by_source)

    filter_report = [
        {"metric": "input_rows", "value": len(items)},
        {"metric": "measured_rows", "value": len(rows)},
        {"metric": "llm_only_rows", "value": len(groups["llm_only"])},
        {"metric": "non_glossary_rows", "value": len(groups["non_glossary"])},
        {"metric": "glossary_fast_rows", "value": len(groups["glossary_fast"])},
        {"metric": "rows_with_citations", "value": len(groups["with_citations"])},
        {"metric": "rows_without_citations", "value": len(groups["without_citations"])},
    ]
    save_csv(OUT_DIR / "gemma_sgr_filter_report.csv", filter_report, fieldnames=["metric", "value"])

    runs_path = Path(args.runs_output) if args.runs_output else DEFAULT_RUNS_PATH


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--top-ctx", type=int, default=5)
    parser.add_argument("--context-source", choices=["retrieved", "gold"], default="retrieved")
    parser.add_argument("--warmup", action="store_true")
    parser.add_argument("--save-raw", action="store_true")
    parser.add_argument("--runs-output", default=None)
    parser.add_argument("--no-save-runs", action="store_true")
    args = parser.parse_args()
    run(args)
