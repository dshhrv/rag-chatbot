import argparse
import json
import sys
from pathlib import Path
from time import perf_counter
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.agent import agent_router
from src.retrieval.bm25 import bm25_search, N_GRAM_SIZE


def parse_rel(rel_value):
    if type(rel_value) == list:
        return [str(p).strip() for p in rel_value if str(p).strip()]
    return [p.strip() for p in str(rel_value).split(";") if p.strip()] if rel_value else []


def flush_record(records, current, stem):
    if current:
        records.append({
            "id": f"{stem}:{len(records) + 1:04d}",
            "lang": current.get("lang", "ru").strip(),
            "query": current.get("q", "").strip(),
            "rel_raw": current.get("rel", ""),
            "rel": parse_rel(current.get("rel", "")),
            "expected_action": current.get("expected_action", "").strip().upper(),
        })


def load_dataset(path):
    path = Path(path)
    records = []
    
    if path.suffix == ".jsonl":
        for i, line in enumerate(open(path, "r", encoding="utf-8"), 1):
            if not line.strip(): continue
            obj = json.loads(line)
            records.append({
                "id": str(obj.get("id", f"{path.stem}:{i:04d}")),
                "lang": obj.get("lang", "ru"),
                "query": str(obj.get("text") or obj.get("query") or obj.get("q")),
                "rel_raw": obj.get("rel"),
                "rel": parse_rel(obj.get("rel")),
                "expected_action": str(obj.get("expected_action", "")).upper(),
            })
    else:
        current = {}
        for line in open(path, "r", encoding="utf-8"):
            line = line.rstrip("\n")
            if not line.strip():
                flush_record(records, current, path.stem)
                current = {}
            else:
                k, _, v = line.partition(":")
                current[k.strip()] = v.lstrip()
        flush_record(records, current, path.stem)
        
    return records


def collect_ctx_ids(state):
    if getattr(state, "intent", None) == "COMPARISON":
        chunks = getattr(state, "chunks_a", [])[:2] + getattr(state, "chunks_b", [])[:2]
        return list({c["id"]: None for c in chunks if c.get("id")}.keys())
    return [c.get("id") for c in getattr(state, "chunks", [])[:3] if c.get("id")]


def bm25_final_ids(query, lang, top_k, top_final):
    ranked = bm25_search(agent_router.bm25, agent_router.ids, query, lang, top_k, agent_router.meta.get("ngram_n", N_GRAM_SIZE))
    return [item["id"] for item in ranked[:top_final]]


def retrieve_search_bm25_only(state, top_bm25):
    state.chunks = agent_router.rerank_items(state.query, bm25_final_ids(state.query, state.lang, top_bm25, state.top_final))
    state.defs = []
    llm_ok, _ = agent_router.llm_retrieval_check(state.query, [i["id"] for i in state.chunks[:3]], False, 3)
    state.need_retry = (not state.chunks) or agent_router.retrieve_again(state.chunks, agent_router.CONFIDENCE_STATS) or (llm_ok is False)
    state.retrieval_ok = not state.need_retry
    return state


def retrieve_comparison_bm25_only(state, top_bm25):
    state.chunks_a = agent_router.rerank_items(state.entity_a, bm25_final_ids(state.entity_a, state.lang, top_bm25, state.top_final))
    state.chunks_b = agent_router.rerank_items(state.entity_b, bm25_final_ids(state.entity_b, state.lang, top_bm25, state.top_final))
    state.defs = [[], []]
    ctx_ids = list({i["id"]: None for i in state.chunks_a[:2] + state.chunks_b[:2]}.keys())
    llm_ok, _ = agent_router.llm_retrieval_check(state.query, ctx_ids, True, 4)
    state.need_retry = (not state.chunks_a) or (not state.chunks_b) or agent_router.retrieve_again(state.chunks_a, agent_router.CONFIDENCE_STATS) or agent_router.retrieve_again(state.chunks_b, agent_router.CONFIDENCE_STATS) or (llm_ok is False)
    state.retrieval_ok = not state.need_retry
    return state


def execute_flow(state, mode, type_):
    if type_ == "SEARCH":
        state = retrieve_search_bm25_only(state, 10) if mode == "bm25_only" else agent_router.retrieve_search(state)
        state = agent_router.judge_search(state)
        if state.intent == "SEARCH" and not getattr(state, "retrieval_ok", False) and getattr(state, "need_retry", False) and getattr(state, "retry_count", 0) == 1:
            state = retrieve_search_bm25_only(state, 20) if mode == "bm25_only" else agent_router.retry_search(state)
            state = agent_router.judge_search(state)
    elif type_ == "COMPARISON":
        state = retrieve_comparison_bm25_only(state, 10) if mode == "bm25_only" else agent_router.retrieve_comparison(state)
        state = agent_router.judge_comparison(state)
        if state.intent == "COMPARISON" and not getattr(state, "retrieval_ok", False) and getattr(state, "need_retry", False) and getattr(state, "retry_count", 0) == 1:
            state = retrieve_comparison_bm25_only(state, 20) if mode == "bm25_only" else agent_router.retry_comparison(state)
            state = agent_router.judge_comparison(state)
    return state


def run_agent(query, lang="ru", mode="hybrid"):
    state = agent_router.route_query(agent_router.AgentState(query=query, lang=lang))
    init_action = state.intent
    
    if state.intent in {"SEARCH", "COMPARISON"}:
        state = execute_flow(state, mode, state.intent)

    routes = {"SEARCH": "generate_search_answer", "COMPARISON": "generate_comparison_answer", "CLARIFY": "clarify", "REFUSE": "refuse", "ESCALATION": "escalate"}
    
    return {
        "initial_action": init_action, "predicted_action": state.intent, "terminal_route": routes.get(state.intent),
        "retry_count": getattr(state, "retry_count", 0), "retrieval_ok": getattr(state, "retrieval_ok", False),
        "need_retry": getattr(state, "need_retry", False), "escalation_reason": getattr(state, "escalation_reason", None),
        "entity_a": getattr(state, "entity_a", None), "entity_b": getattr(state, "entity_b", None),
        "ctx_ids": collect_ctx_ids(state), "retrieval_backend": mode,
    }


def evaluate(dataset, runs_path, prog, mode):
    Path(runs_path).parent.mkdir(parents=True, exist_ok=True)
    rows, lats = [], []

    with open(runs_path, "w", encoding="utf-8") as f:
        for idx, item in enumerate(dataset, 1):
            t0 = perf_counter()
            res = run_agent(item["query"], item["lang"], mode)
            lat = (perf_counter() - t0) * 1000.0
            lats.append(lat)

            run = {**item, **res, "correct": int(res["predicted_action"] == item["expected_action"]), "latency_ms": round(lat, 3)}
            f.write(json.dumps(run, ensure_ascii=False) + "\n")
            rows.append(run)
            if prog and idx % prog == 0: print(f"processed {idx}/{len(dataset)}")

    df = pd.DataFrame(rows)
    per_label = df.groupby("expected_action", as_index=False).agg(support=("id", "count"), correct=("correct", "sum"))
    per_label["accuracy_percent"] = (per_label["correct"] / per_label["support"] * 100).round(2)

    timing = {
        "n_examples": len(df), "n_correct": int(df["correct"].sum()), "accuracy_percent": round(df["correct"].mean() * 100, 2),
        "avg_latency_ms": round(sum(lats) / len(lats), 3) if lats else 0
    }
    return df, per_label, df[df["correct"] == 0], pd.crosstab(df["expected_action"], df["predicted_action"]), timing


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default=str(ROOT / "data" / "sets" / "agent_set.jsonl"))
    parser.add_argument("--runs", default=None)
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--disable-llm-judge", action="store_true")
    parser.add_argument("--bm25-only", action="store_true")
    parser.add_argument("--progress-every", type=int, default=50)
    args = parser.parse_args()

    dp = Path(args.dataset)
    out_dir = Path(args.out_dir) if args.out_dir else ROOT / "data" / "eval" / f"agent_actions_{dp.stem}"
    runs_path = Path(args.runs) if args.runs else ROOT / "data" / "runs" / f"agent_actions_{dp.stem}.jsonl"

    out_dir.mkdir(parents=True, exist_ok=True)
    
    if args.disable_llm_judge:
        agent_router.llm_retrieval_check = lambda *a, **k: (None, "SKIPPED")

    dataset = load_dataset(dp)
    if args.limit: dataset = dataset[:args.limit]

    df, per_label, errs, conf, timing = evaluate(dataset, runs_path, args.progress_every, "bm25_only" if args.bm25_only else "hybrid")

    print(f"\nOverall accuracy: {timing['accuracy_percent']:.2f}%\n\n{per_label.to_string(index=False)}")

    df.to_csv(out_dir / "action_predictions.csv", index=False)
    per_label.to_csv(out_dir / "action_accuracy_by_label.csv", index=False)
    errs.to_csv(out_dir / "action_errors.csv", index=False)
    conf.to_csv(out_dir / "action_confusion.csv")
    
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump({"dataset": str(dp), "runs": str(runs_path), **timing}, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    main()
