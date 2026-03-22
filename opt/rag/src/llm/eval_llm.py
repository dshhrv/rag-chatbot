import json
import re
import argparse
from collections import defaultdict

import pandas as pd

from client import MODEL, MODEL_TAG


REFUSAL_RU = "В документе нет прямого подтверждения"
REFUSAL_EN = "No direct confirmation"

OUT_PATH_ALL = "data/popatkus_all_v5.jsonl"
BRACKET_ID_RE = re.compile(r"\[([^\]\n]{1,120})\]")


def citation_id(chunk):
    cl = chunk.get("clause_id")
    if cl is not None:
        cl = str(cl).strip()
        if cl:
            return cl
    hp = chunk.get("heading_path")
    hp = ", ".join(str(x).strip() for x in hp if str(x).strip())
    return hp


def load_chunks_map(jsonl_path):
    m = {}
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            o = json.loads(line)
            m[o["id"]] = o
    return m


def clause_to_id(chunks_map):
    d = defaultdict(set)
    for chunk_id, ch in chunks_map.items():
        cid = citation_id(ch)
        if cid:
            d[cid].add(chunk_id)
    return d


def extract_clause_ids(answer, all_clause_ids):
    out = set(BRACKET_ID_RE.findall(answer or ""))
    return {x for x in out if x in all_clause_ids}


def is_refusal(answer):
    a_low = (answer or "").lower()
    return int((REFUSAL_RU.lower() in a_low) or (REFUSAL_EN.lower() in a_low))


def parse_target_refuse(value):
    if value is None:
        return None
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)):
        return int(value != 0)
    s = str(value).strip().lower()
    if s in {"1", "true", "yes"}:
        return 1
    if s in {"0", "false", "no"}:
        return 0
    return None


def llm_metrics(answer, ctx_clause_ids, ctx_ids, rel, clause_to_chunk_ids, all_clause_ids):
    cited = extract_clause_ids(answer, all_clause_ids)
    cite_any = int(bool(cited))

    if cite_any:
        supported = [c for c in cited if c in ctx_clause_ids]
        cite_supported_rate = len(supported) / len(cited)
    else:
        cite_supported_rate = 0.0

    rel_set = set(rel or [])
    cited_chunk_ids = set()
    for cl in cited:
        cited_chunk_ids |= clause_to_chunk_ids.get(cl, set())

    cite_rel_any = int(bool(cited_chunk_ids & rel_set))
    hit_in_ctx = int(bool(set(ctx_ids) & rel_set))
    no_hit = int(not hit_in_ctx)

    abstain_ok = 0
    if no_hit:
        abstain_ok = is_refusal(answer)

    return {
        "cite_any": cite_any,
        "cite_supported_rate": cite_supported_rate,
        "cite_rel_any": cite_rel_any,
        "no_hit_in_ctx": no_hit,
        "abstain_ok": abstain_ok,
        "cited_clause_ids": sorted(cited),
    }


def evaluate(runs_path, out_csv, chunks_path=OUT_PATH_ALL):
    chunks_map = load_chunks_map(chunks_path)
    clause_to_chunk_ids = clause_to_id(chunks_map)

    all_clause_ids = set()
    for ch in chunks_map.values():
        cid = citation_id(ch)
        if cid:
            all_clause_ids.add(cid)

    agg = {
        "n": 0,
        "cite_any": 0,
        "cite_rel_any": 0,
        "sup_sum": 0.0,
        "no_hit": 0,
        "abstain_ok": 0,
        "refuse_eval_n": 0,
        "refuse_tp": 0,
        "refuse_fp": 0,
        "refuse_tn": 0,
        "refuse_fn": 0,
    }
    latencies = []

    with open(runs_path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)

            answer = obj.get("answer", "")
            ctx_ids = obj.get("ctx_ids", [])
            rel = obj.get("rel", [])
            latency_s = obj.get("latency_s")
            if latency_s is not None:
                latencies.append(latency_s)

            ctx_clause_ids = set()
            for chunk_id in ctx_ids:
                chunk = chunks_map.get(chunk_id)
                if chunk is None:
                    continue
                cid_txt = citation_id(chunk)
                if cid_txt:
                    ctx_clause_ids.add(cid_txt)

            m = llm_metrics(
                answer=answer,
                ctx_clause_ids=ctx_clause_ids,
                ctx_ids=ctx_ids,
                rel=rel,
                clause_to_chunk_ids=clause_to_chunk_ids,
                all_clause_ids=all_clause_ids,
            )

            agg["n"] += 1
            agg["cite_any"] += m["cite_any"]
            agg["cite_rel_any"] += m["cite_rel_any"]
            agg["sup_sum"] += m["cite_supported_rate"]

            if m["no_hit_in_ctx"]:
                agg["no_hit"] += 1
                agg["abstain_ok"] += m["abstain_ok"]

            gold_refuse = parse_target_refuse(obj.get("target_refuse"))
            pred_refuse = is_refusal(answer)

            if gold_refuse is not None:
                agg["refuse_eval_n"] += 1
                if gold_refuse == 1 and pred_refuse == 1:
                    agg["refuse_tp"] += 1
                elif gold_refuse == 0 and pred_refuse == 1:
                    agg["refuse_fp"] += 1
                elif gold_refuse == 0 and pred_refuse == 0:
                    agg["refuse_tn"] += 1
                elif gold_refuse == 1 and pred_refuse == 0:
                    agg["refuse_fn"] += 1

    n = max(1, agg["n"])

    avg_latency_s = sum(latencies) / len(latencies) if latencies else None
    if latencies:
        lat_sorted = sorted(latencies)
        idx = int(0.95 * (len(lat_sorted) - 1))
        p95_latency_s = lat_sorted[idx]
    else:
        p95_latency_s = None

    refuse_eval_n = agg["refuse_eval_n"]
    tp = agg["refuse_tp"]
    fp = agg["refuse_fp"]
    tn = agg["refuse_tn"]
    fn = agg["refuse_fn"]

    refusal_accuracy = (tp + tn) / refuse_eval_n if refuse_eval_n else None
    refusal_error_rate = (fp + fn) / refuse_eval_n if refuse_eval_n else None
    false_refusal_rate = fp / (fp + tn) if (fp + tn) else None

    summary = {
        "model": MODEL,
        "model_tag": MODEL_TAG,
        "n": agg["n"],
        "cite_any_rate": agg["cite_any"] / n,
        "cite_rel_any_rate": agg["cite_rel_any"] / n,
        "cite_supported_rate_avg": agg["sup_sum"] / n,
        "abstain_ok_rate_when_no_hit": agg["abstain_ok"] / max(1, agg["no_hit"]),
        "avg_latency_s": avg_latency_s,
        "p95_latency_s": p95_latency_s,
        "refusal_accuracy": refusal_accuracy,
        "refusal_error_rate": refusal_error_rate,
        "false_refusal_rate": false_refusal_rate,
    }

    pd.DataFrame([summary]).to_csv(out_csv, index=False)
    print(summary)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", default=None)
    parser.add_argument("--out-csv", default=None)
    parser.add_argument("--chunks", default=OUT_PATH_ALL)

    args = parser.parse_args()

    if args.runs is None:
        args.runs = f"/home/dshhrv666/server1/popatkus-rag-bot/opt/rag/data/llm/runs_llm/llm_refuse_{MODEL_TAG}.jsonl"
    if args.out_csv is None:
        args.out_csv = f"/home/dshhrv666/server1/popatkus-rag-bot/opt/rag/data/llm/csv/metrics_llm_refuse_{MODEL_TAG}.csv"

    evaluate(args.runs, args.out_csv, args.chunks)