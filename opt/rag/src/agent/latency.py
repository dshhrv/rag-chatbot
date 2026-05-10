import argparse
import csv
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
AGENT_DIR = Path(__file__).resolve().parent

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(AGENT_DIR))

from agent_router import run_agent_timed, run_baseline_timed


GOLDEN_PATH = ROOT / "data" / "sets" / "all_golden_set.jsonl"
OUTPUT_PIPELINE_SUMMARY = ROOT / "data" / "agent" / "latency_pipeline_summary.csv"
OUTPUT_AGENT_SUMMARY = ROOT / "data" / "agent" /"latency_agent_vs_baseline_summary.csv"
OUTPUT_FILTER_REPORT = ROOT / "data" / "agent" /"latency_filter_report.csv"
OUTPUT_RAW = ROOT / "data" / "agent" /"latency_raw_optional.csv"

NON_LLM_SOURCES = {
    "skipped",
    "fallback_context",
    "fallback_empty",
    "static_clarify",
    "static_refuse",
    "static_escalate",
    "static_email",
    "none",
    "unknown",
}


def calc_percentile(data, p):
    if not data:
        return 0.0
    s = sorted(float(x) for x in data)
    k = (len(s) - 1) * (p / 100.0)
    f = int(k)
    c = min(f + 1, len(s) - 1)
    d = k - f
    return round(s[f] + d * (s[c] - s[f]), 2)


def mean(data):
    if not data:
        return 0.0
    return round(sum(float(x) for x in data) / len(data), 2)


def state_get(state, key, default=None):
    if isinstance(state, dict):
        return state.get(key, default)
    return getattr(state, key, default)


def state_timings(state):
    return state_get(state, "timings", {}) or {}


def read_queries(path, limit=None):
    queries = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            if isinstance(obj, str):
                q = obj
            else:
                q = obj.get("text") or obj.get("query") or obj.get("question")
            if q:
                queries.append(str(q))
            if limit is not None and len(queries) >= limit:
                break
    return queries


def get_stage_ms(timings, *names):
    return round(sum(float(timings.get(name, 0.0) or 0.0) for name in names), 2)


def make_record(query_id, query, mode, include_generation, state):
    timings = state_timings(state)

    if mode == "baseline":
        retrieve_ms = get_stage_ms(timings, "baseline_retrieve")
        rerank_ms = get_stage_ms(timings, "baseline_rerank")
        retrieval_total_ms = round(retrieve_ms + rerank_ms, 2)
        generation_ms = get_stage_ms(timings, "baseline_generate")
        verify_ms = 0.0
        route_ms = 0.0
        judge_ms = 0.0
        retry_retrieval_ms = 0.0
        static_ms = 0.0
        agent_logic_ms = 0.0
    else:
        retrieve_ms = get_stage_ms(timings, "retrieve_search", "retrieve_comparison")
        retry_retrieval_ms = get_stage_ms(timings, "retry_search", "retry_comparison")
        rerank_ms = 0.0
        retrieval_total_ms = round(retrieve_ms + retry_retrieval_ms, 2)
        generation_ms = get_stage_ms(timings, "generate_search_answer", "generate_comparison_answer")
        verify_ms = get_stage_ms(timings, "verify_answer")
        route_ms = get_stage_ms(timings, "route_intent")
        judge_ms = get_stage_ms(timings, "judge_search", "judge_comparison")
        static_ms = get_stage_ms(timings, "draft_email", "clarify", "refuse", "escalate")
        agent_logic_ms = get_stage_ms(timings, "agentic_overhead")

    total_ms = round(float(timings.get("total", 0.0) or 0.0), 2)
    if total_ms <= 0:
        total_ms = round(retrieval_total_ms + generation_ms + agent_logic_ms, 2)

    return {
        "query_id": query_id,
        "query": query,
        "mode": mode,
        "include_generation": int(bool(include_generation)),
        "intent": state_get(state, "intent", "UNKNOWN"),
        "lang": state_get(state, "lang", "UNKNOWN"),
        "answer_source": state_get(state, "answer_source", "unknown"),
        "generation_called": int(bool(state_get(state, "generation_called", False))),
        "answer_len_chars": int(state_get(state, "answer_len_chars", 0) or 0),
        "total_ms": total_ms,
        "retrieval_total_ms": retrieval_total_ms,
        "retrieve_ms": retrieve_ms,
        "rerank_ms": rerank_ms,
        "retry_retrieval_ms": retry_retrieval_ms,
        "generation_ms": generation_ms,
        "agent_logic_ms": agent_logic_ms,
        "route_ms": route_ms,
        "judge_ms": judge_ms,
        "verify_ms": verify_ms,
        "static_answer_ms": static_ms,
    }


def is_real_llm_generation(record, min_generation_ms):
    if not int(record.get("generation_called", 0)):
        return False, "generation_not_called"

    source = str(record.get("answer_source", "unknown"))
    if source in NON_LLM_SOURCES:
        return False, f"non_llm_source:{source}"

    generation_ms = float(record.get("generation_ms", 0.0) or 0.0)
    if generation_ms < min_generation_ms:
        return False, f"generation_too_fast_lt_{int(min_generation_ms)}ms"

    return True, "included"


def pair_filter_reason(baseline_record, agent_record, include_generation, min_generation_ms, include_fast_generation):
    if not include_generation:
        return "included"

    if include_fast_generation:
        return "included"

    b_ok, b_reason = is_real_llm_generation(baseline_record, min_generation_ms)
    a_ok, a_reason = is_real_llm_generation(agent_record, min_generation_ms)

    if b_ok and a_ok:
        return "included"

    reasons = []
    if not b_ok:
        reasons.append("baseline_" + b_reason)
    if not a_ok:
        reasons.append("agent_" + a_reason)
    return ";".join(reasons)


def aggregate_records(records):
    grouped = defaultdict(list)
    for record in records:
        grouped[(record["mode"], record["include_generation"])].append(record)

    metric_names = [
        "total_ms",
        "retrieval_total_ms",
        "retrieve_ms",
        "rerank_ms",
        "retry_retrieval_ms",
        "generation_ms",
        "agent_logic_ms",
        "route_ms",
        "judge_ms",
        "verify_ms",
        "static_answer_ms",
    ]

    rows = []
    for (mode, include_generation), group in sorted(grouped.items()):
        row = {
            "mode": mode,
            "include_generation": include_generation,
            "count": len(group),
        }
        for metric in metric_names:
            values = [float(r[metric]) for r in group]
            prefix = metric.replace("_ms", "")
            row[f"{prefix}_mean_ms"] = mean(values)
            row[f"{prefix}_p50_ms"] = calc_percentile(values, 50)
            row[f"{prefix}_p95_ms"] = calc_percentile(values, 95)
            row[f"{prefix}_p99_ms"] = calc_percentile(values, 99)
            row[f"{prefix}_min_ms"] = round(min(values), 2) if values else 0.0
            row[f"{prefix}_max_ms"] = round(max(values), 2) if values else 0.0
        rows.append(row)
    return rows


def aggregate_pairs(pair_rows):
    if not pair_rows:
        return []

    grouped = defaultdict(list)
    for row in pair_rows:
        grouped[row["include_generation"]].append(row)

    metrics = [
        "delta_total_ms",
        "delta_retrieval_total_ms",
        "delta_generation_ms",
        "agent_logic_ms",
        "agent_verify_ms",
        "agent_route_ms",
        "agent_judge_ms",
        "ratio_agent_to_baseline",
    ]

    rows = []
    for include_generation, group in sorted(grouped.items()):
        row = {
            "include_generation": include_generation,
            "count": len(group),
        }
        for metric in metrics:
            values = [float(r[metric]) for r in group]
            prefix = metric.replace("_ms", "")
            suffix = "" if metric.endswith("ratio_agent_to_baseline") else "_ms"
            row[f"{prefix}_mean{suffix}"] = mean(values)
            row[f"{prefix}_p50{suffix}"] = calc_percentile(values, 50)
            row[f"{prefix}_p95{suffix}"] = calc_percentile(values, 95)
            row[f"{prefix}_p99{suffix}"] = calc_percentile(values, 99)
            row[f"{prefix}_min{suffix}"] = round(min(values), 2) if values else 0.0
            row[f"{prefix}_max{suffix}"] = round(max(values), 2) if values else 0.0
        rows.append(row)
    return rows


def make_pair_row(baseline, agent):
    baseline_total = float(baseline["total_ms"])
    agent_total = float(agent["total_ms"])
    return {
        "query_id": agent["query_id"],
        "include_generation": agent["include_generation"],
        "delta_total_ms": round(agent_total - baseline_total, 2),
        "delta_retrieval_total_ms": round(float(agent["retrieval_total_ms"]) - float(baseline["retrieval_total_ms"]), 2),
        "delta_generation_ms": round(float(agent["generation_ms"]) - float(baseline["generation_ms"]), 2),
        "agent_logic_ms": round(float(agent["agent_logic_ms"]), 2),
        "agent_verify_ms": round(float(agent["verify_ms"]), 2),
        "agent_route_ms": round(float(agent["route_ms"]), 2),
        "agent_judge_ms": round(float(agent["judge_ms"]), 2),
        "ratio_agent_to_baseline": round(agent_total / baseline_total, 4) if baseline_total > 0 else 0.0,
    }


def write_csv(path, rows, fieldnames=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        keys = []
        for row in rows:
            for key in row.keys():
                if key not in keys:
                    keys.append(key)
        fieldnames = keys
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def print_pipeline_summary(rows):
    print("\n=== AGGREGATED PIPELINE SUMMARY ===")
    for row in rows:
        gen = "with_generation" if int(row["include_generation"]) else "no_generation"
        print(
            f'{row["mode"]:8} | {gen:16} | N={row["count"]:3} | '
            f'total_mean={row["total_mean_ms"]:9.2f} ms | '
            f'total_p95={row["total_p95_ms"]:9.2f} ms | '
            f'retrieval_mean={row["retrieval_total_mean_ms"]:9.2f} ms | '
            f'generation_mean={row["generation_mean_ms"]:9.2f} ms | '
            f'agent_logic_mean={row["agent_logic_mean_ms"]:9.2f} ms | '
            f'agent_logic_p95={row["agent_logic_p95_ms"]:9.2f} ms'
        )


def print_agent_summary(rows):
    print("\n=== AGGREGATED AGENT VS BASELINE ===")
    for row in rows:
        gen = "with_generation" if int(row["include_generation"]) else "no_generation"
        print(
            f'{gen:16} | N={row["count"]:3} | '
            f'delta_total_mean={row["delta_total_mean_ms"]:9.2f} ms | '
            f'delta_total_p95={row["delta_total_p95_ms"]:9.2f} ms | '
            f'agent_logic_mean={row["agent_logic_mean_ms"]:9.2f} ms | '
            f'agent_logic_p95={row["agent_logic_p95_ms"]:9.2f} ms | '
            f'agent_logic_max={row["agent_logic_max_ms"]:9.2f} ms | '
            f'verify_mean={row["agent_verify_mean_ms"]:9.2f} ms | '
            f'route_mean={row["agent_route_mean_ms"]:9.2f} ms | '
            f'judge_mean={row["agent_judge_mean_ms"]:9.2f} ms'
        )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=30)
    parser.add_argument("--with-generation", action="store_true")
    parser.add_argument("--both", action="store_true")
    parser.add_argument("--warmup", action="store_true")
    parser.add_argument(
        "--min-generation-ms",
        type=float,
        default=5000.0,
        help="With generation enabled, exclude pairs where baseline or agent generation is faster than this. This removes glossary/fallback fast paths from final aggregated stats.",
    )
    parser.add_argument(
        "--include-fast-generation",
        action="store_true",
        help="Do not filter fast glossary/fallback generation from final aggregated stats.",
    )
    parser.add_argument(
        "--save-raw",
        action="store_true",
        help="Optionally save per-query rows to latency_raw_optional.csv. Disabled by default.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    include_generation_values = [False, True] if args.both else [bool(args.with_generation)]
    queries = read_queries(GOLDEN_PATH, limit=args.limit)

    if not queries:
        print(f"No queries found in {GOLDEN_PATH}")
        return

    if args.warmup:
        warmup_q = queries[0]
        print("Warmup on 1 query. Warmup is not saved.")
        for include_generation in include_generation_values:
            try:
                run_baseline_timed(warmup_q, include_generation=include_generation)
                run_agent_timed(warmup_q, include_generation=include_generation)
            except Exception as e:
                print(f"Warmup failed: {e}")

    print(f"Running {len(queries)} measured queries from {GOLDEN_PATH.name}")
    print(f"Generation variants: {include_generation_values}")
    if True in include_generation_values and not args.include_fast_generation:
        print(f"Fast/fallback generation will be excluded from final stats: min_generation_ms={args.min_generation_ms:.0f}")

    included_records = []
    raw_records = []
    pair_rows = []
    filter_counter = Counter()
    total_pairs = Counter()

    for idx, query in enumerate(queries, 1):
        for include_generation in include_generation_values:
            total_pairs[int(include_generation)] += 1
            label = "with generation" if include_generation else "without generation"

            try:
                baseline_state, _ = run_baseline_timed(query, include_generation=include_generation)
                baseline_record = make_record(idx, query, "baseline", include_generation, baseline_state)

                agent_state, _ = run_agent_timed(query, include_generation=include_generation)
                agent_record = make_record(idx, query, "agent", include_generation, agent_state)

                reason = pair_filter_reason(
                    baseline_record,
                    agent_record,
                    include_generation=include_generation,
                    min_generation_ms=args.min_generation_ms,
                    include_fast_generation=args.include_fast_generation,
                )

                raw_records.extend([baseline_record, agent_record])

                if reason == "included":
                    included_records.extend([baseline_record, agent_record])
                    pair_rows.append(make_pair_row(baseline_record, agent_record))
                    filter_counter[(int(include_generation), "included")] += 1
                    status = "included"
                else:
                    filter_counter[(int(include_generation), reason)] += 1
                    status = "excluded: " + reason

                print(
                    f"[{idx}/{len(queries)}] {label:18} | {status} | "
                    f"baseline={baseline_record['total_ms']:.2f}ms "
                    f"agent={agent_record['total_ms']:.2f}ms | "
                    f"b_gen={baseline_record['generation_ms']:.2f}ms/{baseline_record['answer_source']} "
                    f"a_gen={agent_record['generation_ms']:.2f}ms/{agent_record['answer_source']} | "
                    f"Q: {query[:70]}..."
                )

            except KeyboardInterrupt:
                raise
            except Exception as e:
                filter_counter[(int(include_generation), "failed:" + str(e)[:120])] += 1
                print(f"[{idx}/{len(queries)}] {label:18} | FAILED: {e}")

    pipeline_summary = aggregate_records(included_records)
    agent_summary = aggregate_pairs(pair_rows)

    filter_rows = []
    for include_generation in sorted(total_pairs.keys()):
        for (gen, reason), count in sorted(filter_counter.items(), key=lambda x: (x[0][0], x[0][1])):
            if gen != include_generation:
                continue
            filter_rows.append({
                "include_generation": include_generation,
                "reason": reason,
                "count": count,
                "total_candidate_pairs": total_pairs[include_generation],
                "min_generation_ms": args.min_generation_ms if include_generation else 0,
            })

    write_csv(OUTPUT_PIPELINE_SUMMARY, pipeline_summary)
    write_csv(OUTPUT_AGENT_SUMMARY, agent_summary)
    write_csv(OUTPUT_FILTER_REPORT, filter_rows)

    if args.save_raw:
        write_csv(OUTPUT_RAW, raw_records)

    print_pipeline_summary(pipeline_summary)
    print_agent_summary(agent_summary)

if __name__ == "__main__":
    main()
