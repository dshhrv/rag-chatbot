import argparse
import csv
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

def find_root():
    here = Path(__file__).resolve()
    for candidate in [here.parent, *here.parents]:
        if (candidate / 'src').exists() and (candidate / 'data').exists():
            return candidate
    return here.parents[1]
ROOT = find_root()
sys.path.insert(0, str(ROOT))
from src.llm.nli import get_nli_pipeline, verify_answer_decomposed, verify_answer_simple, verify_answer_with_threshold
from src.llm.promts import PROMT_BASE
from src.retrieval.bm25 import INDEX_PATH, load_index
from src.retrieval.encoder import load_chunks_map
from src.retrieval.retrieve import retrieve_top
DEFAULT_GOLDEN_PATH = ROOT / 'data' / 'sets' / 'all_golden_set.jsonl'
DEFAULT_CHUNKS_PATH = ROOT / 'data' / 'popatkus_all_v5.jsonl'
DEFAULT_OUTPUT_DIR = ROOT / 'data' / 'nli'

def detect_lang(text):
    ru = sum((1 for ch in text if 'а' <= ch.lower() <= 'я' or ch.lower() == 'ё'))
    en = sum((1 for ch in text if 'a' <= ch.lower() <= 'z'))
    return 'en' if en > ru else 'ru'

def calc_percentile(data, p):
    if not data:
        return 0.0
    s = sorted(data)
    k = (len(s) - 1) * (p / 100.0)
    f = int(k)
    c = min(f + 1, len(s) - 1)
    d = k - f
    return round(s[f] + d * (s[c] - s[f]), 2)

def mean(data):
    return round(sum(data) / len(data), 2) if data else 0.0

def read_jsonl(path, limit=None):
    rows = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
            if limit is not None and len(rows) >= limit:
                break
    return rows

def get_chunk_text(chunk):
    if isinstance(chunk, dict):
        return str(chunk.get('text') or chunk.get('content') or '')
    return str(chunk or '')

def build_premise(ctx_ids, chunks_map, max_chars):
    parts = []
    for cid in ctx_ids:
        chunk = chunks_map.get(cid)
        if not chunk:
            continue
        text = get_chunk_text(chunk).strip()
        if text:
            parts.append(text)
    return '\n'.join(parts)[:max_chars]

def get_answer_from_item(item):
    for key in ['answer', 'prediction', 'generated_answer', 'hypothesis', 'response']:
        value = item.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ''

def find_model_path(explicit_model_path):
    candidates = []
    if explicit_model_path:
        candidates.append(Path(explicit_model_path))
    for env_key in ['LLAMA_MODEL_PATH', 'GGUF_MODEL_PATH', 'MODEL_PATH']:
        value = os.getenv(env_key)
        if value:
            candidates.append(Path(value))
    for base in [ROOT / 'models', ROOT.parent / 'models', ROOT.parent.parent / 'models']:
        if base.exists():
            candidates.extend(sorted(base.glob('*.gguf')))
    for path in candidates:
        if path.exists() and path.is_file():
            return path
    raise FileNotFoundError('Не найден .gguf файл модели. Передай путь через --model-path или положи модель в opt/rag/models/.')
_llm = None

def get_llm(model_path, n_threads):
    global _llm
    if _llm is None:
        from llama_cpp import Llama
        _llm = Llama(model_path=str(model_path), n_ctx=3072, n_threads=n_threads, n_batch=512, n_ubatch=512, verbose=False, use_mlock=True)
    return _llm

def generate_answer(query, premise, model_path, n_threads, max_tokens):
    if not premise.strip():
        return {'answer': '', 'generation_latency_ms': 0.0}
    llm = get_llm(model_path, n_threads)
    user_content = f'ВОПРОС:\n{query}\n\nКОНТЕКСТ:\n{premise}'
    messages = [{'role': 'system', 'content': PROMT_BASE}, {'role': 'user', 'content': user_content}]
    t0 = time.perf_counter()
    res = llm.create_chat_completion(messages=messages, temperature=0.0, max_tokens=max_tokens, stream=False)
    latency_ms = round((time.perf_counter() - t0) * 1000, 2)
    answer = res['choices'][0]['message']['content'].strip()
    return {'answer': answer, 'generation_latency_ms': latency_ms}

def run_retrieval(query, lang, bm25, ids, meta, top_final, only_english):
    t0 = time.perf_counter()
    final_ids, _ = retrieve_top(query=query, lang=lang, bm25=bm25, ids=ids, meta=meta, top_dense=80, top_bm25=10, top_final=top_final, only_english=only_english)
    latency_ms = round((time.perf_counter() - t0) * 1000, 2)
    return {'ctx_ids': final_ids[:top_final], 'retrieval_latency_ms': latency_ms}

def run_methods(premise, answer, entailment_threshold, contradiction_threshold):
    return {'simple': verify_answer_simple(premise, answer), 'threshold': verify_answer_with_threshold(premise, answer, entailment_threshold=entailment_threshold, contradiction_threshold=contradiction_threshold), 'decomposed': verify_answer_decomposed(premise, answer, entailment_threshold=entailment_threshold, contradiction_threshold=contradiction_threshold)}

def flat_scores(result):
    scores = result.get('scores') or {}
    return {'entailment_score': float(scores.get('entailment', 0.0)), 'neutral_score': float(scores.get('neutral', 0.0)), 'contradiction_score': float(scores.get('contradiction', 0.0))}

def write_csv(path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)

def summarize(detailed_rows):
    by_method = defaultdict(list)
    for row in detailed_rows:
        by_method[row['method']].append(row)
    summary_rows = []
    for method in sorted(by_method):
        rows = by_method[method]
        latencies = [float(r['nli_latency_ms']) for r in rows]
        count = len(rows)
        label_counts = defaultdict(int)
        hallucination_count = 0
        entailment_scores = []
        neutral_scores = []
        contradiction_scores = []
        for row in rows:
            label_counts[row['label']] += 1
            if str(row['is_hallucination']).lower() == 'true':
                hallucination_count += 1
            entailment_scores.append(float(row['entailment_score']))
            neutral_scores.append(float(row['neutral_score']))
            contradiction_scores.append(float(row['contradiction_score']))
        summary_rows.append({'method': method, 'count': count, 'mean_nli_latency_ms': mean(latencies), 'p50_nli_latency_ms': calc_percentile(latencies, 50), 'p95_nli_latency_ms': calc_percentile(latencies, 95), 'p99_nli_latency_ms': calc_percentile(latencies, 99), 'min_nli_latency_ms': round(min(latencies), 2) if latencies else 0.0, 'max_nli_latency_ms': round(max(latencies), 2) if latencies else 0.0, 'entailment_count': label_counts['entailment'], 'neutral_count': label_counts['neutral'], 'contradiction_count': label_counts['contradiction'], 'entailment_rate_pct': round(label_counts['entailment'] / count * 100, 2) if count else 0.0, 'neutral_rate_pct': round(label_counts['neutral'] / count * 100, 2) if count else 0.0, 'contradiction_rate_pct': round(label_counts['contradiction'] / count * 100, 2) if count else 0.0, 'hallucination_count': hallucination_count, 'hallucination_rate_pct': round(hallucination_count / count * 100, 2) if count else 0.0, 'mean_entailment_score': round(sum(entailment_scores) / count, 4) if count else 0.0, 'mean_neutral_score': round(sum(neutral_scores) / count, 4) if count else 0.0, 'mean_contradiction_score': round(sum(contradiction_scores) / count, 4) if count else 0.0})
    return summary_rows

def make_label_count_rows(detailed_rows):
    counts = defaultdict(lambda: defaultdict(int))
    totals = defaultdict(int)
    for row in detailed_rows:
        method = row['method']
        label = row['label']
        counts[method][label] += 1
        totals[method] += 1
    rows = []
    for method in sorted(counts):
        for label in ['entailment', 'neutral', 'contradiction']:
            count = counts[method][label]
            total = totals[method]
            rows.append({'method': method, 'label': label, 'count': count, 'rate_pct': round(count / total * 100, 2) if total else 0.0})
    return rows

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--golden-path', default=str(DEFAULT_GOLDEN_PATH))
    parser.add_argument('--chunks-path', default=str(DEFAULT_CHUNKS_PATH))
    parser.add_argument('--output-dir', default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument('--limit', type=int, default=None)
    parser.add_argument('--top-final', type=int, default=10)
    parser.add_argument('--premise-chars', type=int, default=1000)
    parser.add_argument('--max-tokens', type=int, default=256)
    parser.add_argument('--n-threads', type=int, default=4)
    parser.add_argument('--model-path', default=None)
    parser.add_argument('--entailment-threshold', type=float, default=0.65)
    parser.add_argument('--contradiction-threshold', type=float, default=0.5)
    parser.add_argument('--only-english', action='store_true')
    parser.add_argument('--no-generate', action='store_true')
    parser.add_argument('--warmup', action='store_true')
    return parser.parse_args()

def run_benchmark():
    args = parse_args()
    golden_path = Path(args.golden_path)
    chunks_path = Path(args.chunks_path)
    output_dir = Path(args.output_dir)
    rows = read_jsonl(golden_path, limit=args.limit)
    if not rows:
        return
    bm25, ids, meta = load_index(INDEX_PATH)
    chunks_map = load_chunks_map(chunks_path) if chunks_path.exists() else {}
    model_path = None
    if not args.no_generate:
        model_path = find_model_path(args.model_path)
    if args.warmup:
        get_nli_pipeline()({'text': 'Кошка сидит на ковре.', 'text_pair': 'Кошка находится на ковре.'}, top_k=None)
    detailed_rows = []
    sentence_rows = []
    skipped = 0
    for query_id, item in enumerate(rows, 1):
        query = str(item.get('text') or item.get('query') or '').strip()
        if not query:
            skipped += 1
            continue
        lang = str(item.get('lang') or detect_lang(query))
        retrieval = run_retrieval(query=query, lang=lang, bm25=bm25, ids=ids, meta=meta, top_final=args.top_final, only_english=args.only_english)
        ctx_ids = retrieval['ctx_ids']
        premise = build_premise(ctx_ids[:3], chunks_map, args.premise_chars)
        if not premise.strip():
            skipped += 1
            continue
        answer = get_answer_from_item(item)
        answer_source = 'jsonl'
        generation_latency_ms = 0.0
        if not answer:
            if args.no_generate:
                skipped += 1
                continue
            generated = generate_answer(query=query, premise=premise, model_path=model_path, n_threads=args.n_threads, max_tokens=args.max_tokens)
            answer = generated['answer']
            generation_latency_ms = generated['generation_latency_ms']
            answer_source = 'generated'
        if not answer.strip():
            skipped += 1
            continue
        results = run_methods(premise=premise, answer=answer, entailment_threshold=args.entailment_threshold, contradiction_threshold=args.contradiction_threshold)
        labels_for_print = []
        for method, result in results.items():
            scores = flat_scores(result)
            base_row = {'query_id': query_id, 'method': method, 'query': query, 'lang': lang, 'answer_source': answer_source, 'retrieval_latency_ms': retrieval['retrieval_latency_ms'], 'generation_latency_ms': generation_latency_ms, 'nli_latency_ms': result.get('latency_ms', 0.0), 'pipeline_latency_ms': round(retrieval['retrieval_latency_ms'] + generation_latency_ms + result.get('latency_ms', 0.0), 2), 'label': result.get('label', 'neutral'), 'score': result.get('score', 0.0), 'entailment_score': scores['entailment_score'], 'neutral_score': scores['neutral_score'], 'contradiction_score': scores['contradiction_score'], 'is_hallucination': bool(result.get('is_hallucination', False)), 'n_sentences': result.get('n_sentences', 1), 'answer': answer}
            detailed_rows.append(base_row)
            labels_for_print.append(f"{method}={base_row['label']}")
            if method == 'decomposed':
                for sent in result.get('per_sentence', []):
                    sentence_rows.append({'query_id': query_id, 'sentence_id': sent.get('sentence_id'), 'sentence': sent.get('sentence'), 'label': sent.get('label'), 'score': sent.get('score'), 'entailment_score': sent.get('entailment_score'), 'neutral_score': sent.get('neutral_score'), 'contradiction_score': sent.get('contradiction_score'), 'is_hallucination': sent.get('is_hallucination'), 'latency_ms': sent.get('latency_ms')})
    if not detailed_rows:
        return
    detailed_fields = ['query_id', 'method', 'query', 'lang', 'answer_source', 'retrieval_latency_ms', 'generation_latency_ms', 'nli_latency_ms', 'pipeline_latency_ms', 'label', 'score', 'entailment_score', 'neutral_score', 'contradiction_score', 'is_hallucination', 'n_sentences', 'answer']
    summary_fields = ['method', 'count', 'mean_nli_latency_ms', 'p50_nli_latency_ms', 'p95_nli_latency_ms', 'p99_nli_latency_ms', 'min_nli_latency_ms', 'max_nli_latency_ms', 'entailment_count', 'neutral_count', 'contradiction_count', 'entailment_rate_pct', 'neutral_rate_pct', 'contradiction_rate_pct', 'hallucination_count', 'hallucination_rate_pct', 'mean_entailment_score', 'mean_neutral_score', 'mean_contradiction_score']
    label_count_fields = ['method', 'label', 'count', 'rate_pct']
    sentence_fields = ['query_id', 'sentence_id', 'sentence', 'label', 'score', 'entailment_score', 'neutral_score', 'contradiction_score', 'is_hallucination', 'latency_ms']
    detailed_path = output_dir / 'mini_nli_detailed.csv'
    summary_path = output_dir / 'mini_nli_summary.csv'
    counts_path = output_dir / 'mini_nli_label_counts.csv'
    sentence_path = output_dir / 'mini_nli_sentence_detailed.csv'
    write_csv(detailed_path, detailed_rows, detailed_fields)
    write_csv(summary_path, summarize(detailed_rows), summary_fields)
    write_csv(counts_path, make_label_count_rows(detailed_rows), label_count_fields)
    write_csv(sentence_path, sentence_rows, sentence_fields)
if __name__ == '__main__':
    run_benchmark()
