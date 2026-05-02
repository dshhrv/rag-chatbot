import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import json
import re
import time
import argparse
from src.llm.client import build_clauses_text, call_ollama
from src.retrieval.glossary import make_dict, detect_terms, format_definitions
from eval_llm import evaluate

make_dict()

def parse_json_from_llm(text):
    if not text:
        return None
    cleaned = re.sub(r"^```(?:json)?\s*", "", text.strip(), flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    try:
        return json.loads(cleaned[start:end+1])
    except json.JSONDecodeError:
        return None

def generate_sgr(query, lang, ctx_ids, top_ctx=5):
    selected = ctx_ids[:top_ctx]
    clauses_text = build_clauses_text(selected)
    terms = detect_terms(query, lang)
    defs = format_definitions(terms, lang)
    if defs:
        glossary_blocks = []
        for term, definition in zip(terms, defs):
            glossary_blocks.append(f"[Glossary, {term}]: {definition}")
            
        clauses_text = "ИНФОРМАЦИЯ ИЗ ГЛОССАРИЯ (ОПРЕДЕЛЕНИЯ):\n" + "\n".join(glossary_blocks) + "\n\nТЕКСТЫ ДОКУМЕНТОВ:\n" + clauses_text

    if not clauses_text.strip():
        return {"answer": "", "citations": [], "found": False, "defs": []}
        
    from src.llm.promts import PROMPT_SGR_JSON
    system_prompt = PROMPT_SGR_JSON.format(lang=lang)
    user_content = f"QUESTION:\n{query}\n\nCONTEXT:\n{clauses_text}"
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]
    raw = call_ollama(
        messages=messages,
        temperature=0.0,
        num_ctx=2048,
        num_predict=512,
        format="json",
        timeout=600,
    )

    result = parse_json_from_llm(raw)
    if not result or not isinstance(result, dict):
        return {"answer": "", "citations": [], "found": False, "defs": defs}
        
    answer = str(result.get("answer", "")).strip()
    citations = [str(cid) for cid in result.get("citations", [])]
    found = bool(result.get("found", False))
    
    if found and not citations and not answer.lower().startswith("не найдено"):
        pass
    
    return {
        "answer": answer,
        "citations": citations,
        "found": found,
        "defs": defs,
    }

def process_dataset_and_evaluate(input_path, output_path, out_csv, chunks_path):
    with open(input_path, "r", encoding="utf-8") as fin, \
         open(output_path, "w", encoding="utf-8") as fout:
        
        for line in fin:
            if not line.strip():
                continue
                
            obj = json.loads(line)
            question = obj.get("text", "")
            lang = obj.get("lang", "ru")
            ctx_ids = obj.get("rel", [])
            
            start = time.time()
            sgr_res = generate_sgr(question, lang, ctx_ids)
            latency = time.time() - start
            
            final_answer = sgr_res.get("answer", "")
            citations = sgr_res.get("citations", [])
            if citations:
                final_answer += " " + " ".join([f"[{c}]" for c in citations])
            
            run_obj = {
                "question": question,
                "answer": final_answer,
                "ctx_ids": ctx_ids,
                "rel": obj.get("rel", []),
                "target_refuse": obj.get("target_refuse"),
                "latency_s": latency,
                "found": sgr_res.get("found", False)
            }
            
            fout.write(json.dumps(run_obj, ensure_ascii=False) + "\n")
            fout.flush()
            
    evaluate(runs_path=output_path, out_csv=out_csv, chunks_path=chunks_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=str(ROOT / "data/sets/golden_50.jsonl"))
    parser.add_argument("--output", default=str(ROOT / "data/llm/runs_llm/runs_sgr.jsonl"))
    parser.add_argument("--out-csv", default=str(ROOT / "data/llm/csv/metrics_sgr.csv"))
    parser.add_argument("--chunks", default=str(ROOT / "data/popatkus_all_v5.jsonl"))
    
    args = parser.parse_args()

    process_dataset_and_evaluate(
        input_path=args.input,
        output_path=args.output,
        out_csv=args.out_csv,
        chunks_path=args.chunks
    )