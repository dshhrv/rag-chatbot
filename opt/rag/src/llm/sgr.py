import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import json
import re
from src.llm.client import build_clauses_text, call_ollama
from src.retrieval.glossary import make_dict, detect_terms, format_definitions

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
    defs = format_definitions(detect_terms(query, lang), lang)
    if defs:
        clauses_text = "ИНФОРМАЦИЯ ИЗ ГЛОССАРИЯ (ОПРЕДЕЛЕНИЯ):\n" + "\n".join(defs) + "\n\nТЕКСТЫ ДОКУМЕНТОВ:\n" + clauses_text

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