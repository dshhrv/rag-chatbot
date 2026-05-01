import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import json
import re
from src.llm.client import build_clauses_text, call_ollama
from src.llm.glossary_utils import detect_glossary_terms, format_glossary_footer

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
    if not clauses_text.strip():
        return {"answer": "", "citations": [], "found": False}
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
        return {"answer": "", "citations": [], "found": False}
    answer = str(result.get("answer", "")).strip()
    citations = [cid for cid in result.get("citations", []) if cid in selected]
    found = bool(result.get("found", False))
    if found and not citations and not answer.lower().startswith("не найдено"):
        found = False
        answer = "В предоставленных документах нет точной информации для ответа."
    return {
        "answer": answer,
        "citations": citations,
        "found": found,
    }