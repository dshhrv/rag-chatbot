import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import json
import re
import os
from typing import Optional, Dict, Any, List
from llama_cpp import Llama, LlamaGrammar

from src.llm.client import build_clauses_text
from src.retrieval.glossary import make_dict, detect_terms, format_definitions

make_dict()

MODEL_PATH = ROOT / "models" / "qwen2_5_1_5b_q5_k_m.gguf"
GRAMMAR_PATH = Path(__file__).parent / "json_schema.gbnf"

_llm: Optional[Llama] = None
_grammar: Optional[LlamaGrammar] = None


def _get_llm():
    global _llm
    if _llm is None:
        _llm = Llama(
            model_path=str(MODEL_PATH),
            n_ctx=2048,
            n_threads=min(4, os.cpu_count() or 2),
            n_batch=128,
            n_ubatch=128,
            n_gpu_layers=0,
            verbose=False,
            use_mlock=False,
        )
    return _llm

def _get_grammar():
    global _grammar
    if _grammar is None:
        _grammar = LlamaGrammar.from_file(str(GRAMMAR_PATH))
    return _grammar


def is_definition_query(query):
    q = query.lower().replace("ё", "е")
    markers = [
        "что такое",
        "что значит",
        "что означает",
        "определение",
        "расшифруй",
        "расшифровка",
    ]
    return any(marker in q for marker in markers)


def parse_json_from_llm(text):
    if not text:
        return None
    cleaned = re.sub(r"^```(?:json)?\s*", "", text.strip(), flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    try:
        result = json.loads(cleaned[start:end+1])
        
        answer = result.get("answer", "").strip()
        
        if "QUESTION:" in text and answer.lower().startswith(text.split("QUESTION:")[1].split("\n\nCONTEXT:")[0].strip().lower()[:50]):
            parts = re.split(r'\?[\s\[]', answer, maxsplit=1)
            answer = parts[-1].strip() if len(parts) > 1 else answer
        citations = []
        for c in result.get("citations", []):
            c_str = str(c).strip()
            match = re.match(r'^\[([^\]]+)\]', c_str)
            if match:
                citations.append(match.group(1).split('.')[0].strip())
            else:
                citations.append(c_str.split('.')[0].strip())
        
        return {
            "answer": answer,
            "citations": citations,
            "found": result.get("found", bool(answer)),
            "defs": result.get("defs", [])
        }
    except json.JSONDecodeError:
        return None


def generate_sgr(query, lang, ctx_ids, top_ctx=5):
    defs = format_definitions(detect_terms(query, lang), lang)

    if defs and is_definition_query(query):
        terms = detect_terms(query, lang)
        citations = [f"Glossary, {term.upper()}" for term in terms]

        return {
            "answer": "\n".join(defs),
            "citations": citations,
            "found": True,
            "defs": defs,
        }
    selected = ctx_ids[:top_ctx]
    if not selected:
        return {
            "answer": "В предоставленных документах нет информации для ответа.",
            "citations": [],
            "found": False,
            "defs": []
        }
    clauses_text = build_clauses_text(selected)
    clauses_text = clauses_text[:3500]
    if not clauses_text.strip():
        if lang == "ru":
            return {
                "answer": "В предоставленных документах нет информации для ответа.", 
                "citations": [], 
                "found": False, 
                "defs": []
            }
        return {
                "answer": "No direct confirmation.", 
                "citations": [], 
                "found": False, 
                "defs": []
            }
    
    # defs = format_definitions(detect_terms(query, lang), lang)
    if defs:
        clauses_text = "ИНФОРМАЦИЯ ИЗ ГЛОССАРИЯ (ОПРЕДЕЛЕНИЯ):\n" + "\n".join(defs) + "\n\nТЕКСТЫ ДОКУМЕНТОВ:\n" + clauses_text
        clauses_text = clauses_text[:4000]
        
    if not clauses_text.strip():
        return {"answer": "", "citations": [], "found": False, "defs": []}
        
    from src.llm.promts import PROMPT_SGR_JSON
    system_prompt = PROMPT_SGR_JSON.format(lang=lang)
    user_content = f"QUESTION:\n{query}\n\nCONTEXT:\n{clauses_text}"
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]

    llm = _get_llm()
    grammar = _get_grammar()
    
    raw = llm.create_chat_completion(
        messages=messages,
        temperature=0.0,
        max_tokens=220,
        stream=False,
        grammar=grammar,
    )["choices"][0]["message"]["content"]

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