import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import json
import re
import os
import time
from typing import Optional
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
            n_ctx=2300,
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
        result = json.loads(cleaned[start:end + 1])
        answer = str(result.get("answer", "") or "").strip()

        if "QUESTION:" in text and answer.lower().startswith(text.split("QUESTION:")[1].split("\n\nCONTEXT:")[0].strip().lower()[:50]):
            parts = re.split(r'\?[\s\[]', answer, maxsplit=1)
            answer = parts[-1].strip() if len(parts) > 1 else answer

        citations = []
        for c in result.get("citations", []) or []:
            c_str = str(c).strip()
            match = re.match(r'^\[([^\]]+)\]', c_str)
            if match:
                citations.append(match.group(1).split('.')[0].strip())
            else:
                citations.append(c_str.split('.')[0].strip())

        return {
            "answer": answer,
            "citations": citations,
            "found": bool(result.get("found", bool(answer))),
            "defs": result.get("defs", []),
        }
    except json.JSONDecodeError:
        return None


def _empty_response(answer, lang="ru", source="no_context", found=False, defs=None, citations=None, llm_called=False, llm_latency_ms=0.0):
    return {
        "answer": answer,
        "citations": citations or [],
        "found": found,
        "defs": defs or [],
        "answer_source": source,
        "llm_called": llm_called,
        "llm_latency_ms": round(llm_latency_ms, 2),
    }


def generate_sgr(query, lang, ctx_ids, top_ctx=5):
    terms = detect_terms(query, lang)
    defs = format_definitions(terms, lang)

    if defs and is_definition_query(query):
        citations = [f"Glossary, {term.upper()}" for term in terms]
        return _empty_response(
            answer="\n".join(defs),
            lang=lang,
            source="glossary_fast",
            found=True,
            defs=defs,
            citations=citations,
            llm_called=False,
        )

    selected = ctx_ids[:top_ctx]
    if not selected:
        answer = "В предоставленных документах нет информации для ответа." if lang == "ru" else "No direct confirmation."
        return _empty_response(answer, lang=lang, source="no_context", found=False, defs=defs)

    clauses_text = build_clauses_text(selected)
    clauses_text = clauses_text[:3500]

    if not clauses_text.strip():
        answer = "В предоставленных документах нет информации для ответа." if lang == "ru" else "No direct confirmation."
        return _empty_response(answer, lang=lang, source="empty_context", found=False, defs=defs)

    if defs:
        clauses_text = "ИНФОРМАЦИЯ ИЗ ГЛОССАРИЯ (ОПРЕДЕЛЕНИЯ):\n" + "\n".join(defs) + "\n\nТЕКСТЫ ДОКУМЕНТОВ:\n" + clauses_text
        clauses_text = clauses_text[:4000]

    if not clauses_text.strip():
        return _empty_response("", lang=lang, source="empty_context", found=False, defs=defs)

    from src.llm.promts import PROMPT_SGR_JSON
    system_prompt = PROMPT_SGR_JSON.format(lang=lang)
    user_content = f"QUESTION:\n{query}\n\nCONTEXT:\n{clauses_text}"
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]

    llm = _get_llm()
    grammar = _get_grammar()

    t0 = time.perf_counter()
    raw = llm.create_chat_completion(
        messages=messages,
        temperature=0.0,
        max_tokens=220,
        stream=False,
        grammar=grammar,
    )["choices"][0]["message"]["content"]
    llm_latency_ms = (time.perf_counter() - t0) * 1000

    result = parse_json_from_llm(raw)
    if not result or not isinstance(result, dict):
        return _empty_response("", lang=lang, source="parse_failed", found=False, defs=defs, llm_called=True, llm_latency_ms=llm_latency_ms)

    answer = str(result.get("answer", "") or "").strip()
    citations = [str(cid) for cid in result.get("citations", []) or []]
    found = bool(result.get("found", False))

    return _empty_response(
        answer=answer,
        lang=lang,
        source="llm_sgr",
        found=found,
        defs=defs,
        citations=citations,
        llm_called=True,
        llm_latency_ms=llm_latency_ms,
    )
