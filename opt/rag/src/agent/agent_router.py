import re
import sys
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Literal

from pymorphy3 import MorphAnalyzer
from sentence_transformers import CrossEncoder
from transformers import pipeline

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import time

from src.retrieval.bm25 import load_index, INDEX_PATH
from src.retrieval.retrieve import retrieve_top
from src.retrieval.encoder import rerank_one, MODEL_RERANK, load_chunks_map
from src.retrieval.crag import refuse, retrieve_again, load_refuse_model
from src.llm.client import build_clauses_text
from src.llm.sgr import generate_sgr
from src.llm.promts import PROMT_BASE, PROMT_COMPARISON
from langgraph.graph import StateGraph, START, END
from src.llm.nli import verify_answer_simple as nli_verify

import os
from langfuse import observe
from dotenv import load_dotenv
load_dotenv()


REFUSE_MODEL_PATH = ROOT / "data" / "crag" / "action_eval" / "refuse-logreg.joblib"

if REFUSE_MODEL_PATH.exists():
    load_refuse_model(REFUSE_MODEL_PATH)

import src.retrieval.crag as crag
print("REFUSE_MODEL:", "OK" if crag.REFUSE_MODEL else "NOT LOADED")
print("VECTORIZER:", "OK" if crag.REFUSE_VECTORIZER else "NOT LOADED")


morph = MorphAnalyzer()
bm25, ids, meta = load_index(INDEX_PATH)

POPATKUS_PATH = ROOT / "data" / "popatkus_all_v5.jsonl"
CONFIDENCE_STATS = {
    "top1_q20": 0.0,
    "mean_top3_q20": 0.0,
    "gap12_q20": 0.0,
}

_chunks_map = None
_reranker = None


def detect_lang(text: str) -> str:
    ru_chars = len(re.findall(r'[а-яА-ЯёЁ]', text))
    en_chars = len(re.findall(r'[a-zA-Z]', text))
    return "en" if en_chars > ru_chars else "ru"

COMPARISON_REGEX_PATTERNS = [
    r"чем\s+отличается\s+(.+?)\s+от\s+(.+)",
    r"в\s+чем\s+разница\s+между\s+(.+?)\s+и\s+(.+)",
    r"разница\s+между\s+(.+?)\s+и\s+(.+)",
    r"сравни\s+(.+?)\s+и\s+(.+)",
    r"сравнить\s+(.+?)\s+и\s+(.+)",
    r"how\s+does\s+(.+?)\s+differ\s+from\s+(.+)",
    r"what\s+is\s+the\s+difference\s+between\s+(.+?)\s+and\s+(.+)",
    r"difference\s+between\s+(.+?)\s+and\s+(.+)",
    r"compare\s+(.+?)\s+and\s+(.+)",
]

COMPARISON_MARKERS = [
    "чем отличается", "в чем разница", "разница между", "сравни", "сравнить", "what is the difference between", "difference between", "how does", "differ from", "compare"
    ]

EMAIL_PATTERNS = [
    "напиши письмо", "составь письмо", "черновик письма", "draft email", "write email", "email to",
]

VAGUE_PATTERNS = [
    "что делать", "а дальше", "а потом", "это обязательно", "не успею",
    "кому писать", "куда писать", "где это", "как это",
]

VAGUE_WORDS = {
    "это", "этот", "эта", "эти", "то", "такое", "такой",
    "он", "она", "они", "там", "тут", "сюда", "туда",
}

DOMAIN_HINTS = {
    "академ", "академический", "иуп", "комиссия",
    "пересдача", "долг", "задолженность", "справка",
}

Intent = Literal["ESCALATION", "COMPARISON", "REFUSE", "CLARIFY", "SEARCH", "EMAIL"]


@dataclass
class AgentState():
    query: str
    lang: str
    intent: Optional[Intent] = None
    entity_a: Optional[str] = None
    entity_b: Optional[str] = None
    retry_count: int = 0
    top_final: int = 10
    sgr_result: Optional[Dict[str, Any]] = None 
    is_hallucination: bool = False
    chunks: List[Dict[str, Any]] = field(default_factory=list)
    chunks_a: List[Dict[str, Any]] = field(default_factory=list)
    chunks_b: List[Dict[str, Any]] = field(default_factory=list)
    retrieval_ok: bool = False
    need_retry: bool = False
    defs: list = field(default_factory=list)
    answer: Optional[str] = None
    escalation_reason: Optional[str] = None
    timings: Dict[str, float] = field(default_factory=dict)
    skip_generation: bool = False
    generation_called: bool = False
    answer_len_chars: int = 0
    answer_source: str = "none"


def get_chunks_map():
    global _chunks_map
    if _chunks_map is None:
        if POPATKUS_PATH.exists():
            _chunks_map = load_chunks_map(POPATKUS_PATH)
        else:
            _chunks_map = {}
    return _chunks_map


def get_reranker():
    global _reranker
    if _reranker is None:
        try:
            _reranker = CrossEncoder(MODEL_RERANK, trust_remote_code=True, max_length=512)
        except Exception:
            _reranker = False
    return _reranker


def rerank_items(query, final_ids):
    if not final_ids:
        return []

    reranker = get_reranker()
    if reranker is False:
        items = []
        for idx, chunk_id in enumerate(final_ids):
            items.append({"id": chunk_id, "ce_score": float(len(final_ids) - idx)})
        return items

    return rerank_one(
        reranker=reranker,
        query=query,
        final_ids=final_ids,
        chunks_map=get_chunks_map(),
        batch_size=10,
    )


def state_value(state, key, default=None):
    if isinstance(state, dict):
        return state.get(key, default)
    return getattr(state, key, default)


def set_state_value(state, key, value):
    if isinstance(state, dict):
        state[key] = value
    else:
        setattr(state, key, value)
    return state


def state_timings(state):
    if isinstance(state, dict):
        return state.setdefault("timings", {})
    if not hasattr(state, "timings") or state.timings is None:
        state.timings = {}
    return state.timings


def add_timing(state, name, seconds):
    timings = state_timings(state)
    ms = round(seconds * 1000, 2)
    timings[name] = round(timings.get(name, 0.0) + ms, 2)
    return state


def set_answer_metadata(state, source, generation_called=False):
    answer = state_value(state, "answer", "") or ""
    set_state_value(state, "answer_source", source)
    set_state_value(state, "generation_called", bool(generation_called))
    set_state_value(state, "answer_len_chars", len(str(answer)))
    return state


def timed_node(name, fn):
    def wrapper(state):
        t0 = time.perf_counter()
        out = fn(state)
        add_timing(out, name, time.perf_counter() - t0)
        return out

    return wrapper


def strip_inline_citations(answer, citations):
    cleaned = str(answer or "").strip()

    for citation in citations or []:
        citation = str(citation).strip()
        if citation:
            cleaned = re.sub(rf"\s*\[{re.escape(citation)}\](?=\s|$|[.,:;!?])", "", cleaned)

    return re.sub(r"[ \t]{2,}", " ", cleaned).strip()


def format_generated_answer(result, ctx_ids, fallback_limit=1400, empty_message="Не удалось сформировать ответ, хотя релевантные фрагменты были найдены."):
    final_answer = str(result.get("answer", "") or "").strip()
    citations = result.get("citations", []) or []

    if final_answer:
        source = "sgr_answer"
    else:
        fallback = build_clauses_text(ctx_ids[:1]).strip()

        if fallback:
            fallback = fallback[:fallback_limit]
            if len(fallback) >= fallback_limit:
                fallback = fallback.rsplit(" ", 1)[0] + "..."

            final_answer = fallback
            source = "fallback_context"
        else:
            final_answer = empty_message
            source = "fallback_empty"

    return strip_inline_citations(final_answer, citations), source


def normalize_query(query):
    q = query.lower().replace("ё", "е")
    q = re.sub(r"[^a-zа-я0-9\s]", " ", q)
    tokens = q.split()
    return [morph.parse(tok)[0].normal_form for tok in tokens]


def extract_comparison_entities(query):
    q = query.lower().replace("ё", "е").strip()
    for pattern in COMPARISON_REGEX_PATTERNS:
        m = re.search(pattern, q)
        if m:
            a = m.group(1).strip()
            b = m.group(2).strip()
            a = re.sub(r"[?!.…,;:]+$", "", a).strip()
            b = re.sub(r"[?!.…,;:]+$", "", b).strip()
            return a, b
    return None, None


def is_meaningful_part(text):
    if not text:
        return False
    DOMAIN_ENTITIES = {
        "иуп", "иупа", "индивидуальный", "учебный", "план",
        "академ", "академический", "отпуск",
        "пересдача", "долг", "задолженность",
        "справка", "комиссия", "отчисление",
    }
    q = text.lower().replace("ё", "е")
    if any(entity in q for entity in DOMAIN_ENTITIES):
        return True
    lemmas = normalize_query(text)
    if not lemmas:
        return False
    content = [w for w in lemmas if w not in VAGUE_WORDS]
    return len(content) > 0


def is_definition_query(query):
    q = query.lower().replace("ё", "е")
    return bool(re.search(r"что такое|определение|значени[ея]|это кто|это что|кто такой|как называется|what is|define|definition of", q))

def expand_definition_query(query, lang):
    if not is_definition_query(query):
        return query
    m = re.search(r"(?:что такое|определение|значени[ея]|это кто|это что|кто такой|как называется|what is|define|definition of)\s+(.+?)[?.!]?$", query.lower().replace("ё", "е"))
    if m:
        term = m.group(1).strip()
        if lang == "en":
            return f"{query} glossary term definition {term}"
        return f"{query} глоссарий термин определение {term}"
    return query


def is_comparison(query):
    a, b = extract_comparison_entities(query)
    if a is not None and b is not None:
        return True

    q = query.lower().replace("ё", "е")
    return any(marker in q for marker in COMPARISON_MARKERS)


def is_email_request(query):
    q = query.lower().replace("ё", "е")
    return any(pattern in q for pattern in EMAIL_PATTERNS)


def needs_clarification(query):
    q = query.lower().replace("ё", "е").strip()
    lemmas = normalize_query(query)

    if is_comparison(q):
        a, b = extract_comparison_entities(q)
        if a is None or b is None:
            return True
        if not is_meaningful_part(a) or not is_meaningful_part(b):
            return True
        return False

    if len(lemmas) <= 2:
        if any(l in DOMAIN_HINTS for l in lemmas):
            return False
        return True

    if any(p in q for p in VAGUE_PATTERNS) and len(lemmas) <= 5:
        return True

    return False

def get_sgr_citations(state):
    result = state.sgr_result or {}
    citations = result.get("citations") or []
    if isinstance(citations, str):
        citations = [item.strip() for item in citations.split("|") if item.strip()]
    return [str(item).strip() for item in citations if str(item).strip()]


def get_verifier_context_ids(state):
    context_ids = []
    if state.intent == "COMPARISON":
        chunks = state.chunks_a[:2] + state.chunks_b[:2]
    else:
        chunks = state.chunks[:3]
    for chunk in chunks:
        chunk_id = chunk.get("id")
        if chunk_id and chunk_id not in context_ids:
            context_ids.append(chunk_id)
    return context_ids


def verify_answer_node(state):
    answer = state.answer or ""
    answer_lower = answer.lower()
    empty_answer_markers = [
        "no direct confirmation",
        "в предоставленных документах нет информации для ответа",
        "не удалось сформировать ответ",
        "не удалось сформировать сравнение",
    ]
    if any(marker in answer_lower for marker in empty_answer_markers):
        state.is_hallucination = False
        return state
    citations = get_sgr_citations(state)
    if citations:
        state.is_hallucination = False
        return state
    context_ids = get_verifier_context_ids(state)
    premise = build_clauses_text(context_ids).strip()[:1500]
    result = nli_verify(premise, answer)
    label = str(result.get("label", "")).lower().strip()
    if label == "contradiction":
        state.is_hallucination = True
        state.escalation_reason = "no_citations_nli_contradiction"
    else:
        state.is_hallucination = False
    return state

def route_query(state):
    if is_email_request(state.query):
        state.intent = "EMAIL"
        return state

    if is_comparison(state.query):
        a, b = extract_comparison_entities(state.query)
        if a is not None and b is not None:
            state.intent = "COMPARISON"
            state.entity_a = a
            state.entity_b = b
            return state

    if refuse(state.query, state.lang):
        state.intent = "REFUSE"
        return state
    
    if needs_clarification(state.query):
        state.intent = "CLARIFY"
        return state

    state.intent = "SEARCH"
    return state


@observe(as_type="span")
def retrieve_search(state):
    search_query = expand_definition_query(state.query, state.lang)
    final_ids, defs = retrieve_top(
        query=search_query,
        lang=state.lang,
        bm25=bm25,
        ids=ids,
        meta=meta,
        top_dense=80,
        top_bm25=10,
        top_final=state.top_final,
        only_english=False,
    )

    reranked = rerank_items(state.query, final_ids)
    state.chunks = reranked
    state.defs = defs
    state.need_retry = (not reranked) or retrieve_again(reranked, CONFIDENCE_STATS)
    state.retrieval_ok = not state.need_retry
    return state


def judge_search(state):
    if state.retrieval_ok:
        return state
    if state.need_retry and state.retry_count == 0:
        state.retry_count += 1
        state.top_final = 20
        return state
    state.intent = "ESCALATION"
    state.escalation_reason = "search_retrieval_failed"
    return state


def retry_search(state):
    final_ids, defs = retrieve_top(
        query=state.query,
        lang=state.lang,
        bm25=bm25,
        ids=ids,
        meta=meta,
        top_dense=100,
        top_bm25=20,
        top_final=state.top_final,
        only_english=False,
    )

    reranked = rerank_items(state.query, final_ids)
    state.chunks = reranked
    state.defs = defs
    state.need_retry = (not reranked) or retrieve_again(reranked, CONFIDENCE_STATS)
    state.retrieval_ok = not state.need_retry

    if not state.retrieval_ok:
        state.intent = "ESCALATION"
        state.escalation_reason = "double_retrieval_failed"

    return state


@observe(as_type="span")
def retrieve_comparison(state):
    if not state.entity_a or not state.entity_b:
        state.intent = "CLARIFY"
        state.retrieval_ok = False
        state.need_retry = False
        state.escalation_reason = "comparison_entities_missing"
        return state

    final_ids_a, defs_a = retrieve_top(
        query=state.entity_a, lang=state.lang, bm25=bm25, ids=ids, meta=meta,
        top_dense=80, top_bm25=10, top_final=state.top_final, only_english=False,
    )
    final_ids_b, defs_b = retrieve_top(
        query=state.entity_b, lang=state.lang, bm25=bm25, ids=ids, meta=meta,
        top_dense=80, top_bm25=10, top_final=state.top_final, only_english=False,
    )
    
    reranked_a = rerank_items(state.entity_a, final_ids_a)
    reranked_b = rerank_items(state.entity_b, final_ids_b)
    
    need_retry_a = (not reranked_a) or retrieve_again(reranked_a, CONFIDENCE_STATS)
    need_retry_b = (not reranked_b) or retrieve_again(reranked_b, CONFIDENCE_STATS)
    
    state.chunks_a = reranked_a
    state.chunks_b = reranked_b
    state.defs = [defs_a, defs_b]
    state.need_retry = need_retry_a or need_retry_b
    state.retrieval_ok = not state.need_retry
    return state


def retry_comparison(state):
    final_ids_a, defs_a = retrieve_top(
        query=state.entity_a,
        lang=state.lang,
        bm25=bm25,
        ids=ids,
        meta=meta,
        top_dense=100,
        top_bm25=20,
        top_final=state.top_final,
        only_english=False,
    )

    final_ids_b, defs_b = retrieve_top(
        query=state.entity_b,
        lang=state.lang,
        bm25=bm25,
        ids=ids,
        meta=meta,
        top_dense=100,
        top_bm25=20,
        top_final=state.top_final,
        only_english=False,
    )

    reranked_a = rerank_items(state.entity_a, final_ids_a)
    reranked_b = rerank_items(state.entity_b, final_ids_b)

    state.chunks_a = reranked_a
    state.chunks_b = reranked_b
    state.defs = [defs_a, defs_b]
    state.need_retry = ((not reranked_a) or retrieve_again(reranked_a, CONFIDENCE_STATS)) or ((not reranked_b) or retrieve_again(reranked_b, CONFIDENCE_STATS))
    state.retrieval_ok = not state.need_retry

    if not state.retrieval_ok:
        state.intent = "ESCALATION"
        state.escalation_reason = "double_comparison_retrieval_failed"

    return state


def draft_email(state):
    if state.lang == "en":
        state.answer = "The email draft is ready. Please add the recipient, subject, and your details before sending."
    else:
        state.answer = "Черновик письма готов. Добавь адресата, тему и свои данные перед отправкой."
    return set_answer_metadata(state, "static_email", generation_called=False)


def judge_comparison(state):
    if state.retrieval_ok:
        return state
    if state.need_retry and state.retry_count == 0:
        state.retry_count += 1
        state.top_final = 20
        return state
    state.intent = "ESCALATION"
    state.escalation_reason = "comparison_retrieval_failed"
    return state


def clarify_node(state):
    if state.lang == "en":
        state.answer = "Please clarify what exactly you mean. For example: retakes, academic leave, or certificates."
    else:
        state.answer = "Уточни, пожалуйста, что именно ты имеешь в виду. Например: пересдачи, академический отпуск, ИУП или справка."
    return set_answer_metadata(state, "static_clarify", generation_called=False)

def refuse_node(state):
    if state.lang == "en":
        state.answer = "I cannot help with this request. Please rephrase your question according to Popatkus rules."
    else:
        state.answer = "Я не могу помочь с таким запросом. Лучше переформулируй вопрос в рамках правил и регламентов Попаткуса."
    return set_answer_metadata(state, "static_refuse", generation_called=False)



def escalation_node(state):
    if state.lang == "en":
        state.answer = "I couldn't find a reliable answer. Let me transfer you to a human operator."
    else:
        state.answer = "Не удалось надежно найти ответ. Лучше передать вопрос оператору."
    return set_answer_metadata(state, "static_escalate", generation_called=False)


@observe(as_type="generation")
def generate_search_answer(state):
    if state.skip_generation:
        state.answer = "__GENERATION_SKIPPED__"
        state.sgr_result = {"answer": state.answer, "citations": []}
        return set_answer_metadata(state, "skipped", generation_called=False)

    ctx_ids = [item["id"] for item in state.chunks[:3]]
    result = generate_sgr(state.query, state.lang, ctx_ids, top_ctx=1)
    state.sgr_result = result
    state.answer, source = format_generated_answer(
        result=result,
        ctx_ids=ctx_ids,
        fallback_limit=1400,
        empty_message="Не удалось сформировать ответ, хотя релевантные фрагменты были найдены.",
    )
    return set_answer_metadata(state, source, generation_called=True)


@observe(as_type="generation")
def generate_comparison_answer(state):
    if state.skip_generation:
        state.answer = "__GENERATION_SKIPPED__"
        state.sgr_result = {"answer": state.answer, "citations": []}
        return set_answer_metadata(state, "skipped", generation_called=False)

    ctx_ids = []

    for item in state.chunks_a[:2] + state.chunks_b[:2]:
        cid = item["id"]
        if cid not in ctx_ids:
            ctx_ids.append(cid)

    result = generate_sgr(state.query, state.lang, ctx_ids, top_ctx=4)
    state.sgr_result = result
    state.answer, source = format_generated_answer(
        result=result,
        ctx_ids=ctx_ids[:2],
        fallback_limit=1800,
        empty_message="Не удалось сформировать сравнение, хотя релевантные фрагменты были найдены.",
    )
    return set_answer_metadata(state, source, generation_called=True)

def route_intent_edge(state):
    intent = state_value(state, "intent")
    if intent == "SEARCH":
        return "retrieve_search"
    if intent == "COMPARISON":
        return "retrieve_comparison"
    if intent == "EMAIL":
        return "draft_email"
    if intent == "CLARIFY":
        return "clarify"
    return "refuse"



def judge_search_edge(state):
    if state_value(state, "retrieval_ok", False):
        return "generate_search_answer"

    if state_value(state, "need_retry", False) and state_value(state, "retry_count", 0) == 0:
        return "retry_search"

    return "escalate"



def judge_comparison_edge(state):
    if state_value(state, "retrieval_ok", False):
        return "generate_comparison_answer"

    if state_value(state, "need_retry", False) and state_value(state, "retry_count", 0) == 0:
        return "retry_comparison"
    return "escalate"


def verify_router(state):
    return "escalate" if state_value(state, "is_hallucination", False) else "end"


agent_rag = StateGraph(AgentState)
agent_rag.add_node("route_intent", timed_node("route_intent", route_query))
agent_rag.add_node("retrieve_search", timed_node("retrieve_search", retrieve_search))
agent_rag.add_node("judge_search", timed_node("judge_search", judge_search))
agent_rag.add_node("retry_search", timed_node("retry_search", retry_search))
agent_rag.add_node("retrieve_comparison", timed_node("retrieve_comparison", retrieve_comparison))
agent_rag.add_node("judge_comparison", timed_node("judge_comparison", judge_comparison))
agent_rag.add_node("retry_comparison", timed_node("retry_comparison", retry_comparison))
agent_rag.add_node("generate_search_answer", timed_node("generate_search_answer", generate_search_answer))
agent_rag.add_node("generate_comparison_answer", timed_node("generate_comparison_answer", generate_comparison_answer))
agent_rag.add_node("draft_email", timed_node("draft_email", draft_email))
agent_rag.add_node("clarify", timed_node("clarify", clarify_node))
agent_rag.add_node("verify_answer", timed_node("verify_answer", verify_answer_node))
agent_rag.add_node("refuse", timed_node("refuse", refuse_node))
agent_rag.add_node("escalate", timed_node("escalate", escalation_node))

agent_rag.add_edge(START, "route_intent")
agent_rag.add_conditional_edges("route_intent", route_intent_edge)
agent_rag.add_edge("retrieve_search", "judge_search")
agent_rag.add_conditional_edges("judge_search", judge_search_edge)
agent_rag.add_edge("generate_search_answer", "verify_answer")
agent_rag.add_edge("generate_comparison_answer", "verify_answer")
agent_rag.add_edge("retry_search", "judge_search")
agent_rag.add_edge("retrieve_comparison", "judge_comparison")
agent_rag.add_conditional_edges("judge_comparison", judge_comparison_edge)
agent_rag.add_edge("retry_comparison", "judge_comparison")
agent_rag.add_edge("draft_email", END)
agent_rag.add_edge("clarify", END)
agent_rag.add_edge("refuse", END)
agent_rag.add_conditional_edges("verify_answer", verify_router, {"escalate": "escalate", "end": END})
agent_rag.add_edge("escalate", END)

graph = agent_rag.compile()


def finalize_timings(state, total_s, mode):
    timings = state_timings(state)
    total_ms = round(total_s * 1000, 2)
    timings["total"] = total_ms

    if mode == "agent":
        core_ms = (
            timings.get("retrieve_search", 0.0)
            + timings.get("retry_search", 0.0)
            + timings.get("retrieve_comparison", 0.0)
            + timings.get("retry_comparison", 0.0)
            + timings.get("generate_search_answer", 0.0)
            + timings.get("generate_comparison_answer", 0.0)
        )
        timings["agentic_overhead"] = round(max(total_ms - core_ms, 0.0), 2)

    return state


@observe()
def run_agent(query, include_generation=True):
    detected_lang = detect_lang(query)
    state = AgentState(
        query=query,
        lang=detected_lang,
        skip_generation=not include_generation,
    )
    return graph.invoke(state, config={"recursion_limit": 50})


def run_agent_timed(query, include_generation=True):
    t0 = time.perf_counter()
    state = run_agent(query, include_generation=include_generation)
    total_s = time.perf_counter() - t0
    finalize_timings(state, total_s, mode="agent")
    return state, round(total_s, 3)


def run_baseline(query, include_generation=True):
    lang = detect_lang(query)
    timings = {}
    total_t0 = time.perf_counter()

    search_query = expand_definition_query(query, lang)

    t0 = time.perf_counter()
    final_ids, defs = retrieve_top(
        query=search_query,
        lang=lang,
        bm25=bm25,
        ids=ids,
        meta=meta,
        top_dense=80,
        top_bm25=10,
        top_final=10,
        only_english=False,
    )
    timings["baseline_retrieve"] = round((time.perf_counter() - t0) * 1000, 2)

    t0 = time.perf_counter()
    chunks = rerank_items(query, final_ids)
    timings["baseline_rerank"] = round((time.perf_counter() - t0) * 1000, 2)

    result = None
    answer = "__GENERATION_SKIPPED__"
    answer_source = "skipped"
    generation_called = False

    if include_generation:
        t0 = time.perf_counter()
        ctx_ids = [item["id"] for item in chunks[:3]]
        result = generate_sgr(query, lang, ctx_ids, top_ctx=1)
        answer, answer_source = format_generated_answer(
            result=result,
            ctx_ids=ctx_ids,
            fallback_limit=1400,
            empty_message="Не удалось сформировать ответ, хотя релевантные фрагменты были найдены.",
        )
        generation_called = True
        timings["baseline_generate"] = round((time.perf_counter() - t0) * 1000, 2)
    else:
        timings["baseline_generate"] = 0.0

    total_ms = round((time.perf_counter() - total_t0) * 1000, 2)
    timings["total"] = total_ms

    return {
        "query": query,
        "lang": lang,
        "intent": "BASELINE_SEARCH",
        "chunks": chunks,
        "defs": defs,
        "answer": answer,
        "sgr_result": result,
        "generation_called": generation_called,
        "answer_source": answer_source,
        "answer_len_chars": len(str(answer or "")),
        "timings": timings,
    }


def run_baseline_timed(query, include_generation=True):
    t0 = time.perf_counter()
    state = run_baseline(query, include_generation=include_generation)
    total_s = time.perf_counter() - t0
    state["timings"]["total"] = round(total_s * 1000, 2)
    return state, round(total_s, 3)
