import re
import sys
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Literal

from pymorphy3 import MorphAnalyzer
from sentence_transformers import CrossEncoder

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.retrieval.bm25 import load_index, INDEX_PATH
from src.retrieval.retrieve import retrieve_top
from src.retrieval.encoder import rerank_one, MODEL_RERANK, load_chunks_map
from src.retrieval.crag import refuse, retrieve_again, load_refuse_model
from src.llm.client import generate_answer
from src.llm.promts import PROMT_BASE, PROMT_COMPARISON
from langgraph.graph import StateGraph, START, END


REFUSE_MODEL_PATH = ROOT / "data" / "crag" / "action_eval" / "refuse-logreg.joblib"

if REFUSE_MODEL_PATH.exists():
    load_refuse_model(REFUSE_MODEL_PATH)


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

COMPARISON_REGEX_PATTERNS = [
    r"чем\s+(.+?)\s+отличается\s+от\s+(.+)",
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
    "чем отличается",
    "в чем разница",
    "разница между",
    "сравни",
    "сравнить",
    "what is the difference between",
    "difference between",
    "how does",
    "differ from",
    "compare",
]

EMAIL_PATTERNS = [
    "напиши письмо",
    "составь письмо",
    "черновик письма",
    "draft email",
    "write email",
    "email to",
]

VAGUE_PATTERNS = [
    "что делать",
    "а дальше",
    "а потом",
    "это обязательно",
    "не успею",
    "что будет",
    "кому писать",
    "куда писать",
    "где это",
    "как это",
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
    chunks: List[Dict[str, Any]] = field(default_factory=list)
    chunks_a: List[Dict[str, Any]] = field(default_factory=list)
    chunks_b: List[Dict[str, Any]] = field(default_factory=list)
    retrieval_ok: bool = False
    need_retry: bool = False
    defs: list = field(default_factory=list)
    answer: Optional[str] = None
    escalation_reason: Optional[str] = None


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
            return a, b
    return None, None


def is_meaningful_part(text):
    if not text:
        return False
    lemmas = normalize_query(text)
    if not lemmas:
        return False
    content = [w for w in lemmas if w not in VAGUE_WORDS]
    return len(content) > 0


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

    if any(p in q for p in VAGUE_PATTERNS):
        return True

    return False


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

    if needs_clarification(state.query):
        state.intent = "CLARIFY"
        return state

    if refuse(state.query, state.lang):
        state.intent = "REFUSE"
        return state

    state.intent = "SEARCH"
    return state


def retrieve_search(state):
    final_ids, defs = retrieve_top(
        query=state.query,
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


def retrieve_comparison(state):
    if not state.entity_a or not state.entity_b:
        state.intent = "CLARIFY"
        state.retrieval_ok = False
        state.need_retry = False
        state.escalation_reason = "comparison_entities_missing"
        return state

    final_ids_a, defs_a = retrieve_top(
        query=state.entity_a,
        lang=state.lang,
        bm25=bm25,
        ids=ids,
        meta=meta,
        top_dense=80,
        top_bm25=10,
        top_final=state.top_final,
        only_english=False,
    )
    final_ids_b, defs_b = retrieve_top(
        query=state.entity_b,
        lang=state.lang,
        bm25=bm25,
        ids=ids,
        meta=meta,
        top_dense=80,
        top_bm25=10,
        top_final=state.top_final,
        only_english=False,
    )
    reranked_a = rerank_items(state.entity_a, final_ids_a)
    reranked_b = rerank_items(state.entity_b, final_ids_b)

    state.chunks_a = reranked_a
    state.chunks_b = reranked_b
    state.defs = [defs_a, defs_b]

    need_retry_a = (not reranked_a) or retrieve_again(reranked_a, CONFIDENCE_STATS)
    need_retry_b = (not reranked_b) or retrieve_again(reranked_b, CONFIDENCE_STATS)

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
    state.answer = "Черновик письма готов. Добавь адресата, тему и свои данные перед отправкой."
    return state


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
    state.answer = "Уточни, пожалуйста, что именно ты имеешь в виду. Например: пересдачи, академический отпуск, ИУП или справка."
    return state


def refuse_node(state):
    state.answer = "Я не могу помочь с таким запросом. Лучше переформулируй вопрос в рамках правил и регламентов Попаткуса."
    return state


def escalation_node(state):
    state.answer = "Не удалось надежно найти ответ. Лучше передать вопрос оператору."
    return state


def generate_search_answer(state):
    ctx_ids = [item["id"] for item in state.chunks[:3]]
    try:
        state.answer = generate_answer(
            query=state.query,
            lang=state.lang,
            ctx_ids=ctx_ids,
            promt=PROMT_BASE,
            top_ctx=3,
        )
    except Exception as exc:
        state.answer = f"LLM недоступна: {exc}"
    return state


def generate_comparison_answer(state):
    ctx_ids = []
    for item in state.chunks_a[:2] + state.chunks_b[:2]:
        chunk_id = item["id"]
        if chunk_id not in ctx_ids:
            ctx_ids.append(chunk_id)
    try:
        state.answer = generate_answer(
            query=state.query,
            lang=state.lang,
            ctx_ids=ctx_ids,
            promt=PROMT_COMPARISON,
            top_ctx=4,
        )
    except Exception as exc:
        state.answer = f"LLM недоступна: {exc}"
    return state


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

    if state_value(state, "need_retry", False) and state_value(state, "retry_count", 0) == 1:
        return "retry_search"

    return "escalate"



def judge_comparison_edge(state):
    if state_value(state, "retrieval_ok", False):
        return "generate_comparison_answer"

    if state_value(state, "need_retry", False) and state_value(state, "retry_count", 0) == 1:
        return "retry_comparison"
    return "escalate"


agent_rag = StateGraph(AgentState)
agent_rag.add_node("route_intent", route_query)
agent_rag.add_node("retrieve_search", retrieve_search)
agent_rag.add_node("judge_search", judge_search)
agent_rag.add_node("retry_search", retry_search)
agent_rag.add_node("retrieve_comparison", retrieve_comparison)
agent_rag.add_node("judge_comparison", judge_comparison)
agent_rag.add_node("retry_comparison", retry_comparison)
agent_rag.add_node("generate_search_answer", generate_search_answer)
agent_rag.add_node("generate_comparison_answer", generate_comparison_answer)
agent_rag.add_node("draft_email", draft_email)
agent_rag.add_node("clarify", clarify_node)
agent_rag.add_node("refuse", refuse_node)
agent_rag.add_node("escalate", escalation_node)

agent_rag.add_edge(START, "route_intent")
agent_rag.add_conditional_edges("route_intent", route_intent_edge)
agent_rag.add_edge("retrieve_search", "judge_search")
agent_rag.add_conditional_edges("judge_search", judge_search_edge)
agent_rag.add_edge("retry_search", "judge_search")
agent_rag.add_edge("retrieve_comparison", "judge_comparison")
agent_rag.add_conditional_edges("judge_comparison", judge_comparison_edge)
agent_rag.add_edge("retry_comparison", "judge_comparison")
agent_rag.add_edge("draft_email", END)
agent_rag.add_edge("clarify", END)
agent_rag.add_edge("refuse", END)
agent_rag.add_edge("generate_search_answer", END)
agent_rag.add_edge("generate_comparison_answer", END)
agent_rag.add_edge("escalate", END)

graph = agent_rag.compile()


def run_agent(query, lang="ru"):
    state = AgentState(query=query, lang=lang)
    return graph.invoke(state)