import json
import argparse
import numpy as np
import pandas as pd
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
from src.retrieval.crag import crag_retrieved, refuse, retrieve_again
from src.llm.client import initialize, OUT_PATH, IN_PATH
from src.llm.promts import PROMT_BASE, PROMT_COMPARISON
from langgraph.graph import StateGraph


morph = MorphAnalyzer()
bm25, ids, meta = load_index(INDEX_PATH)

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

VAGUE_PATTERNS = ["что делать","а дальше", "а потом", "это обязательно", "не успею",
                  "что будет", "кому писать", "куда писать", "где это", "как это"]

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
    if needs_clarification(state.query):
        state.intent = "CLARIFY"
        return state
    if is_email_request(state.query):
        state.intent = "EMAIL"
        return state
    if is_comparison(state.query):
        a, b = extract_comparison_entities(state.query)
        state.intent = "COMPARISON"
        state.entity_a = a
        state.entity_b = b
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

    state.chunks = final_ids
    state.defs = defs

    if retrieve_again(final_ids, stats):
        state.need_retry = True
        state.retrieval_ok = False
        return state

    state.need_retry = False
    state.retrieval_ok = True
    return state

def judge_search(state: AgentState):
    if state.retrieval_ok:
        return state

    if state.need_retry and state.retry_count == 0:
        state.retry_count += 1
        state.top_final = 20
        return state

    state.intent = "ESCALATION"
    state.escalation_reason = "search_retrieval_failed"
    return state

def retry_search(state: AgentState):
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

    state.chunks = final_ids
    state.defs = defs

    state.need_retry = False
    state.retrieval_ok = not retrieve_again(final_ids, defs)

    if not state.retrieval_ok:
        state.intent = "ESCALATION"
        state.escalation_reason = "double_retrieval_failed"

    return state

def retrieve_comparison(state: AgentState):
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

    state.chunks_a = final_ids_a
    state.chunks_b = final_ids_b
    state.defs = [defs_a, defs_b]

    need_retry_a = retrieve_again(final_ids_a, defs_a)
    need_retry_b = retrieve_again(final_ids_b, defs_b)

    state.need_retry = need_retry_a or need_retry_b
    state.retrieval_ok = not state.need_retry
    return state

def draft_email(state):
    state.answer = "Черновик письма готов."
    reutrn state


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
    state.answer = "Уточни, пожалуйста, что именно ты имеешь в виду."
    return state

def refuse_node(state):
    state.answer = "Я не могу помочь с таким запросом."
    return state

def escalation_node(state):
    state.answer = "Не удалось надежно найти ответ. Лучше передать вопрос оператору."
    return state

def generate_search_answer(state):
    state.answer = initialize(
        in_path=IN_PATH, out_path=OUT_PATH, promt=PROMT_BASE, top_ctx=3
    )
    return state

def generate_comparison_answer(state):
    state.answer = initialize(
        in_path=IN_PATH, out_path=OUT_PATH, promt=PROMT_COMPARISON, top_ctx=3
    )
    return state

def route_intent_edge(state):
    intent = state["intent"]
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
    if state.get("retrieval_ok", False):
        return "generate_search_answer"

    if state.get("need_retry", False) and state.get("retry_count", 0) == 0:
        return "retry_search"

    return "escalate"


def judge_comparison_edge(state):
    if state.get("retrieval_ok", False):
        return "generate_comparison_answer"

    if state.get("need_retry", False) and state.get("retry_count", 0) == 0:
        return "retry_comparison"
    return "escalate"



agent_rag = StateGraph(AgentState)
agent_rag.add_node("route_intent", route_intent)

agent_rag.add_node("retrieve_search", retrieve_search)
agent_rag.add_node("judge_search", judge_search)
agent_rag.add_node("retry_search", retry_search)

agent_rag.add_node("retrieve_comparison", retrieve_comparison)
agent_rag.add_node("judge_comparison", judge_comparison)
agent_rag.add_node("retry_comparison", retry_comparison)

agent_rag.add_node("generate_search_answer", generate_search_answer)
agent_rag.add_node("generate_comparison_answer", generate_comparison_answer)

agent_rag.add_node("draft_email", draft_email)
agent_rag.add_node("clarify", clarify)
agent_rag.add_node("refuse", refuse_node)
agent_rag.add_node("escalate", escalate)


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



