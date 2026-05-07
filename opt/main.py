import sys
import time
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
from starlette.responses import Response
import uvicorn

CURRENT_DIR = Path(__file__).resolve().parent

rag_path = CURRENT_DIR / "rag"
sys.path.insert(0, str(rag_path))

from src.agent.agent_router import run_agent, detect_lang
from src.monitoring.metrics import (
    RAG_REQUESTS,
    RAG_ERRORS,
    RAG_LATENCY,
    RAG_ESCALATIONS,
    RAG_HALLUCINATION_BLOCKS,
    RAG_IN_PROGRESS,
)

app = FastAPI(title="Попаткус API")


class ChatRequest(BaseModel):
    message: str


@app.get("/metrics")
def metrics_endpoint():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/api/chat")
def chat_endpoint(req: ChatRequest):
    message = req.message

    if not message or not message.strip():
        return {"reply": "Пожалуйста, напишите вопрос."}

    lang = detect_lang(message)
    intent = "unknown"
    status = "ok"
    t0 = time.perf_counter()

    with RAG_IN_PROGRESS.track_inprogress():
        try:
            state = run_agent(message)

            if isinstance(state, dict):
                answer = state.get("answer", "")
                sgr_result = state.get("sgr_result")
                intent = state.get("intent") or "unknown"
                escalation_reason = state.get("escalation_reason")
                is_hallucination = state.get("is_hallucination", False)
            else:
                answer = getattr(state, "answer", "")
                sgr_result = getattr(state, "sgr_result", None)
                intent = getattr(state, "intent", "unknown")
                escalation_reason = getattr(state, "escalation_reason", None)
                is_hallucination = getattr(state, "is_hallucination", False)

            if escalation_reason:
                status = "escalated"
                RAG_ESCALATIONS.labels(reason=str(escalation_reason)).inc()

            if is_hallucination:
                status = "blocked"
                RAG_HALLUCINATION_BLOCKS.inc()

            if sgr_result is None:
                sgr_result = {}

            citations = sgr_result.get("citations", [])
            citations_text = ""

            if citations:
                citations_text = "\n\n**Источники:**\n" + "\n".join(f"- [{cid}]" for cid in citations)

            full_response = answer + citations_text

            return {
                "reply": full_response if full_response else "Не удалось получить ответ."
            }

        except Exception as e:
            status = "error"
            RAG_ERRORS.labels(stage="chat_endpoint").inc()
            return {"reply": f"Ошибка на сервере: {str(e)}"}

        finally:
            latency = time.perf_counter() - t0
            RAG_LATENCY.labels(lang=lang, intent=str(intent)).observe(latency)
            RAG_REQUESTS.labels(
                lang=lang,
                intent=str(intent),
                status=status,
            ).inc()


app.mount("/", StaticFiles(directory=CURRENT_DIR / "static", html=True), name="static")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)