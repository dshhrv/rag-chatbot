import sys
from pathlib import Path
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import uvicorn

CURRENT_DIR = Path(__file__).resolve().parent

rag_path = CURRENT_DIR / "rag"
sys.path.insert(0, str(rag_path))

from src.agent.agent_router import run_agent

app = FastAPI(title="Попаткус API")
class ChatRequest(BaseModel):
    message: str

@app.post("/api/chat")
def chat_endpoint(req: ChatRequest):
    message = req.message
    if not message or not message.strip():
        return {"reply": "Пожалуйста, напишите вопрос."}
    
    try:
        state = run_agent(message)
        if isinstance(state, dict):
            answer = state.get('answer', '')
            sgr_result = state.get('sgr_result')
        else:
            answer = getattr(state, 'answer', '')
            sgr_result = getattr(state, 'sgr_result', None)
        
        if sgr_result is None:
            sgr_result = {}
        citations = sgr_result.get('citations',[])
        citations_text = ""
        if citations:
            citations_text = "\n\n**Источники:**\n" + "\n".join(f"- [{cid}]" for cid in citations)
        
        full_response = answer + citations_text
        
        return {"reply": full_response if full_response else "Не удалось получить ответ."}
    
    except Exception as e:
        return {"reply": f"Ошибка на сервере: {str(e)}"}

app.mount("/", StaticFiles(directory="static", html=True), name="static")

if __name__ == "__main__":
    print("Сервер запущен! Откройте браузер по адресу: http://localhost:8000")
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)