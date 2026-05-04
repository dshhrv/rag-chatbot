import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "opt" / "rag"))

import gradio as gr
from src.agent.agent_router import run_agent

def respond(message: str, history: list):
    if not message or not message.strip():
        return "Пожалуйста, напишите вопрос."
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
        
        citations = sgr_result.get('citations', [])
        citations_text = ""
        if citations:
            citations_text = "\n\n**Источники:**\n" + "\n".join(f"- [{cid}]" for cid in citations)
        
        full_response = answer + citations_text
        return full_response if full_response else "Не удалось получить ответ."
    except Exception as e:
        return f"Ошибка: {str(e)}"

demo = gr.ChatInterface(
    fn=respond,
    title="RAG Чат-бот (Попаткус)",
    description="Задай вопрос по правилам и документам НИУ ВШЭ."
)

if __name__ == "__main__":
    demo.launch(share=True)