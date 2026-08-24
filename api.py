"""
API-сервер. Запуск:
  python api.py
или
  uvicorn api:app --host 0.0.0.0 --port 8000

После запуска доступно:
  GET  http://<IP-компа>:8000/                     - веб-интерфейс (открой в браузере)
  POST http://<IP-компа>:8000/query                - задать вопрос (свой формат, используется интерфейсом)
  POST http://<IP-компа>:8000/reindex              - обновить индекс (git pull + переиндексация изменений)
  GET  http://<IP-компа>:8000/health               - проверка что сервер жив
  POST http://<IP-компа>:8000/v1/chat/completions  - OpenAI-совместимый формат (для Open WebUI и подобных)
  GET  http://<IP-компа>:8000/v1/models            - список "моделей" (для Open WebUI и подобных)
"""
import time
import uuid
from pathlib import Path
from typing import Optional

import uvicorn
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

import config
from ingest import run_ingest, get_collection
from rag import ask

app = FastAPI(title="Repo RAG API")

STATIC_DIR = Path(__file__).parent / "static"


@app.get("/", response_class=HTMLResponse)
def index():
    """Отдаёт минимальный веб-интерфейс (static/index.html)."""
    html_path = STATIC_DIR / "index.html"
    return html_path.read_text(encoding="utf-8")


class QueryRequest(BaseModel):
    question: str
    top_k: int = 5
    folder_filter: Optional[str] = None


class QueryResponse(BaseModel):
    answer: str
    sources: list


@app.get("/health")
def health():
    try:
        count = get_collection().count()
        return {"status": "ok", "chunks_in_db": count, "llm_model": config.LLM_MODEL}
    except Exception as e:
        return {"status": "error", "detail": str(e)}


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest):
    result = ask(req.question, top_k=req.top_k, folder_filter=req.folder_filter)
    return result


@app.post("/reindex")
def reindex(full: bool = False):
    run_ingest(full_reindex=full)
    return {"status": "ok", "chunks_in_db": get_collection().count()}


# ============================================================================
# OpenAI-совместимый API - чтобы Open WebUI (и любой другой клиент,
# понимающий OpenAI API) мог обращаться к твоему RAG-пайплайну как к обычной
# "модели", не зная, что внутри на самом деле поиск по Chroma + Ollama.
# ============================================================================

MODEL_ID = "repo-rag"


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str = MODEL_ID
    messages: list[ChatMessage]
    stream: bool = False
    # Остальные поля OpenAI-запроса (temperature, max_tokens и т.д.)
    # принимаем, но не используем - ими управляет сама модель в Ollama.
    class Config:
        extra = "allow"


def _extract_question(messages: list[ChatMessage]) -> str:
    """Берём последнее сообщение пользователя как вопрос для RAG."""
    for msg in reversed(messages):
        if msg.role == "user":
            return msg.content
    return ""


@app.get("/v1/models")
def list_models():
    """Open WebUI при подключении спрашивает список доступных моделей."""
    return {
        "object": "list",
        "data": [
            {
                "id": MODEL_ID,
                "object": "model",
                "created": int(time.time()),
                "owned_by": "repo-rag",
            }
        ],
    }


@app.post("/v1/chat/completions")
def chat_completions(req: ChatCompletionRequest):
    question = _extract_question(req.messages)
    result = ask(question)

    # Добавляем список источников прямо в текст ответа - удобно видеть в чате,
    # из каких файлов документации взята информация.
    answer_text = result["answer"]
    if result["sources"]:
        sources_lines = "\n".join(
            f"- {s['file']} ({s['section']})" for s in result["sources"]
        )
        answer_text += f"\n\n---\n**Источники:**\n{sources_lines}"

    # Формат ответа, ожидаемый любым OpenAI-совместимым клиентом
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": req.model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": answer_text},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        },
    }


if __name__ == "__main__":
    uvicorn.run(app, host=config.API_HOST, port=config.API_PORT)
