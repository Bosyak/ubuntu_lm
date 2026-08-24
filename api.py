"""
API-сервер. Запуск:
  python api.py
или
  uvicorn api:app --host 0.0.0.0 --port 8000

После запуска доступно:
  POST http://<IP-компа>:8000/query    - задать вопрос
  POST http://<IP-компа>:8000/reindex  - обновить индекс (git pull + переиндексация изменений)
  GET  http://<IP-компа>:8000/health   - проверка что сервер жив
"""
from typing import Optional

import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

import config
from ingest import run_ingest, get_collection
from rag import ask

app = FastAPI(title="Repo RAG API")


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


if __name__ == "__main__":
    uvicorn.run(app, host=config.API_HOST, port=config.API_PORT)
