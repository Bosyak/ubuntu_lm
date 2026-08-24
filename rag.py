"""
Логика самого RAG-запроса: находим релевантные чанки, собираем промпт, спрашиваем модель.

Поиск гибридный:
  - векторный (семантическая близость) - хорош для вопросов "как" / "почему" / смысловых
  - BM25 по ключевым словам - хорош для точных терминов, имён файлов, названий папок
Результаты обоих сливаются через Reciprocal Rank Fusion (RRF).
"""
import re

import ollama
from rank_bm25 import BM25Okapi

import config
from ingest import embed_text, get_collection


SYSTEM_PROMPT = """Ты — ассистент, отвечающий на вопросы по документации проекта.
Отвечай только на основе предоставленного контекста ниже.
Если в контексте нет ответа - честно скажи, что не нашёл информацию в документации, не выдумывай.
Обязательно указывай, из какого файла взята информация (смотри пометки [Файл: ...] в контексте)."""

_TOKEN_RE = re.compile(r"[a-zA-Zа-яА-ЯёЁ0-9_]+")


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


def _bm25_search(question: str, all_docs: list[str], all_ids: list[str], top_k: int) -> list[str]:
    """Возвращает ids топ-N документов по BM25 (точное совпадение слов)."""
    if not all_docs:
        return []
    tokenized_corpus = [_tokenize(d) for d in all_docs]
    bm25 = BM25Okapi(tokenized_corpus)
    scores = bm25.get_scores(_tokenize(question))
    ranked = sorted(zip(all_ids, scores), key=lambda x: x[1], reverse=True)
    return [doc_id for doc_id, score in ranked[:top_k] if score > 0]


def _reciprocal_rank_fusion(rank_lists: list[list[str]], k: int = 60) -> list[str]:
    """
    Классический RRF: объединяет несколько ранжированных списков id в один.
    Чем выше объект в каждом списке, тем больше он получает очков.
    """
    scores: dict[str, float] = {}
    for rank_list in rank_lists:
        for rank, doc_id in enumerate(rank_list):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)
    return [doc_id for doc_id, _ in sorted(scores.items(), key=lambda x: x[1], reverse=True)]


def retrieve(
    question: str, top_k: int = 5, folder_filter: str | None = None, hybrid: bool = True
) -> list[dict]:
    """
    Ищет top_k наиболее релевантных чанков.
    folder_filter - опционально, можно искать только внутри конкретной папки,
    например folder_filter="docs/architecture"
    hybrid - использовать ли BM25 в дополнение к векторному поиску (рекомендуется)
    """
    collection = get_collection()
    where = {"folder_path": folder_filter} if folder_filter else None

    # 1. Векторный поиск (берём с запасом, чтобы RRF было из чего выбирать)
    q_embedding = embed_text(question)
    vector_n = max(top_k * 3, 15)
    vector_results = collection.query(
        query_embeddings=[q_embedding], n_results=vector_n, where=where
    )
    vector_ids = vector_results.get("ids", [[]])[0]

    doc_by_id = {}
    meta_by_id = {}
    for doc_id, doc, meta in zip(
        vector_ids, vector_results["documents"][0], vector_results["metadatas"][0]
    ):
        doc_by_id[doc_id] = doc
        meta_by_id[doc_id] = meta

    if not hybrid:
        return [
            {"text": doc_by_id[i], "metadata": meta_by_id[i]} for i in vector_ids[:top_k]
        ]

    # 2. BM25 по ключевым словам - тянем все документы под тем же фильтром папки
    all_data = collection.get(where=where) if where else collection.get()
    all_ids, all_docs = all_data["ids"], all_data["documents"]
    for doc_id, doc, meta in zip(all_ids, all_docs, all_data["metadatas"]):
        doc_by_id.setdefault(doc_id, doc)
        meta_by_id.setdefault(doc_id, meta)

    bm25_ids = _bm25_search(question, all_docs, all_ids, top_k=vector_n)

    # 3. Слияние через RRF и обрезка до top_k
    fused_ids = _reciprocal_rank_fusion([vector_ids, bm25_ids])[:top_k]

    return [{"text": doc_by_id[i], "metadata": meta_by_id[i]} for i in fused_ids]


def build_prompt(question: str, chunks: list[dict]) -> str:
    context = "\n\n---\n\n".join(c["text"] for c in chunks)
    return f"""Контекст из документации:

{context}

---

Вопрос: {question}"""


def ask(
    question: str, top_k: int = 5, folder_filter: str | None = None, hybrid: bool = True
) -> dict:
    chunks = retrieve(question, top_k=top_k, folder_filter=folder_filter, hybrid=hybrid)

    if not chunks:
        return {
            "answer": "В базе нет проиндексированных документов (или ни один не подошёл под фильтр).",
            "sources": [],
        }

    prompt = build_prompt(question, chunks)

    client = ollama.Client(host=config.OLLAMA_HOST)
    response = client.chat(
        model=config.LLM_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
    )

    sources = [
        {
            "file": c["metadata"].get("source_path"),
            "section": c["metadata"].get("heading_path"),
        }
        for c in chunks
    ]

    return {"answer": response["message"]["content"], "sources": sources}


if __name__ == "__main__":
    import sys

    q = " ".join(sys.argv[1:]) or "О чём этот репозиторий?"
    result = ask(q)
    print("\n=== ОТВЕТ ===")
    print(result["answer"])
    print("\n=== ИСТОЧНИКИ ===")
    for s in result["sources"]:
        print(f"  - {s['file']}  [{s['section']}]")
