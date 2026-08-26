"""
Отладка ретривала: показывает, какие чанки реально находит векторный поиск,
без похода в LLM. Помогает понять - проблема в поиске или в том, что модель
плохо использует найденный контекст.

Запуск:
  python debug_search.py "твой вопрос"
  python debug_search.py "твой вопрос" --top_k 10
"""
import argparse

from ingest import embed_text, get_collection
from rag import retrieve, expand_query, _extract_identifiers


def debug_search(question: str, top_k: int = 8):
    collection = get_collection()
    print(f"Всего чанков в базе: {collection.count()}\n")

    identifiers = _extract_identifiers(question)
    if identifiers:
        print(f"Обнаружены идентификаторы для точного поиска: {identifiers}\n")

    expanded = expand_query(question)
    if expanded != question:
        print(f"Запрос раскрыт глоссарием/LLM: {expanded!r}\n")

    # Чистый векторный поиск (для сравнения)
    q_embedding = embed_text(question)
    vector_results = collection.query(query_embeddings=[q_embedding], n_results=top_k)

    print(f"Запрос: {question}\n{'=' * 80}")
    print("\n### Только векторный поиск ###")
    for i, (doc, meta, dist) in enumerate(
        zip(vector_results["documents"][0], vector_results["metadatas"][0], vector_results["distances"][0])
    ):
        print(f"\n--- {i+1} | distance={dist:.4f} | {meta.get('source_path')} | {meta.get('heading_path')} ---")
        print(doc[:150].replace("\n", " ") + "...")

    print(f"\n{'=' * 80}")
    print("\n### Гибридный поиск (вектор + BM25), то что реально уйдёт в промпт ###")
    hybrid_chunks = retrieve(question, top_k=top_k, hybrid=True)
    for i, c in enumerate(hybrid_chunks):
        meta = c["metadata"]
        print(f"\n--- {i+1} | {meta.get('source_path')} | {meta.get('heading_path')} ---")
        print(c["text"][:150].replace("\n", " ") + "...")

    print(f"\n{'=' * 80}")
    print("Если гибридный поиск находит то, чего не было в чисто векторном - BM25 помог.")
    print("Если оба варианта мимо - дело в чанкинге (слишком крупные/мелкие куски)")
    print("или в том, что вопрос сформулирован сильно иначе, чем текст в документах.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("question")
    parser.add_argument("--top_k", type=int, default=8)
    args = parser.parse_args()
    debug_search(args.question, top_k=args.top_k)
