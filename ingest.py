"""
Основной пайплайн индексации:
  1. Обходит все .md файлы в репозитории
  2. Считает хэш содержимого каждого файла
  3. Сравнивает с сохранённым состоянием (index_state.json) -
     переиндексирует только новые/изменённые файлы,
     удаляет из базы записи об удалённых файлах
  4. Режет markdown на чанки, уважая структуру заголовков (H1 > H2 > H3)
  5. Кладёт в текст и в метаданные путь по папкам и путь по заголовкам
  6. Получает эмбеддинги через Ollama и загружает в Chroma

Запуск:
  python ingest.py            # обычная (инкрементальная) индексация
  python ingest.py --full     # полная переиндексация с нуля
"""
import argparse
import glob
import hashlib
import json
import os

import chromadb
import ollama
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

import config
from git_sync import sync_repo

HEADERS_TO_SPLIT_ON = [
    ("#", "h1"),
    ("##", "h2"),
    ("###", "h3"),
    ("####", "h4"),
]


def file_hash(filepath: str) -> str:
    """Хэш содержимого файла - по нему понимаем, менялся ли файл с прошлого раза."""
    with open(filepath, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def load_state() -> dict:
    if os.path.exists(config.STATE_FILE):
        with open(config.STATE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_state(state: dict) -> None:
    os.makedirs(os.path.dirname(config.STATE_FILE) or ".", exist_ok=True)
    with open(config.STATE_FILE, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)


def chunk_markdown_file(filepath: str, repo_root: str) -> list[dict]:
    """
    Режет один markdown-файл на чанки.
    Каждый чанк получает:
      - текст с "хлебной крошкой" из заголовков в начале
      - метаданные: путь к файлу, путь по папкам, путь по заголовкам
    """
    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()

    rel_path = os.path.relpath(filepath, repo_root)
    folder_parts = rel_path.split(os.sep)[:-1]
    folder_path_str = "/".join(folder_parts) if folder_parts else "(корень)"

    md_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=HEADERS_TO_SPLIT_ON, strip_headers=False
    )
    char_splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE, chunk_overlap=config.CHUNK_OVERLAP
    )

    try:
        header_chunks = md_splitter.split_text(text)
    except Exception:
        # Если файл без заголовков или сплиттер споткнулся - работаем с текстом целиком
        header_chunks = [type("obj", (), {"page_content": text, "metadata": {}})()]

    result = []
    for chunk in header_chunks:
        heading_values = list(chunk.metadata.values()) if chunk.metadata else []
        heading_path_str = " > ".join(heading_values) if heading_values else rel_path

        sub_chunks = char_splitter.split_text(chunk.page_content)
        for i, sub in enumerate(sub_chunks):
            # Кладём контекст прямо в текст чанка - модель видит, откуда кусок
            enriched_text = f"[Файл: {rel_path}] [Раздел: {heading_path_str}]\n\n{sub}"
            result.append(
                {
                    "text": enriched_text,
                    "metadata": {
                        "source_path": rel_path,
                        "file_name": os.path.basename(filepath),
                        "folder_path": folder_path_str,
                        "heading_path": heading_path_str,
                        "chunk_index": i,
                    },
                }
            )
    return result


def embed_text(text: str) -> list[float]:
    client = ollama.Client(host=config.OLLAMA_HOST)
    response = client.embeddings(model=config.EMBED_MODEL, prompt=text)
    return response["embedding"]


def get_collection():
    client = chromadb.PersistentClient(path=config.CHROMA_PATH)
    return client.get_or_create_collection(config.COLLECTION_NAME)


REPO_MAP_ID = "__repo_structure_map__"


def build_repo_map_chunks(all_md_files: list[str], repo_root: str) -> list[dict]:
    """
    Строит один или несколько служебных чанков - дерево файлов репозитория.
    Нужен для запросов вида "что вообще есть в этой документации",
    "какие есть разделы", "покажи структуру" - на такие вопросы
    ни один обычный чанк не отвечает, нужен обзор целиком.

    Если репозиторий большой и дерево не влезает в контекст embedding-модели
    за один раз - режем на несколько чанков той же логикой, что и обычные файлы.
    """
    rel_paths = sorted(os.path.relpath(f, repo_root) for f in all_md_files)

    lines = ["Структура markdown-документации репозитория:", ""]
    for rel_path in rel_paths:
        depth = rel_path.count(os.sep)
        indent = "  " * depth
        lines.append(f"{indent}- {rel_path}")

    tree_text = "\n".join(lines)

    char_splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE, chunk_overlap=config.CHUNK_OVERLAP
    )
    parts = char_splitter.split_text(tree_text)

    chunks = []
    for i, part in enumerate(parts):
        chunks.append(
            {
                "id": f"{REPO_MAP_ID}::{i}",
                "text": f"[Служебный документ: карта структуры репозитория, часть {i+1}/{len(parts)}]\n\n{part}",
                "metadata": {
                    "source_path": "__repo_structure__",
                    "file_name": "__repo_structure__",
                    "folder_path": "(служебный документ)",
                    "heading_path": "Карта репозитория",
                    "chunk_index": i,
                },
            }
        )
    return chunks


def run_ingest(full_reindex: bool = False) -> None:
    print("=== Шаг 1/4: синхронизация репозитория ===")
    repo_root = sync_repo()

    print("=== Шаг 2/4: поиск изменённых файлов ===")
    all_md_files = glob.glob(os.path.join(repo_root, "**", "*.md"), recursive=True)
    print(f"Найдено markdown-файлов: {len(all_md_files)}")

    state = {} if full_reindex else load_state()
    collection = get_collection()

    if full_reindex:
        print("Полная переиндексация: очищаю коллекцию...")
        client = chromadb.PersistentClient(path=config.CHROMA_PATH)
        client.delete_collection(config.COLLECTION_NAME)
        collection = get_collection()

    current_files_rel = set()
    files_to_process = []

    for filepath in all_md_files:
        rel_path = os.path.relpath(filepath, repo_root)
        current_files_rel.add(rel_path)
        h = file_hash(filepath)

        if state.get(rel_path) != h:
            files_to_process.append((filepath, rel_path, h))

    # Файлы, которые были в состоянии, но пропали из репозитория - нужно удалить из базы
    removed_files = set(state.keys()) - current_files_rel

    print(f"Новых/изменённых файлов: {len(files_to_process)}")
    print(f"Удалённых файлов: {len(removed_files)}")

    if not files_to_process and not removed_files:
        print("Изменений нет, индекс уже актуален.")
        return

    print("=== Шаг 3/4: удаление устаревших чанков ===")
    for rel_path in removed_files:
        collection.delete(where={"source_path": rel_path})
        del state[rel_path]

    # На изменённые файлы тоже сначала удаляем старые чанки (иначе будут дубли)
    for _, rel_path, _ in files_to_process:
        collection.delete(where={"source_path": rel_path})

    print("=== Шаг 4/4: чанкинг, эмбеддинги и загрузка в Chroma ===")
    total_chunks = 0
    for filepath, rel_path, h in files_to_process:
        chunks = chunk_markdown_file(filepath, repo_root)
        if not chunks:
            state[rel_path] = h
            continue

        ids, embeddings, documents, metadatas = [], [], [], []
        for i, chunk in enumerate(chunks):
            emb = embed_text(chunk["text"])
            ids.append(f"{rel_path}::{i}")
            embeddings.append(emb)
            documents.append(chunk["text"])
            metadatas.append(chunk["metadata"])

        collection.add(ids=ids, embeddings=embeddings, documents=documents, metadatas=metadatas)
        total_chunks += len(chunks)
        state[rel_path] = h
        print(f"  + {rel_path} -> {len(chunks)} чанков")

    # Обновляем служебную карту репозитория (пересоздаём всегда, она дешёвая)
    print("Обновляю карту структуры репозитория...")
    collection.delete(where={"source_path": "__repo_structure__"})
    map_chunks = build_repo_map_chunks(all_md_files, repo_root)

    map_ids, map_embeddings, map_docs, map_metas = [], [], [], []
    for chunk in map_chunks:
        map_ids.append(chunk["id"])
        map_embeddings.append(embed_text(chunk["text"]))
        map_docs.append(chunk["text"])
        map_metas.append(chunk["metadata"])

    if map_ids:
        collection.add(ids=map_ids, embeddings=map_embeddings, documents=map_docs, metadatas=map_metas)
    print(f"  карта репозитория: {len(map_chunks)} чанк(ов)")

    save_state(state)
    print(f"\nГотово. Загружено чанков: {total_chunks}. Всего в базе: {collection.count()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Индексация markdown-репозитория в Chroma")
    parser.add_argument(
        "--full", action="store_true", help="Полная переиндексация всего репозитория с нуля"
    )
    args = parser.parse_args()
    run_ingest(full_reindex=args.full)
