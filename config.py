"""
Централизованная конфигурация. Читает .env, если он есть.
Ничего руками тут менять не нужно — все настройки в .env
"""
import os
from dotenv import load_dotenv

load_dotenv()


def _get(name: str, default: str) -> str:
    return os.getenv(name, default)


# Репозиторий
REPO_URL = _get("REPO_URL", "https://github.com/org/repo.git")
REPO_PATH = _get("REPO_PATH", "./data/repo")
REPO_BRANCH = _get("REPO_BRANCH", "main")

# Ollama
OLLAMA_HOST = _get("OLLAMA_HOST", "http://localhost:11434")
EMBED_MODEL = _get("EMBED_MODEL", "nomic-embed-text")
LLM_MODEL = _get("LLM_MODEL", "qwen2.5:14b")

# Chroma
CHROMA_PATH = _get("CHROMA_PATH", "./data/chroma_db")
COLLECTION_NAME = _get("COLLECTION_NAME", "repo_docs")

# Чанкинг
CHUNK_SIZE = int(_get("CHUNK_SIZE", "800"))
CHUNK_OVERLAP = int(_get("CHUNK_OVERLAP", "100"))

# Расширение запроса через LLM перед поиском - помогает найти чанки,
# когда вопрос и документация используют разную терминологию/язык
# (например "материализованные представления" в вопросе vs "MATERIALIZED VIEW" в тексте)
ENABLE_QUERY_EXPANSION = _get("ENABLE_QUERY_EXPANSION", "true").lower() == "true"
# Какой моделью расширять запрос. Пусто = использовать ту же LLM_MODEL.
# Можно указать модель полегче (например qwen2.5:3b) - расширение простое,
# тяжёлая модель для этого не обязательна и это ускорит ответ.
EXPANSION_MODEL = _get("EXPANSION_MODEL", "")

# Файл с состоянием индексации (какие файлы уже проиндексированы и с каким хэшем)
STATE_FILE = os.path.join(os.path.dirname(REPO_PATH.rstrip("/")) or ".", "index_state.json")

# API
API_HOST = _get("API_HOST", "0.0.0.0")
API_PORT = int(_get("API_PORT", "8000"))
