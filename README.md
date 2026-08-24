# Repo RAG — RAG-база из markdown-документации GitHub-репозитория + Ollama
###.env.example
### Скопируй этот файл в .env и поправь под себя:

### ---- Репозиторий ----
### URL публичного или приватного (тогда нужен токен в URL) репозитория
REPO_URL=...
### Локальная папка, куда будет склонирован репозиторий
REPO_PATH=./data/repo
### Ветка, которую тянем
REPO_BRANCH=main

### ---- Ollama ----
OLLAMA_HOST=http://localhost:11434
EMBED_MODEL=nomic-embed-text
LLM_MODEL=qwen2.5:14b

### ---- Векторная БД (Chroma, локально на диске) ----
CHROMA_PATH=./data/chroma_db
COLLECTION_NAME=repo_docs

### ---- Чанкинг ----
CHUNK_SIZE=800
CHUNK_OVERLAP=100

### ---- API ----
API_HOST=0.0.0.0
API_PORT=8000


Что делает проект:
1. Клонирует / обновляет (`git pull`) репозиторий с GitHub
2. Режет все `.md`-файлы на чанки **с сохранением структуры** (заголовки H1-H4 + путь по папкам кладутся в метаданные и прямо в текст каждого чанка)
3. Индексирует изменения **инкрементально** — при повторном запуске не пересчитывает то, что не менялось
4. Отдаёт API (`/query`) для вопросов к нейронке с подмешанным контекстом из документации

---

## Шаг 0. Что должно быть готово заранее

- Ubuntu с видеокартой (у тебя 4080), NVIDIA-драйверы установлены (`nvidia-smi` должен показывать карту)
- Установлен и запущен **Ollama**:
  ```bash
  curl -fsSL https://ollama.com/install.sh | sh
  ```
  Проверь, что сервис работает:
  ```bash
  systemctl status ollama
  ```
- Скачаны две модели: LLM и embedding-модель:
  ```bash
  ollama pull qwen2.5:14b
  ollama pull nomic-embed-text
  ```
- Python 3.12 (`python3 --version`)
###Установка 3.12.7 Python на новых версиях ubuntu:
bash
### Зависимости для сборки Python
sudo apt update
sudo apt install -y build-essential libssl-dev zlib1g-dev libbz2-dev \
  libreadline-dev libsqlite3-dev curl git libncursesw5-dev xz-utils \
  tk-dev libxml2-dev libxmlsec1-dev libffi-dev liblzma-dev

###После установки добавь в ~/.bashrc:
echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.bashrc
echo '[[ -d $PYENV_ROOT/bin ]] && export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.bashrc
echo 'eval "$(pyenv init -)"' >> ~/.bashrc
source ~/.bashrc

pyenv install 3.12.7

###Создай venv нужной версией:
cd ~/repo-rag
rm -rf venv
~/.pyenv/versions/3.12.7/bin/python3.12 -m venv venv
source venv/bin/activate
python --version   # должно показать 3.12.7
pip install --upgrade pip
pip install -r requirements.txt


### Установка pyenv
curl https://pyenv.run | bash

- Git (`git --version`)

---

## Шаг 1. Разместить проект и поставить зависимости

Скопируй все файлы проекта в отдельную папку, например `~/repo-rag`, затем:

```bash
cd ~/repo-rag
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

> Дальше во всех шагах предполагается, что venv активирован (`source venv/bin/activate`).
> Если открываешь новый терминал — активируй venv заново.

---

## Шаг 2. Настроить конфиг

```bash
cp .env.example .env
nano .env
```

Обязательно поменяй:
- `REPO_URL` — на реальный адрес твоего репозитория
  - если репозиторий **приватный**, используй URL с токеном:
    `https://<твой_токен>@github.com/org/repo.git`
    (токен создаётся в GitHub: Settings → Developer settings → Personal access tokens, права достаточно `repo:read`)
- `REPO_BRANCH` — если ветка не `main` (например `master`)
- `LLM_MODEL` / `EMBED_MODEL` — если называешь модели по-другому, проверь точное имя командой `ollama list`

Остальное (пути, размер чанков, порт) можно оставить по умолчанию.

---

## Шаг 3. Первая индексация

```bash
python ingest.py
```

Что произойдёт:
- Репозиторий склонируется в `./data/repo`
- Все `.md`-файлы порежутся на чанки
- Для каждого чанка через Ollama посчитается эмбеддинг
- Всё загрузится в локальную базу `./data/chroma_db`
- Появится файл `index_state.json` — в нём хэши файлов, по которым потом определяется, что изменилось

В конце ты увидишь что-то вроде:
```
Готово. Загружено чанков: 342. Всего в базе: 342
```

Если увидишь ошибку — см. раздел «Возможные проблемы» внизу.

---

## Шаг 4. Проверить, что RAG отвечает (без API, из консоли)

```bash
python rag.py Как настроить окружение для разработки?
```

Ты увидишь ответ модели и список файлов-источников, откуда взята информация. Если ответ странный или пустой — проверь, что вопрос действительно пересекается с содержимым репозитория.

---

## Шаг 5. Запустить API-сервер

```bash
python api.py
```

Сервер поднимется на `http://0.0.0.0:8000` (порт можно поменять в `.env`).
Чтобы он не умирал при закрытии терминала — смотри раздел «Автозапуск» ниже.

### Проверка, что сервер жив:
```bash
curl http://localhost:8000/health
```

### Запрос к нейронке с RAG:
```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "Как работает авторизация в проекте?"}'
```

Ответ придёт в формате:
```json
{
  "answer": "Авторизация реализована через... (см. auth/README.md)",
  "sources": [
    {"file": "docs/auth/README.md", "section": "Auth > OAuth Flow"}
  ]
}
```

### Фильтр по папке (если нужно искать только в одном разделе документации):
```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "Как накатить миграцию?", "folder_filter": "docs/database"}'
```

### Обновить индекс после изменений в репозитории (git pull + переиндексация только нового):
```bash
curl -X POST http://localhost:8000/reindex
```
Или полная переиндексация с нуля (если сильно поменялась структура репозитория):
```bash
curl -X POST "http://localhost:8000/reindex?full=true"
```

---

## Шаг 6. Обращаться с других устройств по локальной сети

1. Узнай локальный IP машины с видеокартой:
   ```bash
   ip a | grep inet
   ```
   Найди что-то вроде `192.168.1.50`

2. С любого другого устройства в той же сети (телефон, ноутбук):
   ```bash
   curl -X POST http://192.168.1.50:8000/query \
     -H "Content-Type: application/json" \
     -d '{"question": "..."}'
   ```

3. Если не отвечает — проверь firewall:
   ```bash
   sudo ufw allow 8000/tcp
   ```

> Порт не пробрасывай наружу в интернет напрямую. Если понадобится доступ вне дома — подними Tailscale (`https://tailscale.com`), это займёт 5 минут и безопаснее любых альтернатив.

---

## Шаг 7. Автоматизация — обновление репозитория по расписанию

Чтобы база сама обновлялась, например, каждый час, добавь cron-задачу:

```bash
crontab -e
```

Добавь строку (поправь пути под себя):
```
0 * * * * cd /home/user/repo-rag && /home/user/repo-rag/venv/bin/python ingest.py >> /home/user/repo-rag/ingest.log 2>&1
```

Это будет вызывать `ingest.py` каждый час, инкрементально подтягивая изменения из репозитория.

---

## Шаг 8. Автозапуск API как systemd-сервис (чтобы жил после перезагрузки)

Создай файл сервиса:
```bash
sudo nano /etc/systemd/system/repo-rag.service
```

Содержимое (поменяй пути и пользователя под себя):
```ini
[Unit]
Description=Repo RAG API
After=network.target

[Service]
Type=simple
User=твой_юзер
WorkingDirectory=/home/твой_юзер/repo-rag
ExecStart=/home/твой_юзер/repo-rag/venv/bin/python api.py
Restart=on-failure

[Install]
WantedBy=multi-user.target
```

Включить и запустить:
```bash
sudo systemctl daemon-reload
sudo systemctl enable repo-rag
sudo systemctl start repo-rag
sudo systemctl status repo-rag
```

Логи:
```bash
journalctl -u repo-rag -f
```

---

## Структура проекта

```
repo-rag/
├── .env.example      # шаблон конфига
├── .env               # твой конфиг (создаётся из шаблона, в .gitignore)
├── requirements.txt
├── config.py          # чтение настроек из .env
├── git_sync.py        # клонирование/обновление репозитория
├── ingest.py          # чанкинг + инкрементальная индексация в Chroma
├── rag.py             # поиск + запрос к LLM
├── api.py             # FastAPI сервер
├── README.md          # этот файл
└── data/
    ├── repo/          # склонированный репозиторий (создаётся автоматически)
    └── chroma_db/     # векторная база (создаётся автоматически)
```

---

## Про иерархию файлов (важно понимать)

Сама Chroma — плоская база векторов, дерева папок в ней нет. Но иерархия **не теряется**, потому что:

- у каждого чанка в метаданных лежит `source_path` (полный путь к файлу) и `folder_path` (путь по папкам)
- у каждого чанка есть `heading_path` — путь по заголовкам внутри файла (`H1 > H2 > H3`)
- сам текст чанка начинается с пометки `[Файл: ...] [Раздел: ...]`, так что даже модель "видит" контекст, а не только сырой текст

Это позволяет:
- фильтровать поиск по конкретной папке (`folder_filter` в запросе)
- видеть в ответе, из какого файла и раздела взята информация (поле `sources`)

Если структура репозитория для тебя критична (например, нужно уметь спрашивать "покажи оглавление раздела X"), это отдельная задача поверх RAG — можно дополнительно хранить дерево файлов в обычном JSON и обращаться к нему напрямую, без векторного поиска. Скажи, если это актуально — можно добавить.

---

## Возможные проблемы

**`ollama.embeddings` падает с ошибкой соединения**
→ проверь, что Ollama запущена: `systemctl status ollama` или `ollama serve` в отдельном терминале, и что `OLLAMA_HOST` в `.env` верный.

**`git clone` просит логин/пароль**
→ репозиторий приватный, нужен токен в URL (см. Шаг 2) либо настроенный SSH-ключ (`git@github.com:org/repo.git`).

**Модель отвечает "не знаю", хотя информация есть в репозитории**
→ увеличь `top_k` в запросе (например до 8-10) или уменьши `CHUNK_SIZE` в `.env` для более точного попадания — но тогда нужна полная переиндексация (`python ingest.py --full`).

**Индексация каждый раз обрабатывает все файлы заново**
→ проверь, что файл `index_state.json` не удаляется между запусками и что путь `STATE_FILE` в `config.py` указывает в постоянное место.

**Хочу другую векторную БД (Qdrant) вместо Chroma**
→ меняется только `get_collection()` в `ingest.py` и вызовы `.query()`/`.add()` — вся остальная логика (чанкинг, метаданные, API) не затрагивается.
