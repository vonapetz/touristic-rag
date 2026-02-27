# Touristic RAG API

RAG-система для ответов на вопросы о туристических достопримечательностях России.

**Стек**: FastAPI · Qdrant · multilingual-E5-large · BAAI/bge-reranker-v2-m3 · BM25 · RRF · Qwen3:14b / OpenAI

---

## Архитектура

```
Вопрос пользователя
        │
        ▼
┌───────────────────┐
│  City Detection   │  Извлечение города из запроса (alias matching)
└────────┬──────────┘
         │
    ┌────┴────────────────────────────┐
    │         Parallel Retrieval      │
    │  ┌─────────────┐ ┌───────────┐ │
    │  │ Qdrant Vec  │ │   BM25    │ │  k=10 кандидатов от каждого
    │  │  (E5 Large) │ │ (cached)  │ │
    │  └──────┬──────┘ └─────┬─────┘ │
    └─────────┼──────────────┼───────┘
              │              │
              ▼              ▼
    ┌──────────────────────────────┐
    │   Reciprocal Rank Fusion     │  SOTA fusion — не зависит от масштабов score
    └──────────────┬───────────────┘
                   │
                   ▼
    ┌──────────────────────────────┐
    │  Cross-Encoder Reranking     │  BAAI/bge-reranker-v2-m3
    │  + Score Threshold (-2.0)   │  Фильтрует нерелевантные документы
    └──────────────┬───────────────┘
                   │ top-3
                   ▼
    ┌──────────────────────────────┐
    │      LLM Generation          │  Qwen3:14b (Ollama) / GPT-3.5-turbo
    │  XML-контекст, T=0.1         │  Контекст 3000 символов
    └──────────────┬───────────────┘
                   │
                   ▼
              Ответ + Sources
```

---

## Быстрый старт

### 1. Настройка окружения

```bash
cp .env.example .env
# Заполнить QDRANT_URL, QDRANT_API_KEY
```

### 2. Запуск с Ollama (локально)

```bash
# Запустить Ollama с моделью
ollama run qwen3:14b

# Установить зависимости
pip install -r requirements.txt

# Запустить API
uvicorn app.main:app --host 0.0.0.0 --port 8080 --reload
```

### 3. Запуск через Docker

```bash
docker-compose up --build
```

### 4. Документация

После запуска: [http://localhost:8080/docs](http://localhost:8080/docs)

---

## API

Все эндпоинты доступны с префиксом `/v1/`.

### POST /v1/query

Синхронный запрос — ждёт полного ответа LLM.

```bash
curl -X POST http://localhost:8080/v1/query \
  -H "Content-Type: application/json" \
  -d '{"question": "Что посмотреть в Москве?", "show_sources": true}'
```

**Ответ:**
```json
{
  "query_id": "550e8400-e29b-41d4-a716-446655440000",
  "question": "Что посмотреть в Москве?",
  "answer": "В Москве стоит посетить Кремль — ...",
  "sources": [
    {
      "name": "Московский Кремль",
      "city": "Москва",
      "lat": 55.7505,
      "lon": 37.6175,
      "relevance_rank": 1,
      "score": 4.82,
      "text_snippet": "Московский Кремль, город Москва. Кремль является...",
      "image": "https://upload.wikimedia.org/...",
      "wikidata": "Q1513"
    }
  ],
  "timing": {
    "total_sec": 3.21,
    "retrieval_sec": 0.45,
    "reranking_sec": 0.18,
    "llm_sec": 2.58
  },
  "cached": false,
  "timestamp": "2025-01-15T12:00:00Z"
}
```

### POST /v1/query/stream

SSE-стриминг — первый токен появляется сразу, не нужно ждать полного ответа.

```bash
curl -X POST http://localhost:8080/v1/query/stream \
  -H "Content-Type: application/json" \
  -d '{"question": "Расскажи о Петергофе"}' \
  --no-buffer
```

**Поток событий:**
```
data: {"token": "Петергоф", "query_id": "abc123", "done": false}
data: {"token": " — дворцово-парковый", "query_id": "abc123", "done": false}
...
data: {"token": "", "query_id": "abc123", "done": true, "sources": [...]}
```

### POST /v1/feedback

Оценка качества ответа. Данные сохраняются в SQLite (`data/feedback.db`).

```bash
curl -X POST http://localhost:8080/v1/feedback \
  -H "Content-Type: application/json" \
  -d '{"query_id": "550e8400...", "rating": 5, "comment": "Отличный ответ"}'
```

### GET /v1/health

Реальная проверка доступности: пингует Qdrant, возвращает `"degraded"` если недоступен.

```json
{
  "status": "healthy",
  "qdrant": "ok",
  "documents": 239,
  "llm_model": "qwen3:14b",
  "uptime_sec": 3600.5,
  "cache_size": 42
}
```

---

## Конфигурация

Все параметры задаются через `.env`. Ключевые настройки:

| Переменная | По умолчанию | Описание |
|---|---|---|
| `LLM_PROVIDER` | `local` | `local` (Ollama) или `openai` |
| `LOCAL_LLM_MODEL` | `qwen3:14b` | Модель Ollama |
| `QDRANT_URL` | — | URL Qdrant Cloud или self-hosted |
| `QDRANT_API_KEY` | — | API ключ Qdrant |
| `CROSS_ENCODER_MODEL` | `BAAI/bge-reranker-v2-m3` | Реранкер (SOTA 2024) |
| `MAX_CONTEXT_CHARS` | `3000` | Максимум символов контекста для LLM |
| `RERANKER_SCORE_THRESHOLD` | `-2.0` | Порог отсева нерелевантных документов |
| `RRF_K` | `60` | Параметр Reciprocal Rank Fusion |
| `LLM_TEMPERATURE` | `0.1` | Температура LLM (низкая = меньше галлюцинаций) |
| `RATE_LIMIT` | `20/minute` | Лимит запросов на `/v1/query` |
| `CORS_ORIGINS` | `["*"]` | Разрешённые origins (ограничить в prod) |
| `QUERY_CACHE_TTL` | `3600` | TTL кеша запросов (секунды) |
| `BM25_CACHE_PATH` | `data/processed/bm25.pkl` | Путь к кешу BM25 индекса |
| `FORCE_REBUILD_BM25` | `false` | Принудительно пересобрать BM25 |
| `DEBUG_MODE` | `false` | Включает `/v1/config` endpoint |

---

## Retrieval Pipeline

### Reciprocal Rank Fusion (RRF)

Гибридный поиск объединяет Qdrant (dense) и BM25 (sparse) через RRF — SOTA метод, не зависящий от масштабов score разных систем:

```
score(doc) = Σ 1 / (k + rank_in_list + 1)
```

Дедупликация по первым 200 символам контента. Параметр `k=60` (стандартный).

### Реранкинг с порогом

`BAAI/bge-reranker-v2-m3` — многоязычный cross-encoder (SOTA 2024, лучше для русского чем `mmarco-mMiniLMv2`). После реранкинга применяется `score_threshold`: документы ниже порога отфильтровываются (минимум 1 документ всегда возвращается).

Score сохраняется в `SourceInfo.score` для отладки.

### City Detection + Fallback

Если в запросе обнаруживается название города (через alias matching с морфологическими вариантами), Qdrant фильтрует по `city`. Результаты фильтрации **также проходят реранкинг** (ранее bypass, теперь исправлено).

Если city filter даёт < `k_final` результатов — автоматический fallback на полный hybrid search.

### BM25 Cache

BM25 индекс сериализуется в `data/processed/bm25.pkl` и загружается при старте. При изменении датасета — установить `FORCE_REBUILD_BM25=true`.

---

## Кеш запросов

Идентичные запросы (SHA-256 от `question.lower().strip()`) возвращаются из кеша без вызова LLM. TTL и максимальный размер настраиваются. Признак кеша виден в ответе: `"cached": true`.

---

## Feedback

Оценки сохраняются в SQLite (`data/feedback.db`), не теряются при рестарте.

```bash
# Просмотр оценок
sqlite3 data/feedback.db "SELECT rating, COUNT(*) FROM feedback GROUP BY rating"
```

---

## Разработка

### Пересборка индекса

```bash
python scripts/build_index.py
```

### Принудительный сброс BM25 кеша

```bash
FORCE_REBUILD_BM25=true uvicorn app.main:app --port 8080
```

### Debug конфигурации

```bash
DEBUG_MODE=true uvicorn app.main:app --port 8080
# GET http://localhost:8080/v1/config
```

### Тестирование streaming

```python
import httpx

with httpx.stream("POST", "http://localhost:8080/v1/query/stream",
                  json={"question": "Что такое Эрмитаж?"}) as r:
    for line in r.iter_lines():
        if line.startswith("data: "):
            print(line[6:])
```

---

## Данные

Датасет: туристические достопримечательности России (~14 000 → 239 уникальных записей после очистки).

Pipeline обработки:
1. Базовая фильтрация (длина описания, наличие Name и City)
2. TF-IDF фильтрация подозрительных токенов (`min_df=3`, `max_df=0.7`)
3. Дедупликация по (Name, City) — выбирается лучший вариант
4. Обогащение: `rich_text = "Name, город City. Description"`
5. Валидация координат `lat ∈ [-90, 90]`, `lon ∈ [-180, 180]`

---

## Зависимости

| Категория | Библиотека | Назначение |
|---|---|---|
| API | FastAPI, uvicorn | HTTP сервер |
| Rate limiting | slowapi | Ограничение запросов |
| LLM | langchain-openai | OpenAI / Ollama |
| Embeddings | langchain-huggingface, sentence-transformers | E5 Large |
| Vector DB | qdrant-client, langchain-qdrant | Qdrant |
| Sparse search | rank-bm25 | BM25 |
| Data | pandas, scikit-learn, gdown | Обработка данных |
| Persistence | aiosqlite | SQLite для feedback |
| Logging | structlog | Структурированные логи |
