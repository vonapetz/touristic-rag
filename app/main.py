"""
FastAPI приложение — v1 API.

Улучшения:
- /v1/ prefix для всех эндпоинтов
- Rate limiting через slowapi (20 req/min по умолчанию)
- CORS origins из конфига (не hardcode "*")
- Generic error messages — внутренние детали не утекают клиенту
- SQLite для хранения feedback (persistent, не теряется при рестарте)
- Реальный health check (проверяет доступность Qdrant)
- SSE streaming endpoint /v1/query/stream
- asyncio.to_thread для всех sync операций
- /api/config только в debug_mode
"""

import asyncio
import json
import sqlite3
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse, StreamingResponse
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from app.config import settings
from app.logger import setup_logging, get_logger
from app.models import (
    QueryRequest, QueryResponse, SourceInfo, ModelInfo, TimingInfo,
    FeedbackRequest, FeedbackResponse, HealthResponse, StreamToken,
)
from app.rag_engine import RAGEngine
from app.llm_provider import get_model_name

# ============================================================================
# Rate limiter
# ============================================================================

limiter = Limiter(key_func=get_remote_address)

# ============================================================================
# Globals
# ============================================================================

engine: Optional[RAGEngine] = None
start_time: float = 0.0
_model_name: str = ""
logger = None

# ============================================================================
# SQLite feedback
# ============================================================================

def _init_feedback_db() -> None:
    Path(settings.feedback_db_path).parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(settings.feedback_db_path) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS feedback (
                query_id TEXT PRIMARY KEY,
                rating   INTEGER NOT NULL,
                comment  TEXT,
                ts       TEXT NOT NULL
            )
        """)
        conn.commit()


def _save_feedback(query_id: str, rating: int, comment: Optional[str], ts: str) -> None:
    with sqlite3.connect(settings.feedback_db_path) as conn:
        conn.execute(
            "INSERT OR REPLACE INTO feedback (query_id, rating, comment, ts) VALUES (?, ?, ?, ?)",
            (query_id, rating, comment, ts),
        )
        conn.commit()


# ============================================================================
# Lifespan
# ============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    global engine, start_time, _model_name, logger

    setup_logging()
    logger = get_logger("api")

    # Создаём директории
    for d in ("logs", "data/raw", "data/processed"):
        Path(d).mkdir(parents=True, exist_ok=True)

    # Инициализируем SQLite для feedback
    _init_feedback_db()

    logger.info("starting_application")
    start_time = time.monotonic()

    try:
        # RAGEngine — тяжёлая инициализация в thread pool, не блокирует event loop
        engine = await asyncio.to_thread(RAGEngine)
        _model_name = get_model_name()
    except Exception as e:
        logger.error("startup_failed", error=str(e))
        raise

    logger.info("application_ready", port=settings.port)
    yield
    logger.info("shutting_down")


# ============================================================================
# FastAPI app
# ============================================================================

app = FastAPI(
    title="Touristic RAG API",
    description="RAG-система для ответов о достопримечательностях России. "
                "Vector DB: Qdrant. Реранкинг: BAAI/bge-reranker-v2-m3. RRF fusion.",
    version="2.1.0",
    lifespan=lifespan,
)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "Authorization"],
)


# ============================================================================
# Helpers
# ============================================================================

def _build_sources(docs, show: bool) -> list:
    if not show:
        return []
    sources = []
    for i, doc in enumerate(docs):
        m = doc.metadata
        sources.append(SourceInfo(
            name=m.get("name", "Неизвестно"),
            city=m.get("city", "?"),
            lat=m.get("lat"),
            lon=m.get("lon"),
            relevance_rank=i + 1,
            score=m.get("reranker_score"),
            text_snippet=doc.page_content[:200] if doc.page_content else None,
            image=m.get("image"),
            wikidata=m.get("wikidata"),
        ))
    return sources


# ============================================================================
# Endpoints
# ============================================================================

@app.get("/")
async def root():
    return RedirectResponse(url="/docs")


@app.post("/v1/query", response_model=QueryResponse)
@limiter.limit(settings.rate_limit)
async def query(request: QueryRequest, req: Request):
    try:
        # Запускаем синхронный query в thread pool — не блокирует event loop
        result = await asyncio.to_thread(engine.query, request.question)

        sources = _build_sources(result["context"], request.show_sources)
        timings = result["timings"]

        return QueryResponse(
            query_id=result["query_id"],
            question=request.question,
            answer=result["answer"],
            sources=sources,
            timing=TimingInfo(
                total_sec=timings["total_sec"],
                retrieval_sec=timings["retrieval_sec"],
                reranking_sec=timings["reranking_sec"],
                llm_sec=timings["llm_sec"],
            ),
            model_info=ModelInfo(
                llm_provider=settings.llm_provider,
                llm_model=_model_name,
                embedding_model=settings.embedding_model,
                reranker=settings.reranker_type,
            ),
            timestamp=datetime.now(timezone.utc),
            cached=result.get("cached", False),
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error("query_failed", error=str(e), question=request.question[:100], exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")


@app.post("/v1/query/stream")
@limiter.limit(settings.rate_limit)
async def query_stream(request: QueryRequest, req: Request):
    """
    SSE стриминг ответа LLM.
    Формат: data: {"token": "...", "query_id": "...", "done": false}
    Последнее сообщение: {"token": "", "query_id": "...", "done": true, "sources": [...]}
    """
    query_id = str(uuid.uuid4())

    # Выполняем retrieval один раз — передаём docs и в sources, и в astream_query
    docs, _ = await asyncio.to_thread(
        engine.retriever.invoke_with_timings, request.question
    )
    sources = _build_sources(docs, request.show_sources)
    sources_data = [s.model_dump() for s in sources]

    async def event_stream():
        try:
            # docs передаём явно — избегаем повторного retrieval внутри astream_query
            async for token in engine.astream_query(request.question, docs=docs):
                data = json.dumps(
                    {"token": token, "query_id": query_id, "done": False},
                    ensure_ascii=False,
                )
                yield f"data: {data}\n\n"
        except Exception as e:
            logger.error("stream_failed", query_id=query_id, error=str(e))
            yield f"data: {json.dumps({'error': 'Stream failed', 'done': True})}\n\n"
            return

        # Финальное сообщение с sources
        final = json.dumps(
            {"token": "", "query_id": query_id, "done": True, "sources": sources_data},
            ensure_ascii=False,
        )
        yield f"data: {final}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/v1/feedback", response_model=FeedbackResponse)
async def feedback(request: FeedbackRequest):
    ts = datetime.now(timezone.utc).isoformat()
    await asyncio.to_thread(
        _save_feedback, request.query_id, request.rating, request.comment, ts
    )
    logger.info(
        "feedback_saved",
        query_id=request.query_id,
        rating=request.rating,
        has_comment=bool(request.comment),
    )
    return FeedbackResponse(status="ok", query_id=request.query_id)


@app.get("/v1/health", response_model=HealthResponse)
async def health():
    # Реально проверяем доступность Qdrant
    qdrant_ok = False
    try:
        await asyncio.to_thread(
            engine.retriever._inner.qdrant.client.get_collections
        )
        qdrant_ok = True
    except Exception:
        pass

    status = "healthy" if qdrant_ok else "degraded"

    return HealthResponse(
        status=status,
        qdrant="ok" if qdrant_ok else "unreachable",
        documents=engine.doc_count if engine else 0,
        llm_provider=settings.llm_provider,
        llm_model=_model_name,
        embedding_model=settings.embedding_model,
        reranker=settings.reranker_type,
        uptime_sec=round(time.monotonic() - start_time, 1),
        cache_size=engine.cache.size() if engine else 0,
    )


@app.get("/v1/config")
async def config():
    """Доступен только в debug_mode=True."""
    if not settings.debug_mode:
        raise HTTPException(status_code=404, detail="Not found")
    return {
        "llm_provider": settings.llm_provider,
        "llm_model": _model_name,
        "embedding_model": settings.embedding_model,
        "reranker_type": settings.reranker_type,
        "reranker_model": settings.cross_encoder_model,
        "vector_db": "qdrant",
        "qdrant_collection": settings.qdrant_collection,
        "retrieval_k_initial": settings.retrieval_k_initial,
        "retrieval_k_final": settings.retrieval_k_final,
        "rrf_k": settings.rrf_k,
        "max_context_chars": settings.max_context_chars,
        "score_threshold": settings.reranker_score_threshold,
        "cache_ttl": settings.query_cache_ttl,
        "cache_size": engine.cache.size() if engine else 0,
    }
