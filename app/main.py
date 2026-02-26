"""
FastAPI приложение.
Эндпоинты: /api/query, /api/feedback, /api/health, /api/config
"""

import os
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse

from app.config import settings
from app.logger import setup_logging, get_logger
from app.models import (
    QueryRequest, QueryResponse, SourceInfo, ModelInfo, TimingInfo,
    FeedbackRequest, FeedbackResponse, HealthResponse,
)
from app.rag_engine import RAGEngine
from app.llm_provider import get_model_name

engine: RAGEngine = None
start_time: float = 0
feedback_store: dict = {}
logger = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global engine, start_time, logger

    os.makedirs("logs", exist_ok=True)
    os.makedirs("data/raw", exist_ok=True)
    os.makedirs("data/processed", exist_ok=True)

    setup_logging()
    logger = get_logger("api")
    logger.info("starting_application")

    start_time = time.time()
    engine = RAGEngine()

    logger.info("application_ready", port=settings.port)
    yield
    logger.info("shutting_down")


app = FastAPI(
    title="Touristic RAG API",
    description="RAG-система для ответов о достопримечательностях России. Vector DB: Qdrant Cloud.",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    return RedirectResponse(url="/docs")


@app.get("/api/test-ollama")
async def test_ollama():
    """Проверка: отвечает ли Ollama на простой запрос."""
    try:
        from langchain_core.messages import HumanMessage
        from app.llm_provider import create_llm
        llm = create_llm()
        r = llm.invoke([HumanMessage(content="Скажи только: ок")])
        return {"status": "ok", "response": (r.content or "")[:100]}
    except Exception as e:
        return {"status": "error", "detail": str(e)}


@app.post("/api/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    try:
        result = engine.query(request.question)

        sources = []
        if request.show_sources:
            for i, doc in enumerate(result["context"]):
                m = doc.metadata
                sources.append(SourceInfo(
                    name=m.get("name", "Неизвестно"),
                    city=m.get("city", "?"),
                    lat=m.get("lat"),
                    lon=m.get("lon"),
                    relevance_rank=i + 1,
                ))

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
                llm_model=get_model_name(),
                embedding_model=settings.embedding_model,
                reranker=settings.reranker_type,
            ),
            timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        )

    except Exception as e:
        logger.error("query_failed", error=str(e), question=request.question[:100])
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/feedback", response_model=FeedbackResponse)
async def feedback(request: FeedbackRequest):
    feedback_store[request.query_id] = {
        "rating": request.rating,
        "comment": request.comment,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

    logger.info(
        "feedback_received",
        query_id=request.query_id,
        rating=request.rating,
        comment=request.comment,
    )

    return FeedbackResponse(status="ok", query_id=request.query_id)


@app.get("/api/health", response_model=HealthResponse)
async def health():
    return HealthResponse(
        status="healthy",
        documents=engine.doc_count if engine else 0,
        llm_provider=settings.llm_provider,
        llm_model=get_model_name(),
        embedding_model=settings.embedding_model,
        reranker=settings.reranker_type,
        uptime_sec=round(time.time() - start_time, 1),
    )


@app.get("/api/config")
async def config():
    return {
        "llm_provider": settings.llm_provider,
        "llm_model": get_model_name(),
        "embedding_model": settings.embedding_model,
        "reranker_type": settings.reranker_type,
        "vector_db": "qdrant",
        "qdrant_url": settings.qdrant_url.split("@")[-1] if "@" in settings.qdrant_url else settings.qdrant_url,
        "qdrant_collection": settings.qdrant_collection,
        "retrieval_k_initial": settings.retrieval_k_initial,
        "retrieval_k_final": settings.retrieval_k_final,
        "bm25_weight": settings.bm25_weight,
        "vector_weight": settings.vector_weight,
    }
