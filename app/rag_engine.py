"""
RAG Engine — ядро системы.
Собирает retriever + prompt + LLM в единый pipeline.

Улучшения:
- QueryCache: точный кеш запросов с TTL (SHA-256 ключ)
- asyncio.sleep вместо time.sleep в retry логике
- astream_query: стриминг токенов для SSE endpoint
- Расширенный контекст (3000 символов) с разделителями
- Системный промпт с XML-тегами (улучшает Faithfulness)
- Реальный reranking_sec из retriever
- Полный UUID вместо [:8]
"""

import asyncio
import time
import uuid
from hashlib import sha256
from typing import AsyncGenerator, Dict, Any, List, Optional, Tuple

from langchain_core.documents import Document
from langchain_core.messages import BaseMessage
from langchain_core.prompts import ChatPromptTemplate

from app.config import settings
from app.embeddings import E5EmbeddingsWrapper
from app.llm_provider import create_llm, get_model_name
from app.retriever import RetrieverWrapper, build_retrieval_pipeline
from app.data_processor import load_and_clean_data, create_documents
from app.logger import get_logger

logger = get_logger("rag_engine")


# ============================================================================
# Системный промпт с XML-тегами (улучшает Faithfulness)
# ============================================================================

SYSTEM_PROMPT = """Ты — опытный туристический гид по достопримечательностям России.

СТРОГИЕ ПРАВИЛА:
1. Отвечай ИСКЛЮЧИТЕЛЬНО на основе информации внутри тегов <context>
2. НЕ добавляй факты, даты, цифры, которых нет в контексте
3. Если ответа нет в контексте — скажи "В моей базе нет данных по этому вопросу"
4. Отвечай на языке вопроса
5. 2-4 предложения. Каждый факт — только из контекста

<context>
{context}
</context>"""

PROMPT = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("human", "{input}"),
])


# ============================================================================
# Query Cache
# ============================================================================

class QueryCache:
    """
    Простой in-process кеш запросов с TTL и LRU-подобным вытеснением.
    Ключ — SHA-256 от нормализованного вопроса (lowercase + strip).
    """

    def __init__(self, ttl: int, max_size: int):
        self._store: Dict[str, Tuple[dict, float]] = {}
        self.ttl = ttl
        self.max_size = max_size

    def _key(self, question: str) -> str:
        return sha256(question.lower().strip().encode()).hexdigest()

    def get(self, question: str) -> Optional[dict]:
        key = self._key(question)
        if key in self._store:
            result, ts = self._store[key]
            if time.monotonic() - ts < self.ttl:
                return result
            del self._store[key]
        return None

    def set(self, question: str, result: dict) -> None:
        if len(self._store) >= self.max_size:
            # Вытесняем самый старый элемент
            oldest_key = min(self._store, key=lambda k: self._store[k][1])
            del self._store[oldest_key]
        key = self._key(question)
        self._store[key] = (result, time.monotonic())

    def size(self) -> int:
        return len(self._store)


# ============================================================================
# RAG Engine
# ============================================================================

class RAGEngine:
    """Основной RAG pipeline."""

    def __init__(self):
        logger.info("rag_engine_init_start")

        # Embeddings
        self.embeddings = E5EmbeddingsWrapper()

        # Data
        df = load_and_clean_data()
        self.documents = create_documents(df)

        # Retriever (Qdrant + BM25 + RRF + Reranker)
        self.retriever: RetrieverWrapper = build_retrieval_pipeline(
            self.documents, self.embeddings
        )

        # LLM
        self.llm = create_llm()

        # Cache
        self.cache = QueryCache(
            ttl=settings.query_cache_ttl,
            max_size=settings.query_cache_max_size,
        )

        self.doc_count = len(self.documents)
        self._model_name = get_model_name()

        logger.info(
            "rag_engine_ready",
            documents=self.doc_count,
            llm=self._model_name,
            embedding=settings.embedding_model,
            reranker=settings.cross_encoder_model,
            vector_db="qdrant",
            context_chars=settings.max_context_chars,
        )

    # ------------------------------------------------------------------ #
    # Форматирование контекста                                              #
    # ------------------------------------------------------------------ #

    def _format_context(self, docs: List[Document]) -> str:
        """
        Объединяет документы в строку контекста с разделителями.
        Ограничивает общую длину settings.max_context_chars.
        """
        parts = []
        total = 0
        separator = "\n---\n"
        sep_len = len(separator)

        for doc in docs:
            if total >= settings.max_context_chars:
                break
            text = doc.page_content
            remaining = settings.max_context_chars - total
            if len(text) > remaining:
                text = text[:remaining - 3] + "..."
            parts.append(text)
            total += len(text) + sep_len

        return separator.join(parts)

    # ------------------------------------------------------------------ #
    # Синхронный query (вызывается через asyncio.to_thread из endpoint)   #
    # ------------------------------------------------------------------ #

    def query(self, question: str) -> Dict[str, Any]:
        """
        Выполняет RAG запрос.
        Синхронный метод — должен вызываться через asyncio.to_thread.

        Returns:
            dict с ключами: query_id, answer, context, timings, cached
        """
        # Проверяем кеш
        cached = self.cache.get(question)
        if cached:
            logger.info("cache_hit", question=question[:100])
            return cached

        query_id = str(uuid.uuid4())
        t_start = time.perf_counter()

        # Retrieval + Reranking (раздельные тайминги)
        docs, retrieval_timings = self.retriever.invoke_with_timings(question)

        # Форматируем контекст
        context_text = self._format_context(docs)

        # LLM с retry на 503 (Ollama может быть занят загрузкой модели)
        t_llm = time.perf_counter()
        answer = self._invoke_llm_sync(context_text, question, query_id)
        llm_sec = round(time.perf_counter() - t_llm, 3)

        timings = {
            "retrieval_sec": retrieval_timings["retrieval_sec"],
            "reranking_sec": retrieval_timings["reranking_sec"],
            "llm_sec": llm_sec,
            "total_sec": round(time.perf_counter() - t_start, 3),
        }

        logger.info(
            "query_processed",
            query_id=query_id,
            question=question[:100],
            answer_length=len(answer),
            n_docs=len(docs),
            **timings,
        )

        result = {
            "query_id": query_id,
            "question": question,
            "answer": answer,
            "context": docs,
            "timings": timings,
            "cached": False,
        }
        self.cache.set(question, result)
        return result

    def _invoke_llm_sync(self, context: str, question: str, query_id: str) -> str:
        """
        Синхронный вызов LLM с retry на 503.
        time.sleep здесь допустим — метод вызывается через asyncio.to_thread,
        значит блокирует только worker thread, не event loop.
        """
        messages = PROMPT.format_messages(context=context, input=question)

        for attempt in range(3):
            try:
                response = self.llm.invoke(messages)
                return response.content
            except Exception as e:
                err_str = str(e)
                if "503" in err_str and attempt < 2:
                    wait = (attempt + 1) * 5
                    logger.warning(
                        "llm_503_retry",
                        query_id=query_id,
                        attempt=attempt + 1,
                        wait_sec=wait,
                    )
                    time.sleep(wait)
                else:
                    logger.error("llm_failed", query_id=query_id, error=err_str)
                    raise

        # Не должны сюда попасть, но на всякий случай
        raise RuntimeError("LLM retry exhausted")

    # ------------------------------------------------------------------ #
    # Async streaming (для SSE endpoint)                                   #
    # ------------------------------------------------------------------ #

    async def astream_query(
        self,
        question: str,
        docs: Optional[List[Document]] = None,
    ) -> AsyncGenerator[str, None]:
        """
        Стриминг токенов LLM.
        Если docs не переданы — выполняет retrieval в thread pool.
        Передача docs снаружи позволяет избежать двойного retrieval в endpoint.
        """
        if docs is None:
            docs, _ = await asyncio.to_thread(
                self.retriever.invoke_with_timings, question
            )

        context_text = self._format_context(docs)
        messages = PROMPT.format_messages(context=context_text, input=question)

        async for chunk in self.llm.astream(messages):
            if chunk.content:
                yield chunk.content

    async def aquery(self, question: str) -> Dict[str, Any]:
        """
        Асинхронная обёртка над query().
        Запускает синхронный query() в thread pool.
        """
        return await asyncio.to_thread(self.query, question)
