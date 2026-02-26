"""
RAG Engine — ядро системы.
Собирает retriever + prompt + LLM в единый pipeline.
Логирует тайминги каждого этапа.
"""

import time
import uuid
from typing import List, Dict, Any

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate

from app.config import settings
from app.embeddings import E5EmbeddingsWrapper
from app.llm_provider import create_llm, get_model_name
from app.retriever import build_retrieval_pipeline
from app.data_processor import load_and_clean_data, create_documents
from app.logger import get_logger

logger = get_logger("rag_engine")

SYSTEM_PROMPT = """Ты — опытный туристический гид по достопримечательностям России.

КРИТИЧЕСКИ ВАЖНО:
- Используй ТОЛЬКО информацию из контекста ниже
- НИКОГДА не добавляй факты из своих знаний
- Если в контексте нет информации — ответь: "В моей базе нет информации по этому вопросу"
- Лучше дать короткий точный ответ, чем длинный с домыслами

Правила:
1. Отвечай на языке вопроса
2. Упоминай название достопримечательности и город
3. Будь лаконичен: 2-5 предложений
4. Каждое утверждение должно быть подтверждено контекстом

Контекст:
{context}

Вопрос: {input}

Ответ (ТОЛЬКО на основе контекста):"""

PROMPT = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("human", "{input}"),
])


class RAGEngine:
    """Основной RAG pipeline."""

    def __init__(self):
        logger.info("rag_engine_init_start")

        # Embeddings
        self.embeddings = E5EmbeddingsWrapper()

        # Data
        df = load_and_clean_data()
        self.documents = create_documents(df)

        # Retriever (Qdrant + BM25 + Reranker)
        self.retriever = build_retrieval_pipeline(self.documents, self.embeddings)

        # LLM
        self.llm = create_llm()

        self.doc_count = len(self.documents)
        logger.info(
            "rag_engine_ready",
            documents=self.doc_count,
            llm=get_model_name(),
            embedding=settings.embedding_model,
            vector_db="qdrant",
            qdrant_url=settings.qdrant_url,
        )

    def query(self, question: str) -> Dict[str, Any]:
        """
        Выполняет RAG запрос.

        Returns:
            dict с ключами: query_id, answer, context, timings
        """
        query_id = str(uuid.uuid4())[:8]
        timings = {}

        # Retrieval
        t0 = time.time()
        docs = self.retriever.invoke(question)
        timings["retrieval_sec"] = round(time.time() - t0, 3)

        # Format context (ограничиваем длину — 503 часто из-за слишком длинного промпта)
        MAX_CONTEXT_CHARS = 1200
        parts = []
        total = 0
        for doc in docs:
            if total >= MAX_CONTEXT_CHARS:
                break
            text = doc.page_content
            if total + len(text) > MAX_CONTEXT_CHARS:
                text = text[: MAX_CONTEXT_CHARS - total - 3] + "..."
            parts.append(text)
            total += len(text)
        context_text = "\n\n".join(parts)

        # LLM (retry при 503 — Ollama может быть занят загрузкой модели)
        t1 = time.time()
        messages = PROMPT.format_messages(context=context_text, input=question)
        answer = None
        for attempt in range(3):
            try:
                response = self.llm.invoke(messages)
                answer = response.content
                break
            except Exception as e:
                if "503" in str(e) and attempt < 2:
                    wait = (attempt + 1) * 5
                    logger.warning("llm_503_retry", attempt=attempt + 1, wait_sec=wait, error=str(e))
                    time.sleep(wait)
                else:
                    if "503" in str(e) and context_text:
                        answer = (
                            "Ollama вернул 503. Запустите: ollama run qwen3:14b\n\n"
                            "Найденные места:\n" + context_text[:1500]
                        )
                        logger.warning("llm_fallback_503", used_context=True)
                        break
                    raise
        timings["llm_sec"] = round(time.time() - t1, 3)
        timings["total_sec"] = round(time.time() - t0, 3)
        timings["reranking_sec"] = 0.0

        logger.info(
            "query_processed",
            query_id=query_id,
            question=question[:100],
            answer_length=len(answer),
            n_docs=len(docs),
            **timings,
        )

        return {
            "query_id": query_id,
            "question": question,
            "answer": answer,
            "context": docs,
            "timings": timings,
        }