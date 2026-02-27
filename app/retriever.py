"""
Retrieval pipeline:
1. Qdrant Vector Search (с нативной фильтрацией по метаданным)
2. BM25 для keyword search
3. Hybrid Ensemble через Reciprocal Rank Fusion (SOTA)
4. Reranking (Cross-Encoder) с score threshold
5. Metadata-aware filtering (город из запроса) — теперь тоже реранкирует
"""

import pickle
import time
from pathlib import Path
from typing import List, Optional, Any, Tuple

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_community.retrievers import BM25Retriever

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, PointStruct,
    Filter, FieldCondition, MatchValue,
    PayloadSchemaType,
)

from app.config import settings
from app.embeddings import E5EmbeddingsWrapper
from app.logger import get_logger

logger = get_logger("retriever")


# ============================================================================
# Qdrant Retriever
# ============================================================================

class QdrantRetriever:
    """Retriever на базе Qdrant Cloud с нативной фильтрацией."""

    def __init__(self, client: QdrantClient, embeddings: E5EmbeddingsWrapper, collection: str):
        self.client = client
        self.embeddings = embeddings
        self.collection = collection

    def invoke(self, query: str, k: int = None, city_filter: str = None) -> List[Document]:
        if k is None:
            k = settings.retrieval_k_initial

        query_vector = self.embeddings.embed_query(query)

        qdrant_filter = None
        if city_filter:
            qdrant_filter = Filter(
                must=[
                    FieldCondition(
                        key="city",
                        match=MatchValue(value=city_filter),
                    )
                ]
            )

        results = self.client.query_points(
            collection_name=self.collection,
            query=query_vector,
            query_filter=qdrant_filter,
            limit=k,
            with_payload=True,
        )

        documents = []
        for point in results.points:
            payload = point.payload or {}
            metadata = {
                "name": payload.get("name", "Неизвестно"),
                "city": payload.get("city", "?"),
            }
            if payload.get("lat") is not None:
                metadata["lat"] = payload["lat"]
            if payload.get("lon") is not None:
                metadata["lon"] = payload["lon"]
            if payload.get("wikidata"):
                metadata["wikidata"] = payload["wikidata"]
            if payload.get("image"):
                metadata["image"] = payload["image"]

            documents.append(Document(
                page_content=payload.get("text", ""),
                metadata=metadata,
            ))

        return documents


# ============================================================================
# Cross-Encoder Reranker с score threshold
# ============================================================================

class ManualCrossEncoderReranker:
    """
    Reranker через sentence-transformers CrossEncoder.
    Возвращает (docs, scores) — score сохраняется в metadata для API.
    Применяет score_threshold для отсева нерелевантных документов.
    """

    def __init__(self, model_name: str, top_n: int, score_threshold: float):
        from sentence_transformers import CrossEncoder
        logger.info("loading_cross_encoder", model=model_name)
        self.model = CrossEncoder(model_name)
        self.top_n = top_n
        self.score_threshold = score_threshold
        logger.info("cross_encoder_loaded", model=model_name)

    def rerank(
        self, query: str, documents: List[Document]
    ) -> Tuple[List[Document], List[float]]:
        """
        Возвращает (docs, scores) отсортированных по релевантности.
        Документы ниже score_threshold отфильтровываются, но не менее 1.
        Score сохраняется в doc.metadata["reranker_score"] для API.
        """
        if not documents:
            return [], []

        pairs = [[query, doc.page_content] for doc in documents]
        raw_scores = self.model.predict(pairs)

        # Сортировка по убыванию score
        scored = sorted(zip(documents, raw_scores), key=lambda x: x[1], reverse=True)

        # Применяем threshold, но возвращаем минимум 1 документ
        top = scored[:self.top_n]
        filtered = [(d, float(s)) for d, s in top if s > self.score_threshold]
        result = filtered if filtered else [(scored[0][0], float(scored[0][1]))]

        docs_out = []
        scores_out = []
        for doc, score in result:
            doc.metadata["reranker_score"] = round(score, 4)
            docs_out.append(doc)
            scores_out.append(score)

        return docs_out, scores_out


# ============================================================================
# BM25 с кешированием на диск
# ============================================================================

def load_or_build_bm25(
    documents: List[Document],
    cache_path: str,
    k: int,
    force_rebuild: bool = False,
) -> BM25Retriever:
    """
    Загружает BM25 индекс из pickle-файла или строит заново.
    При force_rebuild=True всегда пересобирает.
    """
    path = Path(cache_path)

    if not force_rebuild and path.exists():
        try:
            with open(path, "rb") as f:
                retriever = pickle.load(f)
            retriever.k = k
            logger.info("bm25_loaded_from_cache", path=str(path))
            return retriever
        except Exception as e:
            logger.warning("bm25_cache_load_failed", error=str(e), path=str(path))

    retriever = BM25Retriever.from_documents(documents, k=k)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with open(path, "wb") as f:
            pickle.dump(retriever, f)
        logger.info("bm25_cached", path=str(path), docs=len(documents))
    except Exception as e:
        logger.warning("bm25_cache_save_failed", error=str(e))

    return retriever


# ============================================================================
# Hybrid Retriever с RRF + Reranking
# ============================================================================

class HybridRetrieverWithRerank:
    """
    BM25 + Qdrant → RRF fusion → cross-encoder reranking.

    Reciprocal Rank Fusion (RRF) — SOTA для гибридного поиска.
    Не зависит от масштабов score разных систем.
    """

    def __init__(
        self,
        bm25: BM25Retriever,
        qdrant: QdrantRetriever,
        reranker: Optional[ManualCrossEncoderReranker],
        k_initial: int,
        k_final: int,
        rrf_k: int = 60,
    ):
        self.bm25 = bm25
        self.qdrant = qdrant
        self.reranker = reranker
        self.k_initial = k_initial
        self.k_final = k_final
        self.rrf_k = rrf_k

    def _reciprocal_rank_fusion(
        self, lists: List[List[Document]]
    ) -> List[Document]:
        """
        Reciprocal Rank Fusion: score = sum(1 / (k + rank + 1)) по всем спискам.
        Дедупликация по первым 200 символам контента.
        """
        scores: dict = {}
        doc_map: dict = {}

        for ranked_list in lists:
            for rank, doc in enumerate(ranked_list):
                key = doc.page_content[:200]
                scores[key] = scores.get(key, 0.0) + 1.0 / (self.rrf_k + rank + 1)
                if key not in doc_map:
                    doc_map[key] = doc

        sorted_keys = sorted(scores, key=lambda x: scores[x], reverse=True)
        return [doc_map[k] for k in sorted_keys]

    def invoke(self, query: str) -> Tuple[List[Document], dict]:
        """
        Возвращает (docs, timings) с раздельными retrieval_sec и reranking_sec.
        """
        t0 = time.perf_counter()

        bm25_results = self.bm25.invoke(query)
        qdrant_results = self.qdrant.invoke(query, k=self.k_initial)

        merged = self._reciprocal_rank_fusion([qdrant_results, bm25_results])

        t_rerank = time.perf_counter()
        retrieval_sec = round(t_rerank - t0, 3)

        if self.reranker and len(merged) > self.k_final:
            final_docs, _ = self.reranker.rerank(query, merged)
        else:
            final_docs = merged[:self.k_final]

        reranking_sec = round(time.perf_counter() - t_rerank, 3)

        timings = {
            "retrieval_sec": retrieval_sec,
            "reranking_sec": reranking_sec,
        }
        return final_docs, timings


# ============================================================================
# Metadata-aware Retriever
# ============================================================================

class MetadataAwareRetriever:
    """
    Retriever с автоматической фильтрацией по городу из запроса.

    FIX: city bypass теперь тоже проходит через реранкинг.
    Было: city filter → сразу return (bypass reranking!)
    Стало: city filter → k_initial документов → reranking → top k_final
    """

    def __init__(
        self,
        hybrid_retriever: HybridRetrieverWithRerank,
        qdrant_retriever: QdrantRetriever,
        documents: List[Document],
    ):
        self.hybrid = hybrid_retriever
        self.qdrant = qdrant_retriever

        self.cities = set()
        for doc in documents:
            city = doc.metadata.get("city", "").strip()
            if city and city.lower() != "nan":
                self.cities.add(city)

        self.city_aliases = {}
        for city in self.cities:
            c = city.lower()
            self.city_aliases[c] = city
            for src, dst in [("ь", "е"), ("ь", "и"), ("а", "е"), ("а", "у"), ("о", "е")]:
                if c.endswith(src):
                    self.city_aliases[c[:-1] + dst] = city
            if not c.endswith(("ь", "а", "о")):
                self.city_aliases[c + "е"] = city
                self.city_aliases[c + "а"] = city

        logger.info("metadata_retriever_init", cities=len(self.cities), aliases=len(self.city_aliases))

    def _extract_city(self, query: str) -> Optional[str]:
        q = query.lower()
        for alias, city in sorted(self.city_aliases.items(), key=lambda x: -len(x[0])):
            if alias in q:
                return city
        return None

    def invoke(self, query: str) -> Tuple[List[Document], dict]:
        """
        Возвращает (docs, meta) где meta содержит path и timings.
        """
        city = self._extract_city(query)

        if city:
            # FIX: получаем k_initial документов через city filter
            # и затем реранкируем — не bypass!
            t0 = time.perf_counter()
            city_results = self.qdrant.invoke(
                query, k=self.hybrid.k_initial, city_filter=city
            )
            retrieval_sec = round(time.perf_counter() - t0, 3)

            if len(city_results) >= self.hybrid.k_final:
                t_rerank = time.perf_counter()
                if self.hybrid.reranker:
                    final_docs, _ = self.hybrid.reranker.rerank(query, city_results)
                else:
                    final_docs = city_results[:self.hybrid.k_final]
                reranking_sec = round(time.perf_counter() - t_rerank, 3)

                timings = {
                    "retrieval_sec": retrieval_sec,
                    "reranking_sec": reranking_sec,
                }
                logger.debug(
                    "city_filter_path",
                    city=city,
                    candidates=len(city_results),
                    final=len(final_docs),
                )
                return final_docs, {"path": "city_filter+rerank", "city": city, **timings}

            # Fallback: city filter дал мало результатов — hybrid search
            logger.debug("city_filter_fallback", city=city, results=len(city_results))

        # Hybrid search (BM25 + Qdrant + RRF + reranking)
        docs, timings = self.hybrid.invoke(query)

        logger.debug(
            "hybrid_path",
            city_extracted=city,
            results_count=len(docs),
        )
        return docs, {"path": "hybrid", "city": city, **timings}


# ============================================================================
# LangChain Wrapper с поддержкой timing
# ============================================================================

class RetrieverWrapper(BaseRetriever):
    """
    Адаптер MetadataAwareRetriever → LangChain BaseRetriever.
    Добавляет метод invoke_with_timings для получения раздельных таймингов.
    """

    _inner: Any = None

    class Config:
        arbitrary_types_allowed = True
        underscore_attrs_are_private = True

    def __init__(self, inner: MetadataAwareRetriever, **kwargs):
        super().__init__(**kwargs)
        self._inner = inner

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun = None
    ) -> List[Document]:
        docs, _ = self._inner.invoke(query)
        return docs

    def invoke_with_timings(self, query: str) -> Tuple[List[Document], dict]:
        """Возвращает (docs, timings) с retrieval_sec и reranking_sec."""
        docs, meta = self._inner.invoke(query)
        timings = {
            "retrieval_sec": meta.get("retrieval_sec", 0.0),
            "reranking_sec": meta.get("reranking_sec", 0.0),
        }
        return docs, timings


# ============================================================================
# Инициализация Qdrant коллекции
# ============================================================================

def _init_qdrant_collection(
    client: QdrantClient,
    collection: str,
    documents: List[Document],
    embeddings: E5EmbeddingsWrapper,
):
    """Создаёт коллекцию в Qdrant, загружает документы, создаёт payload индексы."""

    collections = [c.name for c in client.get_collections().collections]

    if collection in collections:
        info = client.get_collection(collection)
        if info.points_count > 0:
            logger.info("qdrant_collection_exists", collection=collection, points=info.points_count)
            _ensure_payload_indexes(client, collection)
            return

    # Создаём коллекцию
    dim = embeddings.dimension

    if collection in collections:
        client.delete_collection(collection)

    client.create_collection(
        collection_name=collection,
        vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
    )
    logger.info("qdrant_collection_created", collection=collection, dimension=dim)

    # Загружаем документы батчами
    batch_size = 50
    texts = [doc.page_content for doc in documents]

    for i in range(0, len(documents), batch_size):
        batch_docs = documents[i:i + batch_size]
        batch_texts = texts[i:i + batch_size]

        batch_vectors = embeddings.embed_documents(batch_texts)

        points = []
        for j, (doc, vector) in enumerate(zip(batch_docs, batch_vectors)):
            payload = {
                "text": doc.page_content,
                "name": doc.metadata.get("name", ""),
                "city": doc.metadata.get("city", ""),
            }
            if doc.metadata.get("lat") is not None:
                payload["lat"] = doc.metadata["lat"]
            if doc.metadata.get("lon") is not None:
                payload["lon"] = doc.metadata["lon"]
            if doc.metadata.get("wikidata"):
                payload["wikidata"] = doc.metadata["wikidata"]
            if doc.metadata.get("image"):
                payload["image"] = doc.metadata["image"]

            points.append(PointStruct(id=i + j, vector=vector, payload=payload))

        client.upsert(collection_name=collection, points=points)
        logger.debug("qdrant_batch_uploaded", batch=i // batch_size + 1, points=len(points))

    logger.info("qdrant_upload_complete", total_points=len(documents))

    _ensure_payload_indexes(client, collection)


def _ensure_payload_indexes(client: QdrantClient, collection: str):
    """Создаёт payload индексы для полей фильтрации."""

    indexes_to_create = {
        "city": PayloadSchemaType.KEYWORD,
        "name": PayloadSchemaType.KEYWORD,
    }

    for field_name, field_type in indexes_to_create.items():
        try:
            client.create_payload_index(
                collection_name=collection,
                field_name=field_name,
                field_schema=field_type,
            )
            logger.info("payload_index_created", field=field_name, type=str(field_type))
        except Exception as e:
            if "already exists" in str(e).lower() or "already indexed" in str(e).lower():
                logger.debug("payload_index_exists", field=field_name)
            else:
                logger.warning("payload_index_error", field=field_name, error=str(e))


# ============================================================================
# Сборка pipeline
# ============================================================================

def build_retrieval_pipeline(
    documents: List[Document],
    embeddings: E5EmbeddingsWrapper,
) -> RetrieverWrapper:
    """Собирает полный retrieval pipeline."""

    # Qdrant клиент
    client = QdrantClient(
        url=settings.qdrant_url,
        api_key=settings.qdrant_api_key,
    )
    logger.info("qdrant_connected", url=settings.qdrant_url)

    # Инициализация коллекции + payload индексы
    _init_qdrant_collection(client, settings.qdrant_collection, documents, embeddings)

    # Qdrant retriever
    qdrant_retriever = QdrantRetriever(client, embeddings, settings.qdrant_collection)

    # BM25 с кешированием на диск
    bm25 = load_or_build_bm25(
        documents=documents,
        cache_path=settings.bm25_cache_path,
        k=settings.retrieval_k_initial,
        force_rebuild=settings.force_rebuild_bm25,
    )

    # Reranker (SOTA: BAAI/bge-reranker-v2-m3)
    reranker = None
    if settings.reranker_type == "cross-encoder":
        try:
            reranker = ManualCrossEncoderReranker(
                model_name=settings.cross_encoder_model,
                top_n=settings.retrieval_k_final,
                score_threshold=settings.reranker_score_threshold,
            )
        except Exception as e:
            logger.warning("reranker_load_failed", error=str(e))

    # Hybrid retriever с RRF
    hybrid = HybridRetrieverWithRerank(
        bm25=bm25,
        qdrant=qdrant_retriever,
        reranker=reranker,
        k_initial=settings.retrieval_k_initial,
        k_final=settings.retrieval_k_final,
        rrf_k=settings.rrf_k,
    )
    logger.info(
        "hybrid_retriever_built",
        fusion="RRF",
        rrf_k=settings.rrf_k,
        reranker=settings.cross_encoder_model if reranker else "none",
    )

    # Metadata filter
    meta_retriever = MetadataAwareRetriever(hybrid, qdrant_retriever, documents)

    return RetrieverWrapper(inner=meta_retriever)
