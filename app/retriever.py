"""
Retrieval pipeline:
1. Qdrant Vector Search (с нативной фильтрацией по метаданным)
2. BM25 для keyword search
3. Hybrid Ensemble (BM25 + Qdrant)
4. Reranking (Cross-Encoder через sentence-transformers)
5. Metadata-aware filtering (город из запроса)
"""

from typing import List, Optional, Any

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
# Cross-Encoder Reranker (ручная реализация через sentence-transformers)
# ============================================================================

class ManualCrossEncoderReranker:
    """Reranker через sentence-transformers CrossEncoder."""

    def __init__(self, model_name: str, top_n: int):
        from sentence_transformers import CrossEncoder
        logger.info("loading_cross_encoder", model=model_name)
        self.model = CrossEncoder(model_name)
        self.top_n = top_n
        logger.info("cross_encoder_loaded", model=model_name)

    def rerank(self, query: str, documents: List[Document]) -> List[Document]:
        if not documents:
            return []

        pairs = [[query, doc.page_content] for doc in documents]
        scores = self.model.predict(pairs)

        scored = list(zip(documents, scores))
        scored.sort(key=lambda x: x[1], reverse=True)

        return [doc for doc, _ in scored[:self.top_n]]


# ============================================================================
# Hybrid Retriever с reranking
# ============================================================================

class HybridRetrieverWithRerank:
    """
    BM25 + Qdrant → объединение → reranking.
    Заменяет EnsembleRetriever + ContextualCompressionRetriever.
    """

    def __init__(
        self,
        bm25: BM25Retriever,
        qdrant: QdrantRetriever,
        reranker: Optional[ManualCrossEncoderReranker],
        bm25_weight: float,
        vector_weight: float,
        k_initial: int,
        k_final: int,
    ):
        self.bm25 = bm25
        self.qdrant = qdrant
        self.reranker = reranker
        self.bm25_weight = bm25_weight
        self.vector_weight = vector_weight
        self.k_initial = k_initial
        self.k_final = k_final

    def invoke(self, query: str) -> List[Document]:
        # BM25 результаты
        bm25_results = self.bm25.invoke(query)

        # Qdrant результаты (без фильтра по городу — это делает MetadataAwareRetriever)
        qdrant_results = self.qdrant.invoke(query, k=self.k_initial)

        # Объединяем с дедупликацией
        seen_texts = set()
        merged = []

        # Qdrant результаты первыми (выше вес)
        for doc in qdrant_results:
            text_key = doc.page_content[:100]
            if text_key not in seen_texts:
                seen_texts.add(text_key)
                merged.append(doc)

        # Дополняем BM25
        for doc in bm25_results:
            text_key = doc.page_content[:100]
            if text_key not in seen_texts:
                seen_texts.add(text_key)
                merged.append(doc)

        # Reranking
        if self.reranker and len(merged) > self.k_final:
            merged = self.reranker.rerank(query, merged)
        else:
            merged = merged[:self.k_final]

        return merged


# ============================================================================
# Metadata-aware Retriever
# ============================================================================

class MetadataAwareRetriever:
    """Retriever с автоматической фильтрацией по городу из запроса."""

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

    def invoke(self, query: str) -> List[Document]:
        city = self._extract_city(query)

        if city:
            # Нативный Qdrant фильтр — pre-filtering
            city_results = self.qdrant.invoke(
                query, k=settings.retrieval_k_final, city_filter=city
            )
            if len(city_results) >= 2:
                logger.debug("qdrant_native_filter", city=city, results=len(city_results))
                return city_results

        # Hybrid search (BM25 + Qdrant + reranking)
        results = self.hybrid.invoke(query)

        logger.debug(
            "metadata_filter",
            city_extracted=city,
            results_count=len(results),
        )

        return results


# ============================================================================
# LangChain Wrapper
# ============================================================================

class RetrieverWrapper(BaseRetriever):
    """Адаптер MetadataAwareRetriever → LangChain BaseRetriever."""

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
        return self._inner.invoke(query)


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

            # Проверяем/создаём индексы на существующей коллекции
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

    # Создаём payload индексы для фильтрации
    _ensure_payload_indexes(client, collection)


def _ensure_payload_indexes(client: QdrantClient, collection: str):
    """Создаёт payload индексы для полей, по которым будем фильтровать."""

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
            # Индекс уже существует — это нормально
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

    # BM25
    bm25 = BM25Retriever.from_documents(documents, k=settings.retrieval_k_initial)

    # Reranker
    reranker = None
    if settings.reranker_type == "cross-encoder":
        try:
            reranker = ManualCrossEncoderReranker(
                model_name=settings.cross_encoder_model,
                top_n=settings.retrieval_k_final,
            )
        except Exception as e:
            logger.warning("reranker_load_failed", error=str(e))

    # Hybrid retriever
    hybrid = HybridRetrieverWithRerank(
        bm25=bm25,
        qdrant=qdrant_retriever,
        reranker=reranker,
        bm25_weight=settings.bm25_weight,
        vector_weight=settings.vector_weight,
        k_initial=settings.retrieval_k_initial,
        k_final=settings.retrieval_k_final,
    )
    logger.info("hybrid_retriever_built", bm25_w=settings.bm25_weight, vec_w=settings.vector_weight)

    # Metadata filter
    meta_retriever = MetadataAwareRetriever(hybrid, qdrant_retriever, documents)

    return RetrieverWrapper(inner=meta_retriever)