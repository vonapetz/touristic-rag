"""
Обёртка для E5 эмбеддингов.
Автоматически добавляет prefix 'passage:'/'query:' для intfloat/e5 моделей.
BM25 работает с чистым текстом — prefix не попадает в page_content.
"""

from typing import List
from langchain_huggingface import HuggingFaceEmbeddings
from app.config import settings
from app.logger import get_logger

logger = get_logger("embeddings")


class E5EmbeddingsWrapper:
    """
    Обёртка над HuggingFaceEmbeddings для e5 моделей.
    Документы получают prefix 'passage: ', запросы — 'query: '.
    """

    def __init__(self):
        logger.info(
            "loading_embeddings",
            model=settings.embedding_model,
            device=settings.embedding_device,
        )

        self._base = HuggingFaceEmbeddings(
            model_name=settings.embedding_model,
            model_kwargs={"device": settings.embedding_device},
            encode_kwargs={"normalize_embeddings": True, "batch_size": 32},
        )

        self._dim = len(self._base.embed_query("test"))
        logger.info("embeddings_loaded", dimension=self._dim)

    @property
    def dimension(self) -> int:
        return self._dim

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        prefixed = [f"passage: {t}" for t in texts]
        return self._base.embed_documents(prefixed)

    def embed_query(self, text: str) -> List[float]:
        if not text.startswith("query:"):
            text = f"query: {text}"
        return self._base.embed_query(text)

    # Async variants для FastAPI
    async def aembed_documents(self, texts: List[str]) -> List[List[float]]:
        return self.embed_documents(texts)

    async def aembed_query(self, text: str) -> List[float]:
        return self.embed_query(text)