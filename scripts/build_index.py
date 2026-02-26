#!/usr/bin/env python3
"""Пересборка Qdrant коллекции из командной строки."""

import sys
sys.path.insert(0, ".")

from qdrant_client import QdrantClient
from app.config import settings
from app.logger import setup_logging, get_logger
from app.data_processor import load_and_clean_data, create_documents
from app.embeddings import E5EmbeddingsWrapper
from app.retriever import _init_qdrant_collection

if __name__ == "__main__":
    setup_logging()
    logger = get_logger("build_index")

    # Удаляем старую коллекцию
    client = QdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key)

    collections = [c.name for c in client.get_collections().collections]
    if settings.qdrant_collection in collections:
        client.delete_collection(settings.qdrant_collection)
        logger.info("old_collection_deleted", collection=settings.qdrant_collection)

    # Строим новую
    df = load_and_clean_data()
    documents = create_documents(df)
    embeddings = E5EmbeddingsWrapper()

    _init_qdrant_collection(client, settings.qdrant_collection, documents, embeddings)

    info = client.get_collection(settings.qdrant_collection)
    print(f"Коллекция '{settings.qdrant_collection}': {info.points_count} документов")