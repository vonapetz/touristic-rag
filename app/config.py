"""
Конфигурация приложения.
Все параметры читаются из .env файла.
Pydantic валидирует типы при старте.
"""

from pydantic_settings import BaseSettings
from pydantic import Field
from typing import Optional


class Settings(BaseSettings):
    # LLM
    llm_provider: str = Field("local", description="'openai' или 'local'")
    openai_api_key: Optional[str] = None
    openai_model: str = "gpt-3.5-turbo"
    local_llm_url: str = "http://localhost:11434/v1"
    local_llm_model: str = "qwen3:14b"
    llm_temperature: float = 0.3
    llm_max_tokens: int = 1024

    # Embeddings
    embedding_model: str = "intfloat/multilingual-e5-large"
    embedding_device: str = "cpu"

    # Retrieval
    retrieval_k_initial: int = 10
    retrieval_k_final: int = 3
    bm25_weight: float = 0.4
    vector_weight: float = 0.6

    # Qdrant
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: Optional[str] = None
    qdrant_collection: str = "landmarks"

    # Reranker
    reranker_type: str = Field("cross-encoder", description="'cohere', 'cross-encoder', 'none'")
    cross_encoder_model: str = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
    cohere_api_key: Optional[str] = None

    # Data
    data_url: str = "https://drive.google.com/uc?id=1P1BsvI2jPN3fEqjc2YZxmQ-MTs22WVUk"
    data_path: str = "data/raw/file.csv"
    processed_data_path: str = "data/processed/final_clean_data.csv"

    # Server
    host: str = "0.0.0.0"
    port: int = 8080
    log_level: str = "INFO"
    log_format: str = "json"

    # Cleanup
    tfidf_min_df: int = 10
    tfidf_max_df: float = 0.1
    tfidf_suspicious_low: float = 0.001
    tfidf_suspicious_high: float = 50
    min_description_length: int = 10

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


settings = Settings()
