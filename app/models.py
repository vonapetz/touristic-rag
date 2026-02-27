"""
Pydantic модели для API запросов и ответов.
"""

from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=1000, description="Вопрос пользователя")
    show_sources: bool = Field(True, description="Включать источники в ответ")


class SourceInfo(BaseModel):
    name: str
    city: str
    lat: Optional[float] = None
    lon: Optional[float] = None
    relevance_rank: int
    score: Optional[float] = None           # score от cross-encoder реранкера
    text_snippet: Optional[str] = None      # первые 200 символов документа
    image: Optional[str] = None             # полный URL изображения
    wikidata: Optional[str] = None          # WikiData ID


class ModelInfo(BaseModel):
    llm_provider: str
    llm_model: str
    embedding_model: str
    reranker: str


class TimingInfo(BaseModel):
    total_sec: float
    retrieval_sec: float
    reranking_sec: float    # теперь реально измеряется
    llm_sec: float


class QueryResponse(BaseModel):
    query_id: str           # полный UUID v4
    question: str
    answer: str
    sources: List[SourceInfo] = []
    timing: TimingInfo
    model_info: ModelInfo
    timestamp: datetime
    cached: bool = False    # True если ответ из кеша


class StreamToken(BaseModel):
    """Токен для SSE-стриминга."""
    token: str
    query_id: str
    done: bool = False
    sources: List[SourceInfo] = []   # заполняется только при done=True


class FeedbackRequest(BaseModel):
    query_id: str
    rating: int = Field(..., ge=1, le=5, description="Оценка 1-5")
    comment: Optional[str] = Field(None, max_length=500)


class FeedbackResponse(BaseModel):
    status: str
    query_id: str


class HealthResponse(BaseModel):
    status: str             # "healthy" | "degraded"
    qdrant: str             # "ok" | "unreachable"
    documents: int
    llm_provider: str
    llm_model: str
    embedding_model: str
    reranker: str
    uptime_sec: float
    cache_size: int
