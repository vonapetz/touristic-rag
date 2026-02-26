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


class ModelInfo(BaseModel):
    llm_provider: str
    llm_model: str
    embedding_model: str
    reranker: str


class TimingInfo(BaseModel):
    total_sec: float
    retrieval_sec: float
    reranking_sec: float
    llm_sec: float


class QueryResponse(BaseModel):
    query_id: str
    question: str
    answer: str
    sources: List[SourceInfo] = []
    timing: TimingInfo
    model_info: ModelInfo
    timestamp: str


class FeedbackRequest(BaseModel):
    query_id: str
    rating: int = Field(..., ge=1, le=5, description="Оценка 1-5")
    comment: Optional[str] = Field(None, max_length=500)


class FeedbackResponse(BaseModel):
    status: str
    query_id: str


class HealthResponse(BaseModel):
    status: str
    documents: int
    llm_provider: str
    llm_model: str
    embedding_model: str
    reranker: str
    uptime_sec: float