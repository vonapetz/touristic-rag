"""
Абстракция над LLM.
Поддерживает OpenAI API и локальную Qwen3 через Ollama.
Оба варианта предоставляют одинаковый интерфейс ChatOpenAI.
"""

from langchain_openai import ChatOpenAI
from app.config import settings
from app.logger import get_logger

logger = get_logger("llm")


def create_llm() -> ChatOpenAI:
    """
    Создаёт LLM клиент в зависимости от конфигурации.
    Ollama предоставляет OpenAI-совместимый API,
    поэтому используем ChatOpenAI для обоих случаев.
    """

    if settings.llm_provider == "openai":
        if not settings.openai_api_key:
            raise ValueError("OPENAI_API_KEY не задан при LLM_PROVIDER=openai")

        logger.info("llm_init", provider="openai", model=settings.openai_model)

        return ChatOpenAI(
            model=settings.openai_model,
            temperature=settings.llm_temperature,
            max_tokens=settings.llm_max_tokens,
            openai_api_key=settings.openai_api_key,
        )

    elif settings.llm_provider == "local":
        logger.info(
            "llm_init",
            provider="local",
            model=settings.local_llm_model,
            url=settings.local_llm_url,
        )

        return ChatOpenAI(
            model=settings.local_llm_model,
            temperature=settings.llm_temperature,
            max_tokens=settings.llm_max_tokens,
            openai_api_key="not-needed",  # Ollama не требует ключ
            base_url=settings.local_llm_url,
            request_timeout=120,  # Ollama с большим контекстом может отвечать долго
        )

    else:
        raise ValueError(f"Неизвестный LLM_PROVIDER: {settings.llm_provider}")


def get_model_name() -> str:
    """Возвращает имя текущей модели для логирования."""
    if settings.llm_provider == "openai":
        return settings.openai_model
    return settings.local_llm_model
    