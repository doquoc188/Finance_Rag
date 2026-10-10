"""Groq chat construction with a lazy optional integration dependency."""

import os

from langchain_core.language_models.chat_models import BaseChatModel
from pydantic import SecretStr

from finrag.config import ChatModelSettings

from .errors import (
    ProviderConfigurationError,
    ProviderDependencyError,
    ProviderInitializationError,
)


def create_groq_chat_model(config: ChatModelSettings) -> BaseChatModel:
    """Construct a chat model only when explicitly called; never select a substitute."""
    key = os.environ.get("GROQ_API_KEY")
    if key is None or not key.strip():
        raise ProviderConfigurationError("Groq requires a nonblank GROQ_API_KEY")

    try:
        from langchain_groq import ChatGroq
    except ImportError as exc:
        raise ProviderDependencyError(
            "Groq requires the optional langchain-groq integration; run uv sync --extra groq"
        ) from exc
    except Exception as exc:
        raise ProviderInitializationError("Groq integration could not be loaded") from exc

    try:
        return ChatGroq(model=config.model, temperature=config.temperature, api_key=SecretStr(key))
    except Exception as exc:
        raise ProviderInitializationError(
            "Groq model initialization failed; check model settings and GROQ_API_KEY"
        ) from exc
