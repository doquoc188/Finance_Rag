"""OpenAI chat construction; integration imports and credentials are call-local."""

import os

from langchain_core.language_models.chat_models import BaseChatModel
from pydantic import SecretStr

from finrag.config import ChatModelSettings

from .errors import (
    ProviderConfigurationError,
    ProviderDependencyError,
    ProviderInitializationError,
)


def create_openai_chat_model(config: ChatModelSettings) -> BaseChatModel:
    """Construct a chat model without requests, registration, or fallback."""
    key = os.environ.get("OPENAI_API_KEY")
    if key is None or not key.strip():
        raise ProviderConfigurationError("OpenAI requires a nonblank OPENAI_API_KEY")

    try:
        from langchain_openai import ChatOpenAI
    except ImportError as exc:
        raise ProviderDependencyError(
            "OpenAI requires a working langchain-openai installation; run uv sync"
        ) from exc
    except Exception as exc:
        raise ProviderInitializationError("OpenAI integration could not be loaded") from exc

    try:
        return ChatOpenAI(
            model=config.model, temperature=config.temperature, api_key=SecretStr(key)
        )
    except Exception as exc:
        raise ProviderInitializationError(
            "OpenAI model initialization failed; check model settings and OPENAI_API_KEY"
        ) from exc
