"""Local Ollama chat construction without credentials or connectivity checks."""

import os

from langchain_core.language_models.chat_models import BaseChatModel

from finrag.config import ChatModelSettings

from .errors import (
    ProviderConfigurationError,
    ProviderDependencyError,
    ProviderInitializationError,
)


def create_ollama_chat_model(config: ChatModelSettings) -> BaseChatModel:
    """Create a local client; construction does not establish server/model availability."""
    base_url = os.environ.get("OLLAMA_BASE_URL", "http://127.0.0.1:11434").strip()
    if not base_url:
        raise ProviderConfigurationError("OLLAMA_BASE_URL must not be blank when supplied")

    try:
        from langchain_ollama import ChatOllama
    except ImportError as exc:
        raise ProviderDependencyError(
            "Ollama requires langchain-ollama; run uv sync --extra ollama"
        ) from exc
    except Exception as exc:
        raise ProviderInitializationError("Ollama integration could not be loaded") from exc

    try:
        return ChatOllama(model=config.model, temperature=config.temperature, base_url=base_url)
    except Exception as exc:
        raise ProviderInitializationError(
            "Ollama model initialization failed; check model settings and OLLAMA_BASE_URL"
        ) from exc
