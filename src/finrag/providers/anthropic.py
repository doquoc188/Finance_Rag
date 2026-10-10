"""Anthropic chat construction with call-local credentials and integration import."""

import os

from langchain_core.language_models.chat_models import BaseChatModel
from pydantic import SecretStr

from finrag.config import ChatModelSettings

from .errors import (
    ProviderConfigurationError,
    ProviderDependencyError,
    ProviderInitializationError,
)


def create_anthropic_chat_model(config: ChatModelSettings) -> BaseChatModel:
    """Construct Claude's chat interface without requests or registration."""
    key = os.environ.get("ANTHROPIC_API_KEY")
    if key is None or not key.strip():
        raise ProviderConfigurationError("Anthropic requires a nonblank ANTHROPIC_API_KEY")

    try:
        from langchain_anthropic import ChatAnthropic
    except ImportError as exc:
        raise ProviderDependencyError(
            "Anthropic requires langchain-anthropic; run uv sync --extra anthropic"
        ) from exc
    except Exception as exc:
        raise ProviderInitializationError("Anthropic integration could not be loaded") from exc

    try:
        # Upstream Field(None, alias=...) makes mypy require timeout/stop incorrectly.
        # Both default to None in 1.7.5; absent optional SDK also makes this ignore unused.
        return ChatAnthropic(  # type: ignore[call-arg, unused-ignore]
            model_name=config.model, temperature=config.temperature, api_key=SecretStr(key)
        )
    except Exception as exc:
        raise ProviderInitializationError(
            "Anthropic model initialization failed; check model settings and ANTHROPIC_API_KEY"
        ) from exc
