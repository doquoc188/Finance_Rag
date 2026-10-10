"""Gemini Developer API chat construction with a lazy optional integration."""

import os

from langchain_core.language_models.chat_models import BaseChatModel
from pydantic import SecretStr

from finrag.config import ChatModelSettings

from .errors import (
    ProviderConfigurationError,
    ProviderDependencyError,
    ProviderInitializationError,
)


def create_google_chat_model(config: ChatModelSettings) -> BaseChatModel:
    """Construct the API-key Gemini backend; no ambient Vertex AI selection."""
    key = os.environ.get("GOOGLE_API_KEY")
    if key is None or not key.strip():
        raise ProviderConfigurationError("Google requires a nonblank GOOGLE_API_KEY")

    try:
        from langchain_google_genai import ChatGoogleGenerativeAI
    except ImportError as exc:
        raise ProviderDependencyError(
            "Google requires langchain-google-genai; run uv sync --extra google"
        ) from exc
    except Exception as exc:
        raise ProviderInitializationError("Google integration could not be loaded") from exc

    try:
        # Fix the intended API-key backend instead of consulting Vertex/ADC environment.
        return ChatGoogleGenerativeAI(
            model=config.model,
            temperature=config.temperature,
            api_key=SecretStr(key),
            vertexai=False,
        )
    except Exception as exc:
        raise ProviderInitializationError(
            "Google model initialization failed; check model settings and GOOGLE_API_KEY"
        ) from exc
