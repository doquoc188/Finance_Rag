"""Explicit built-in registration on a caller-owned registry; no SDK construction."""

from .anthropic import create_anthropic_chat_model
from .google import create_google_chat_model
from .groq import create_groq_chat_model
from .ollama import create_ollama_chat_model
from .openai import create_openai_chat_model
from .registry import ChatProviderRegistry


def register_builtin_chat_providers(registry: ChatProviderRegistry) -> None:
    """Register all five factories in order, failing on the first duplicate.

    Calls are not idempotent or transactional: earlier registrations remain if a
    later name conflicts. Existing factories are never overwritten or invoked.
    """
    registry.register("openai", create_openai_chat_model)
    registry.register("groq", create_groq_chat_model)
    registry.register("anthropic", create_anthropic_chat_model)
    registry.register("google", create_google_chat_model)
    registry.register("ollama", create_ollama_chat_model)
