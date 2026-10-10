"""Caller-owned built-in registration without invoking concrete model factories."""

import pytest
from langchain_core.language_models.chat_models import BaseChatModel

from finrag.config import ChatModelSettings
from finrag.providers import (
    ChatProviderFactory,
    ChatProviderRegistry,
    DuplicateProviderError,
    ProviderNotRegisteredError,
)
from finrag.providers.anthropic import create_anthropic_chat_model
from finrag.providers.builtin import register_builtin_chat_providers
from finrag.providers.google import create_google_chat_model
from finrag.providers.groq import create_groq_chat_model
from finrag.providers.ollama import create_ollama_chat_model
from finrag.providers.openai import create_openai_chat_model


def test_builtin_helper_registers_exact_factories_only_on_supplied_registry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry = ChatProviderRegistry()
    other_registry = ChatProviderRegistry()
    registrations: list[tuple[str, ChatProviderFactory]] = []
    original_register = registry.register

    def record_registration(name: str, factory: ChatProviderFactory) -> None:
        registrations.append((name, factory))
        original_register(name, factory)

    monkeypatch.setattr(registry, "register", record_registration)
    register_builtin_chat_providers(registry)
    assert registrations == [
        ("openai", create_openai_chat_model),
        ("groq", create_groq_chat_model),
        ("anthropic", create_anthropic_chat_model),
        ("google", create_google_chat_model),
        ("ollama", create_ollama_chat_model),
    ]
    for name, factory in registrations:
        assert registry.get(name) is factory
        with pytest.raises(ProviderNotRegisteredError):
            other_registry.get(name)


def test_duplicate_helper_call_fails_without_overwriting_factories() -> None:
    registry = ChatProviderRegistry()
    register_builtin_chat_providers(registry)
    factories = {
        name: registry.get(name) for name in ("openai", "groq", "anthropic", "google", "ollama")
    }
    with pytest.raises(DuplicateProviderError, match="openai"):
        register_builtin_chat_providers(registry)
    for name, factory in factories.items():
        assert registry.get(name) is factory


def test_registration_conflict_preserves_existing_factory_and_prior_registrations() -> None:
    def custom_factory(config: ChatModelSettings) -> BaseChatModel:
        raise AssertionError("Registration must not construct models")

    registry = ChatProviderRegistry()
    registry.register("google", custom_factory)
    with pytest.raises(DuplicateProviderError, match="google"):
        register_builtin_chat_providers(registry)
    assert registry.get("google") is custom_factory
    assert registry.get("openai") is create_openai_chat_model
    assert registry.get("groq") is create_groq_chat_model
    assert registry.get("anthropic") is create_anthropic_chat_model
    with pytest.raises(ProviderNotRegisteredError):
        registry.get("ollama")
