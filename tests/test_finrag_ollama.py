"""Ollama adapter contracts without a server, credentials, or model requests."""

import sys
from pathlib import Path
from types import ModuleType

import pytest
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.messages import AIMessage
from langchain_core.prompts import ChatPromptTemplate

from finrag.config import ChatModelSettings
from finrag.providers import (
    ChatProviderRegistry,
    ProviderConfigurationError,
    ProviderDependencyError,
    ProviderInitializationError,
)
from finrag.providers.ollama import create_ollama_chat_model


@pytest.fixture(autouse=True)
def isolate_provider_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in (
        "OPENAI_API_KEY",
        "GROQ_API_KEY",
        "ANTHROPIC_API_KEY",
        "GOOGLE_API_KEY",
        "GEMINI_API_KEY",
        "OLLAMA_API_KEY",
        "OLLAMA_BASE_URL",
        "OLLAMA_HOST",
    ):
        monkeypatch.delenv(name, raising=False)


@pytest.mark.parametrize("base_url", [None, "http://127.0.0.1:23456"])
def test_keyless_construction_forwards_default_or_explicit_url_and_composes(
    base_url: str | None, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".env").write_text("OLLAMA_BASE_URL=http://unused.example\n", encoding="utf-8")
    # The adapter's deterministic default must not inherit another Ollama host setting.
    monkeypatch.setenv("OLLAMA_HOST", "http://ignored.example:1234")
    if base_url is not None:
        monkeypatch.setenv("OLLAMA_BASE_URL", base_url)
    calls: list[dict[str, object]] = []
    expected = FakeListChatModel(responses=["Offline Ollama response"])

    def constructor(**kwargs: object) -> BaseChatModel:
        calls.append(kwargs)
        return expected

    integration = ModuleType("langchain_ollama")
    setattr(integration, "ChatOllama", constructor)
    monkeypatch.setitem(sys.modules, "langchain_ollama", integration)
    registry = ChatProviderRegistry()
    registry.register("ollama", create_ollama_chat_model)
    assert calls == []
    settings = ChatModelSettings(provider="ollama", model="test-local-model", temperature=0.3)
    model = registry.get(settings.provider)(settings)
    assert model is expected
    assert calls == [
        {
            "model": "test-local-model",
            "temperature": 0.3,
            "base_url": base_url or "http://127.0.0.1:11434",
        }
    ]
    response = (ChatPromptTemplate.from_messages([("human", "{question}")]) | model).invoke(
        {"question": "Summarize the report"}
    )
    assert isinstance(response, AIMessage)
    assert response.content == "Offline Ollama response"


@pytest.mark.parametrize("base_url", ["", " \t\n"])
def test_blank_explicit_endpoint_fails_before_import(
    base_url: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("OLLAMA_BASE_URL", base_url)
    monkeypatch.setitem(sys.modules, "langchain_ollama", None)
    with pytest.raises(ProviderConfigurationError, match="OLLAMA_BASE_URL"):
        create_ollama_chat_model(ChatModelSettings(provider="ollama"))


def test_missing_optional_integration_is_actionable_and_chained(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(sys.modules, "langchain_ollama", None)
    with pytest.raises(ProviderDependencyError, match="langchain-ollama.*--extra ollama") as caught:
        create_ollama_chat_model(ChatModelSettings(provider="ollama"))
    assert isinstance(caught.value.__cause__, ModuleNotFoundError)


@pytest.mark.parametrize(
    ("failure_type", "expected_error"),
    [(ImportError, ProviderDependencyError), (RuntimeError, ProviderInitializationError)],
)
def test_integration_load_errors_preserve_cause_and_safe_public_text(
    failure_type: type[Exception], expected_error: type[Exception], monkeypatch: pytest.MonkeyPatch
) -> None:
    failure = failure_type("Upstream secret-bearing integration error")
    integration = ModuleType("langchain_ollama")

    def failed_attribute(name: str) -> object:
        raise failure

    setattr(integration, "__getattr__", failed_attribute)
    monkeypatch.setitem(sys.modules, "langchain_ollama", integration)
    with pytest.raises(expected_error) as caught:
        create_ollama_chat_model(ChatModelSettings(provider="ollama"))
    assert caught.value.__cause__ is failure
    assert str(failure) not in str(caught.value)


def test_constructor_failure_is_chained_without_exposing_endpoint_credentials(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    endpoint = "http://test-user:test-only-secret@127.0.0.1:23456"
    monkeypatch.setenv("OLLAMA_BASE_URL", endpoint)
    failure = ValueError(f"SDK rejected {endpoint}")

    def constructor(**kwargs: object) -> BaseChatModel:
        raise failure

    integration = ModuleType("langchain_ollama")
    setattr(integration, "ChatOllama", constructor)
    monkeypatch.setitem(sys.modules, "langchain_ollama", integration)
    with pytest.raises(ProviderInitializationError) as caught:
        create_ollama_chat_model(ChatModelSettings(provider="ollama"))
    assert caught.value.__cause__ is failure
    assert "test-only-secret" not in str(caught.value)
    assert endpoint not in repr(caught.value)
