"""Offline OpenAI/Groq factory checks with explicit integration replacements."""

import builtins
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import cast

import pytest
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.messages import AIMessage
from langchain_core.prompts import ChatPromptTemplate
from pydantic import SecretStr

from finrag.config import ChatModelSettings
from finrag.providers import (
    ChatProviderFactory,
    ChatProviderRegistry,
    ProviderConfigurationError,
    ProviderDependencyError,
    ProviderInitializationError,
    ProviderNotRegisteredError,
)
from finrag.providers.groq import create_groq_chat_model
from finrag.providers.openai import create_openai_chat_model


@dataclass(frozen=True)
class AdapterCase:
    name: str
    factory: ChatProviderFactory
    integration: str
    constructor: str
    key_variable: str


@pytest.fixture(
    params=(
        AdapterCase(
            "openai", create_openai_chat_model, "langchain_openai", "ChatOpenAI", "OPENAI_API_KEY"
        ),
        AdapterCase("groq", create_groq_chat_model, "langchain_groq", "ChatGroq", "GROQ_API_KEY"),
    ),
    ids=("openai", "groq"),
)
def adapter(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> AdapterCase:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    return cast(AdapterCase, request.param)


def install_constructor(
    monkeypatch: pytest.MonkeyPatch,
    adapter: AdapterCase,
    constructor: Callable[..., BaseChatModel],
) -> None:
    integration = ModuleType(adapter.integration)
    setattr(integration, adapter.constructor, constructor)
    monkeypatch.setitem(sys.modules, adapter.integration, integration)


def test_factory_forwards_settings_and_preserves_registry_and_langchain_contract(
    adapter: AdapterCase, monkeypatch: pytest.MonkeyPatch
) -> None:
    key = "test-only-credential"
    monkeypatch.setenv(adapter.key_variable, key)
    calls: list[tuple[str, float, SecretStr]] = []
    expected = FakeListChatModel(responses=["Offline adapter response"])

    def constructor(*, model: str, temperature: float, api_key: SecretStr) -> BaseChatModel:
        calls.append((model, temperature, api_key))
        return expected

    install_constructor(monkeypatch, adapter, constructor)
    settings = ChatModelSettings(provider=adapter.name, model="explicit-model", temperature=0.25)
    registry = ChatProviderRegistry()
    registry.register(adapter.name, adapter.factory)
    factory = registry.get(settings.provider)
    assert calls == []
    model = factory(settings)
    assert model is expected
    assert calls == [("explicit-model", 0.25, SecretStr(key))]
    assert calls[0][2].get_secret_value() == key
    assert key not in repr(calls)

    response = (ChatPromptTemplate.from_messages([("human", "{question}")]) | model).invoke(
        {"question": "Summarize the report"}
    )
    assert isinstance(response, AIMessage)
    assert response.content == "Offline adapter response"
    with pytest.raises(ProviderNotRegisteredError):
        ChatProviderRegistry().get(adapter.name)


@pytest.mark.parametrize("key", [None, "", " \t\n"])
def test_missing_or_blank_key_fails_before_import_and_does_not_load_dotenv(
    adapter: AdapterCase, key: str | None, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".env").write_text(
        f"{adapter.key_variable}=unused-dotenv-value\n", encoding="utf-8"
    )
    if key is not None:
        monkeypatch.setenv(adapter.key_variable, key)
    # A key for another provider must not supply this provider's credential.
    other_key = "GROQ_API_KEY" if adapter.name == "openai" else "OPENAI_API_KEY"
    monkeypatch.setenv(other_key, "other-provider-test-key")
    monkeypatch.setitem(sys.modules, adapter.integration, None)
    with pytest.raises(ProviderConfigurationError, match=adapter.key_variable) as caught:
        adapter.factory(ChatModelSettings(provider=adapter.name))
    assert caught.value.__cause__ is None
    assert "unused-dotenv-value" not in str(caught.value)
    assert "other-provider-test-key" not in str(caught.value)


def test_missing_integration_is_actionable_and_chained(
    adapter: AdapterCase, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(adapter.key_variable, "test-only-credential")
    monkeypatch.setitem(sys.modules, adapter.integration, None)
    with pytest.raises(ProviderDependencyError) as caught:
        adapter.factory(ChatModelSettings(provider=adapter.name))
    assert isinstance(caught.value.__cause__, ModuleNotFoundError)
    assert adapter.integration.replace("_", "-") in str(caught.value)
    assert "uv sync" in str(caught.value)
    if adapter.name == "groq":
        assert "--extra groq" in str(caught.value)
    assert "test-only-credential" not in repr(caught.value)


@pytest.mark.parametrize(
    ("failure_type", "expected_error"),
    [(ImportError, ProviderDependencyError), (RuntimeError, ProviderInitializationError)],
)
def test_integration_load_failure_preserves_cause_without_public_secret_leakage(
    adapter: AdapterCase,
    monkeypatch: pytest.MonkeyPatch,
    failure_type: type[Exception],
    expected_error: type[Exception],
) -> None:
    key = "test-only-sensitive-value"
    monkeypatch.setenv(adapter.key_variable, key)
    failure = failure_type(f"Integration failed with credential {key}")
    original_import = builtins.__import__

    def import_with_failure(
        name: str,
        globals: Mapping[str, object] | None = None,
        locals: Mapping[str, object] | None = None,
        fromlist: Sequence[str] = (),
        level: int = 0,
    ) -> ModuleType:
        if name == adapter.integration:
            raise failure
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", import_with_failure)
    with pytest.raises(expected_error) as caught:
        adapter.factory(ChatModelSettings(provider=adapter.name))
    assert caught.value.__cause__ is failure
    assert key not in str(caught.value)
    assert key not in repr(caught.value)


def test_constructor_failure_preserves_cause_without_public_secret_leakage(
    adapter: AdapterCase, monkeypatch: pytest.MonkeyPatch
) -> None:
    key = "test-only-sensitive-value"
    monkeypatch.setenv(adapter.key_variable, key)
    failure = ValueError(f"SDK rejected credential {key}")

    def constructor(*, model: str, temperature: float, api_key: SecretStr) -> BaseChatModel:
        raise failure

    install_constructor(monkeypatch, adapter, constructor)
    with pytest.raises(ProviderInitializationError) as caught:
        adapter.factory(ChatModelSettings(provider=adapter.name))
    assert caught.value.__cause__ is failure
    assert key not in str(caught.value)
    assert key not in repr(caught.value)
