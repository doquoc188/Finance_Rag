"""Model setup failures must be explicit; SDK boundaries stay offline."""

import runpy
import sys
from types import ModuleType
from unittest.mock import Mock

import pytest

from app.core import llm
from app.core.errors import ModelConfigurationError, ProviderUnavailableError
from app.graph import nodes
from app.graph.workflow import finrag_app

PROVIDERS = [
    ("openai", "langchain_openai", "ChatOpenAI"),
    ("groq", "langchain_groq", "ChatGroq"),
    ("ollama", "langchain_community.chat_models", "ChatOllama"),
]


@pytest.fixture
def configured_model(monkeypatch: pytest.MonkeyPatch) -> None:
    for name, value in {
        "LLM_PROVIDER": "openai",
        "OPENAI_API_KEY": "unit-test-openai-credential",
        "GROQ_API_KEY": "unit-test-groq-credential",
        "OPENAI_MODEL": "test-openai-model",
        "GROQ_MODEL": "test-groq-model",
        "OLLAMA_MODEL": "test-ollama-model",
        "OLLAMA_BASE_URL": "http://localhost:11434",
    }.items():
        monkeypatch.setattr(llm.settings, name, value)


@pytest.mark.parametrize(
    ("provider", "setting", "value"),
    [
        ("openai", "OPENAI_API_KEY", ""),
        ("openai", "OPENAI_API_KEY", "  "),
        ("openai", "OPENAI_API_KEY", "mock-test-credential"),
        ("groq", "GROQ_API_KEY", ""),
        ("groq", "GROQ_API_KEY", "  "),
        ("openai", "OPENAI_MODEL", ""),
        ("groq", "GROQ_MODEL", "  "),
        ("ollama", "OLLAMA_MODEL", ""),
        ("ollama", "OLLAMA_BASE_URL", ""),
        ("", "LLM_PROVIDER", ""),
        ("unsupported", "LLM_PROVIDER", "unsupported"),
    ],
)
def test_missing_or_invalid_configuration(
    configured_model, monkeypatch: pytest.MonkeyPatch, provider: str, setting: str, value: str
) -> None:
    monkeypatch.setattr(llm.settings, "LLM_PROVIDER", provider)
    monkeypatch.setattr(llm.settings, setting, value)

    with pytest.raises(ModelConfigurationError, match=setting):
        llm.get_llm()


@pytest.mark.parametrize(("provider", "module_name", "class_name"), PROVIDERS)
def test_missing_provider_import_is_explicit(
    configured_model,
    monkeypatch: pytest.MonkeyPatch,
    provider: str,
    module_name: str,
    class_name: str,
) -> None:
    monkeypatch.setattr(llm.settings, "LLM_PROVIDER", provider)
    monkeypatch.setitem(sys.modules, module_name, None)

    with pytest.raises(ProviderUnavailableError, match="Package") as caught:
        llm.get_llm()

    assert provider in str(caught.value)
    assert isinstance(caught.value.__cause__, ImportError)


@pytest.mark.parametrize(("provider", "module_name", "class_name"), PROVIDERS)
def test_provider_import_runtime_failure_preserves_safe_context(
    configured_model,
    monkeypatch: pytest.MonkeyPatch,
    provider: str,
    module_name: str,
    class_name: str,
) -> None:
    monkeypatch.setattr(llm.settings, "LLM_PROVIDER", provider)
    cause = RuntimeError("unit-test-private-value")
    module = ModuleType(module_name)

    def broken_lazy_import(name: str) -> None:
        raise cause

    setattr(module, "__getattr__", broken_lazy_import)
    monkeypatch.setitem(sys.modules, module_name, module)

    with pytest.raises(ProviderUnavailableError, match="Failed to load provider") as caught:
        llm.get_llm()

    assert caught.value.__cause__ is cause
    assert "unit-test-private-value" not in str(caught.value)


@pytest.mark.parametrize(("provider", "module_name", "class_name"), PROVIDERS)
@pytest.mark.parametrize("failure_type", [ValueError, RuntimeError, ImportError])
def test_provider_initialization_failure_preserves_cause_without_exposing_secrets(
    configured_model,
    monkeypatch: pytest.MonkeyPatch,
    provider: str,
    module_name: str,
    class_name: str,
    failure_type: type[Exception],
) -> None:
    monkeypatch.setattr(llm.settings, "LLM_PROVIDER", provider)
    cause = failure_type("unit-test-private-value")
    module = ModuleType(module_name)
    constructor = Mock(side_effect=cause)
    setattr(module, class_name, constructor)
    monkeypatch.setitem(sys.modules, module_name, module)

    with pytest.raises(ProviderUnavailableError, match="Failed to initialize") as caught:
        llm.get_llm()

    constructor.assert_called_once()
    assert caught.value.__cause__ is cause
    assert "unit-test-private-value" not in str(caught.value)
    assert "unit-test-openai-credential" not in str(caught.value)
    assert "unit-test-groq-credential" not in str(caught.value)


@pytest.mark.parametrize(("provider", "module_name", "class_name"), PROVIDERS)
def test_valid_provider_preserves_constructor_arguments_and_return_value(
    configured_model,
    monkeypatch: pytest.MonkeyPatch,
    provider: str,
    module_name: str,
    class_name: str,
) -> None:
    # Uppercase selection was supported by the original factory and remains valid.
    monkeypatch.setattr(llm.settings, "LLM_PROVIDER", provider.upper())
    expected_model = object()
    constructor = Mock(return_value=expected_model)
    module = ModuleType(module_name)
    setattr(module, class_name, constructor)
    monkeypatch.setitem(sys.modules, module_name, module)

    result = llm.get_llm(temperature=0.25)

    options = {
        "openai": {"api_key": "unit-test-openai-credential"},
        "groq": {"groq_api_key": "unit-test-groq-credential"},
        "ollama": {"base_url": "http://localhost:11434"},
    }
    constructor.assert_called_once_with(
        model=f"test-{provider}-model", temperature=0.25, **options[provider]
    )
    assert result is expected_model


def test_workflow_without_explicit_test_model_propagates_configuration_error() -> None:
    with pytest.raises(ModelConfigurationError, match="OPENAI_API_KEY"):
        finrag_app.invoke(
            {"question": "Test question", "retry_count": 0, "web_search_needed": False}
        )


def test_missing_prompt_dependency_cannot_generate_a_canned_answer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(sys.modules, "langchain_core.prompts", None)

    with pytest.raises(ProviderUnavailableError, match="langchain-core") as caught:
        runpy.run_path(nodes.__file__)

    assert isinstance(caught.value.__cause__, ImportError)
