"""Offline registry behavior and the real LangChain composition boundary."""

import pytest
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.messages import AIMessage
from langchain_core.prompts import ChatPromptTemplate

from finrag.config import ChatModelSettings
from finrag.providers import (
    ChatProviderFactory,
    ChatProviderRegistry,
    DuplicateProviderError,
    InvalidProviderNameError,
    ProviderNotRegisteredError,
)


def make_test_chat_model(settings: ChatModelSettings) -> BaseChatModel:
    """Local test factory; no credentials, SDK, or environment access."""
    return FakeListChatModel(responses=[f"Offline response from {settings.model}"])


def test_registration_and_lookup_do_not_construct_models() -> None:
    constructed_with: list[ChatModelSettings] = []

    def factory(settings: ChatModelSettings) -> BaseChatModel:
        constructed_with.append(settings)
        return make_test_chat_model(settings)

    registry = ChatProviderRegistry()
    registry.register("local", factory)
    resolved: ChatProviderFactory = registry.get("local")
    assert resolved is factory
    assert constructed_with == []

    settings = ChatModelSettings(provider="local", model="test-model", temperature=0.4)
    model = resolved(settings)
    assert constructed_with == [settings]
    assert constructed_with[0] is settings
    assert isinstance(model, BaseChatModel)


def test_provider_names_are_trimmed_and_lowercased() -> None:
    registry = ChatProviderRegistry()
    registry.register("  LoCaL\t", make_test_chat_model)
    assert registry.get("local") is make_test_chat_model
    assert registry.get(" LOCAL\n") is make_test_chat_model


@pytest.mark.parametrize("name", ["", " \t\n"])
def test_blank_names_are_rejected_for_registration_and_lookup(name: str) -> None:
    registry = ChatProviderRegistry()
    with pytest.raises(InvalidProviderNameError, match="must not be blank"):
        registry.register(name, make_test_chat_model)
    with pytest.raises(InvalidProviderNameError, match="must not be blank"):
        registry.get(name)


def test_missing_provider_requires_explicit_registration() -> None:
    registry = ChatProviderRegistry()
    registry.register("local", make_test_chat_model)
    with pytest.raises(ProviderNotRegisteredError, match="'missing'.*register"):
        registry.get(" MISSING ")


def test_duplicate_canonical_name_does_not_replace_original_factory() -> None:
    def replacement(settings: ChatModelSettings) -> BaseChatModel:
        raise AssertionError("Replacement factory must not be selected")

    registry = ChatProviderRegistry()
    registry.register("local", make_test_chat_model)
    with pytest.raises(DuplicateProviderError, match="'local'.*already registered"):
        registry.register(" LOCAL ", replacement)
    assert registry.get("local") is make_test_chat_model


def test_registry_instances_own_independent_registrations() -> None:
    def other_factory(settings: ChatModelSettings) -> BaseChatModel:
        return FakeListChatModel(responses=["Other instance"])

    registry_a = ChatProviderRegistry()
    registry_b = ChatProviderRegistry()
    registry_a.register("local", make_test_chat_model)
    with pytest.raises(ProviderNotRegisteredError):
        registry_b.get("local")
    registry_b.register("local", other_factory)
    assert registry_a.get("local") is make_test_chat_model
    assert registry_b.get("local") is other_factory


def test_resolved_model_preserves_runnable_composition_and_messages() -> None:
    settings = ChatModelSettings(provider="LOCAL", model="offline-chat")
    registry = ChatProviderRegistry()
    registry.register("local", make_test_chat_model)
    model: BaseChatModel = registry.get(settings.provider)(settings)
    prompt = ChatPromptTemplate.from_messages([("human", "{question}")])

    response = (prompt | model).invoke({"question": "Summarize the report"})

    assert isinstance(response, AIMessage)
    assert response.content == "Offline response from offline-chat"
    assert model.invoke("Rewrite the question").content == response.content
    assert settings.provider == "LOCAL"


def test_factory_failure_propagates_without_substitution() -> None:
    failure = RuntimeError("Test-only construction failure")

    def failing_factory(settings: ChatModelSettings) -> BaseChatModel:
        raise failure

    registry = ChatProviderRegistry()
    registry.register("local", failing_factory)
    with pytest.raises(RuntimeError) as caught:
        registry.get("local")(ChatModelSettings(provider="local"))
    assert caught.value is failure
