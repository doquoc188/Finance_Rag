"""Explicit multi-provider role composition with local chat models only."""

import os
from collections.abc import Iterator, Mapping
from dataclasses import FrozenInstanceError

import pytest
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.messages import AIMessage
from langchain_core.prompts import ChatPromptTemplate

from finrag.config import ChatModelSettings, Settings
from finrag.models import ModelRoles, resolve_model_roles
from finrag.providers import (
    ChatProviderFactory,
    ChatProviderRegistry,
    ProviderConfigurationError,
    ProviderDependencyError,
    ProviderInitializationError,
    ProviderNotRegisteredError,
)


@pytest.fixture(autouse=True)
def isolate_settings_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in list(os.environ):
        if key.upper().startswith("FINRAG_"):
            monkeypatch.delenv(key)


@pytest.fixture
def settings() -> Settings:
    return Settings(
        generation=ChatModelSettings(
            provider="provider_a", model="generation-model", temperature=0.1
        ),
        grading=ChatModelSettings(provider="provider_b", model="grading-model", temperature=0.2),
        query_rewrite=ChatModelSettings(
            provider="provider_c", model="rewrite-model", temperature=0.3
        ),
    )


def test_independent_roles_forward_original_settings_and_preserve_langchain_usage(
    settings: Settings,
) -> None:
    calls: list[tuple[str, ChatModelSettings]] = []

    def make_factory(name: str) -> ChatProviderFactory:
        def factory(config: ChatModelSettings) -> BaseChatModel:
            calls.append((name, config))
            return FakeListChatModel(responses=[f"{name}: {config.model}"])

        return factory

    registry = ChatProviderRegistry()
    for name in ("provider_a", "provider_b", "provider_c"):
        registry.register(name, make_factory(name))

    roles = resolve_model_roles(settings, registry)

    assert isinstance(roles, ModelRoles)
    assert calls == [
        ("provider_a", settings.generation),
        ("provider_b", settings.grading),
        ("provider_c", settings.query_rewrite),
    ]
    assert calls[0][1] is settings.generation
    assert calls[1][1] is settings.grading
    assert calls[2][1] is settings.query_rewrite
    assert len({id(config) for _, config in calls}) == 3
    assert len({id(roles.generation), id(roles.grading), id(roles.query_rewrite)}) == 3
    prompt = ChatPromptTemplate.from_messages([("human", "{question}")])
    for model, expected in (
        (roles.generation, "provider_a: generation-model"),
        (roles.grading, "provider_b: grading-model"),
        (roles.query_rewrite, "provider_c: rewrite-model"),
    ):
        assert isinstance(model, BaseChatModel)
        response = (prompt | model).invoke({"question": "Analyze the report"})
        assert isinstance(response, AIMessage)
        assert response.content == expected
    with pytest.raises(FrozenInstanceError):
        setattr(roles, "generation", roles.grading)


def test_resolver_constructs_each_role_per_call_even_with_identical_configuration() -> None:
    config = ChatModelSettings(provider="local", model="shared-configuration")
    settings = Settings(generation=config, grading=config, query_rewrite=config)
    models: list[BaseChatModel] = []

    def factory(config: ChatModelSettings) -> BaseChatModel:
        model = FakeListChatModel(responses=["Local response"])
        models.append(model)
        return model

    registry = ChatProviderRegistry()
    registry.register("local", factory)
    first = resolve_model_roles(settings, registry)
    second = resolve_model_roles(settings, registry)
    assert len(models) == 6
    assert len({id(model) for model in models}) == 6
    assert first.generation is models[0]
    assert first.grading is models[1]
    assert first.query_rewrite is models[2]
    assert second.generation is models[3]
    assert second.grading is models[4]
    assert second.query_rewrite is models[5]


@pytest.mark.parametrize("missing_provider", ["provider_a", "provider_b", "provider_c"])
def test_missing_provider_propagates_and_stops_resolution(
    missing_provider: str,
    settings: Settings,
) -> None:
    calls: list[str] = []

    def factory(config: ChatModelSettings) -> BaseChatModel:
        calls.append(config.provider)
        return FakeListChatModel(responses=["Local response"])

    names = ("provider_a", "provider_b", "provider_c")
    registry = ChatProviderRegistry()
    for name in names:
        if name != missing_provider:
            registry.register(name, factory)
    with pytest.raises(ProviderNotRegisteredError, match=missing_provider):
        resolve_model_roles(settings, registry)
    assert calls == list(names[: names.index(missing_provider)])


@pytest.mark.parametrize("failed_provider", ["provider_a", "provider_b", "provider_c"])
@pytest.mark.parametrize(
    "error_type",
    [ProviderConfigurationError, ProviderDependencyError, ProviderInitializationError],
)
def test_adapter_errors_propagate_unchanged_without_returning_partial_roles(
    failed_provider: str,
    error_type: type[Exception],
    settings: Settings,
) -> None:
    failure = error_type("Test-only provider failure")
    cause = ValueError("Test-only upstream cause")
    calls: list[str] = []

    def factory(config: ChatModelSettings) -> BaseChatModel:
        calls.append(config.provider)
        if config.provider == failed_provider:
            raise failure from cause
        return FakeListChatModel(responses=["Local response"])

    names = ("provider_a", "provider_b", "provider_c")
    registry = ChatProviderRegistry()
    for name in names:
        registry.register(name, factory)
    with pytest.raises(error_type) as caught:
        resolve_model_roles(settings, registry)
    assert caught.value is failure
    assert caught.value.__cause__ is cause
    assert calls == list(names[: names.index(failed_provider) + 1])


def test_resolver_does_not_read_environment(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    class UnreadableEnvironment(Mapping[str, str]):
        def __getitem__(self, key: str) -> str:
            raise AssertionError("Resolver must not read environment variables")

        def __iter__(self) -> Iterator[str]:
            raise AssertionError("Resolver must not iterate environment variables")

        def __len__(self) -> int:
            raise AssertionError("Resolver must not inspect environment variables")

    model = FakeListChatModel(responses=["Preconstructed local response"])

    def factory(config: ChatModelSettings) -> BaseChatModel:
        return model

    registry = ChatProviderRegistry()
    for name in ("provider_a", "provider_b", "provider_c"):
        registry.register(name, factory)
    with monkeypatch.context() as guard:
        guard.setattr(os, "environ", UnreadableEnvironment())
        roles = resolve_model_roles(settings, registry)
    assert roles.generation is model
    assert roles.grading is model
    assert roles.query_rewrite is model
