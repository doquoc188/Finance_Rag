"""Real compiled graph and application composition, with explicit offline models."""

import socket
import sys
from types import ModuleType

import pytest
from fastapi.testclient import TestClient
from langgraph.graph.state import CompiledStateGraph

from app import runtime as composition
from app.core import llm
from app.graph import nodes
from app.graph.nodes import GradeDocuments
from app.graph.state import FinGraphUpdate
from app.graph.workflow import FallbackStateGraph, build_finrag_graph
from app.main import create_app
from app.runtime import GraphRuntime, compose_runtime
from finrag.config import ChatModelSettings, Settings
from finrag.models import ModelRoles
from finrag.providers import ChatProviderRegistry
from finrag.providers.errors import (
    ProviderConfigurationError,
    ProviderDependencyError,
    ProviderInitializationError,
    ProviderNotRegisteredError,
)
from tests.chat_models import ModelCall, WorkflowChatModel


@pytest.fixture(autouse=True)
def isolate_runtime(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(nodes, "global_retriever", None)

    def reject_legacy_model(*args: object, **kwargs: object) -> None:
        raise AssertionError("The graph must not construct a legacy chat model")

    monkeypatch.setattr(llm, "get_llm", reject_legacy_model)


def role_settings() -> Settings:
    return Settings(
        generation=ChatModelSettings(provider="generate-test", model="generation", temperature=0.1),
        grading=ChatModelSettings(provider="grade-test", model="grading"),
        query_rewrite=ChatModelSettings(
            provider="rewrite-test", model="query_rewrite", temperature=0.2
        ),
    )


def make_test_registry(
    records: list[tuple[ChatModelSettings, WorkflowChatModel]],
) -> ChatProviderRegistry:
    registry = ChatProviderRegistry()

    def factory(configuration: ChatModelSettings) -> WorkflowChatModel:
        model = WorkflowChatModel(
            role=configuration.model, responses=[f"answer:{configuration.model}"]
        )
        records.append((configuration, model))
        return model

    for provider in ("generate-test", "grade-test", "rewrite-test"):
        registry.register(provider, factory)
    return registry


def initial_state(retry_count: int = 0) -> FinGraphUpdate:
    return {"question": "Doanh thu FPT?", "retry_count": retry_count, "web_search_needed": False}


def test_event_loop_socketpair_does_not_allow_provider_connections() -> None:
    reader, writer = socket.socketpair()
    with reader, writer:
        writer.sendall(b"local wakeup")
        assert reader.recv(64) == b"local wakeup"
    with socket.socket() as connection:
        for address in (("127.0.0.1", 11434), ("192.0.2.1", 443)):
            with pytest.raises(RuntimeError, match="Network access is disabled"):
                connection.connect(address)


@pytest.mark.parametrize(("score", "retry_count"), [("yes", 0), ("no", 0), ("no", 2)])
def test_real_compiled_graph_routes_distinct_roles(
    monkeypatch: pytest.MonkeyPatch, score: str, retry_count: int
) -> None:
    generation = WorkflowChatModel(role="generation", responses=["Role-specific financial answer."])
    grading = WorkflowChatModel(role="grading", responses=["unused"], binary_score=score)
    rewrite = WorkflowChatModel(role="query_rewrite", responses=["  FPT annual report 2023  "])
    queries: list[str] = []

    class OfflineSearch:
        def __enter__(self) -> "OfflineSearch":
            return self

        def __exit__(self, *args: object) -> None:
            pass

        def text(self, *, keywords: str, max_results: int) -> list[dict[str, str]]:
            queries.append(keywords)
            assert max_results == 3
            return [
                {"title": "Local report", "href": "https://example.invalid/report", "body": "FPT"}
            ]

    module = ModuleType("duckduckgo_search")
    setattr(module, "DDGS", OfflineSearch)
    monkeypatch.setitem(sys.modules, "duckduckgo_search", module)
    graph = build_finrag_graph(ModelRoles(generation, grading, rewrite))
    assert isinstance(graph, CompiledStateGraph)
    result = graph.invoke(initial_state(retry_count))

    assert result["generation"] == "Role-specific financial answer."
    assert result["question"] == "Doanh thu FPT?"
    assert result["is_relevant"] == score
    assert result["is_hallucinated"] == "no"
    assert result["is_useful"] == "yes"
    assert len(generation.calls) == len(grading.calls) == 1
    generation_call, grading_call = generation.calls[0], grading.calls[0]
    assert generation_call.role == "generation" and generation_call.schema is None
    assert f"QUESTION:\n{result['question']}" in generation_call.input
    for index, document in enumerate(result["documents"], start=1):
        assert f"[SOURCE {index}]" in generation_call.input
        assert document.page_content in generation_call.input
        assert f"source: {document.metadata['source']}" in generation_call.input
        assert f"page: {document.metadata['page']}" in generation_call.input
    assert grading_call.role == "grading" and grading_call.schema is GradeDocuments
    assert f"QUESTION:\n{result['question']}" in grading_call.input
    assert f"CANDIDATE DOCUMENT:\n{result['documents'][0].page_content}" in grading_call.input

    if score == "no" and retry_count < 2:
        assert rewrite.calls == [ModelCall("query_rewrite", "Rewrite query: Doanh thu FPT?")]
        assert result["transformed_question"] == "FPT annual report 2023"
        assert result["retry_count"] == 1
        assert result["web_search_needed"] is False
        assert queries == ["FPT annual report 2023 báo cáo tài chính"]
        assert len(result["documents"]) == len(result["citations"]) == 2
        assert result["citations"][1]["source"] == "DuckDuckGo Web Search"
        assert "Local report" in result["documents"][1].page_content
    else:
        assert rewrite.calls == []
        assert queries == []
        assert result["retry_count"] == retry_count
        assert result["web_search_needed"] is (score == "no")
        assert len(result["documents"]) == len(result["citations"]) == 1


def test_composition_passes_original_configs_and_reuses_models_for_queries() -> None:
    configuration = role_settings()
    records: list[tuple[ChatModelSettings, WorkflowChatModel]] = []
    registry = make_test_registry(records)
    runtime = compose_runtime(configuration, registry=registry)
    assert isinstance(runtime.graph, CompiledStateGraph)
    assert runtime.settings is configuration
    configs = (configuration.generation, configuration.grading, configuration.query_rewrite)
    assert len(records) == 3
    assert all(record[0] is config for record, config in zip(records, configs, strict=True))
    assert runtime.model_roles.generation is records[0][1]
    assert runtime.model_roles.grading is records[1][1]
    assert runtime.model_roles.query_rewrite is records[2][1]
    for _ in range(2):
        assert runtime.graph.invoke(initial_state())["generation"] == "answer:generation"
    assert len(records) == 3
    assert len(records[0][1].calls) == len(records[1][1].calls) == 2
    assert records[2][1].calls == []


def test_default_composition_owns_fresh_empty_registries(monkeypatch: pytest.MonkeyPatch) -> None:
    registries: list[ChatProviderRegistry] = []
    models: list[WorkflowChatModel] = []

    def register(registry: ChatProviderRegistry) -> None:
        with pytest.raises(ProviderNotRegisteredError):
            registry.get("openai")
        registries.append(registry)

        def factory(configuration: ChatModelSettings) -> WorkflowChatModel:
            model = WorkflowChatModel(responses=[configuration.model])
            models.append(model)
            return model

        registry.register("openai", factory)

    monkeypatch.setattr(composition, "register_builtin_chat_providers", register)
    first = compose_runtime(Settings())
    second = compose_runtime(Settings())
    assert len(registries) == 2
    assert registries[0] is not registries[1]
    assert len({id(model) for model in models}) == 6
    assert first.model_roles.generation is not second.model_roles.generation
    assert first.graph is not second.graph


def test_fastapi_lifespan_composes_once_per_instance_and_chat_reuses_graph(
    capsys: pytest.CaptureFixture[str],
) -> None:
    records: list[tuple[ChatModelSettings, WorkflowChatModel]] = []
    runtimes: list[GraphRuntime] = []

    def factory() -> GraphRuntime:
        runtime = compose_runtime(role_settings(), registry=make_test_registry(records))
        runtimes.append(runtime)
        return runtime

    first = create_app(factory)
    second = create_app(factory)
    assert runtimes == []
    with TestClient(first) as client:
        for _ in range(2):
            response = client.post("/api/v1/chat", json={"question": "Doanh thu FPT?"})
            assert response.status_code == 200
            body = response.json()
            assert body["answer"] == "answer:generation"
            assert body["execution_info"]["retry_count"] == 0
            assert body["citations"][0]["source"] == "FPT_Annual_Report_2023.pdf"
        assert len(runtimes) == 1
        assert first.state.runtime is runtimes[0]
        assert client.get("/api/v1/health").json()["llm_provider"] == "generate-test"
    assert not hasattr(first.state, "runtime")
    with TestClient(second) as client:
        assert client.post("/api/v1/chat", json={"question": "Doanh thu FPT?"}).status_code == 200
        assert len(runtimes) == 2
        assert second.state.runtime is runtimes[1]
    assert len(records) == 6
    assert runtimes[0].graph is not runtimes[1].graph
    assert runtimes[0].model_roles.generation is not runtimes[1].model_roles.generation
    assert "[Model role] generation: generate-test / generation" in capsys.readouterr().out


@pytest.mark.parametrize("position", [0, 1, 2])
@pytest.mark.parametrize(
    "error_type",
    [ProviderConfigurationError, ProviderDependencyError, ProviderInitializationError],
)
def test_composition_and_startup_propagate_provider_errors_before_graph_construction(
    monkeypatch: pytest.MonkeyPatch, position: int, error_type: type[Exception]
) -> None:
    error = error_type("safe provider failure")
    calls: list[str] = []
    registry = ChatProviderRegistry()
    configuration = role_settings()
    configs = (configuration.generation, configuration.grading, configuration.query_rewrite)

    def factory(config: ChatModelSettings) -> WorkflowChatModel:
        calls.append(config.model)
        if config is configs[position]:
            raise error
        return WorkflowChatModel(responses=["local"])

    for config in configs:
        registry.register(config.provider, factory)

    def reject_build(roles: ModelRoles) -> None:
        raise AssertionError("Graph must not be built after a model resolution failure")

    monkeypatch.setattr(composition, "build_finrag_graph", reject_build)
    application = create_app(lambda: compose_runtime(configuration, registry=registry))
    with pytest.raises(error_type) as caught, TestClient(application):
        pytest.fail("Provider failure should prevent application startup")
    assert caught.value is error
    assert calls == [config.model for config in configs[: position + 1]]
    assert not hasattr(application.state, "runtime")


def test_unregistered_provider_fails_during_composition() -> None:
    with pytest.raises(ProviderNotRegisteredError, match="generate-test"):
        compose_runtime(role_settings(), registry=ChatProviderRegistry())


def test_existing_engine_fallback_uses_injected_models(monkeypatch: pytest.MonkeyPatch) -> None:
    generation = WorkflowChatModel(role="generation", responses=["fallback engine answer"])
    grading = WorkflowChatModel(role="grading", responses=["unused"])
    rewrite = WorkflowChatModel(role="query_rewrite", responses=["unused"])
    monkeypatch.setitem(sys.modules, "langgraph.graph", None)
    graph = build_finrag_graph(ModelRoles(generation, grading, rewrite))
    assert isinstance(graph, FallbackStateGraph)
    assert graph.invoke(initial_state())["generation"] == "fallback engine answer"
    assert len(generation.calls) == len(grading.calls) == 1
    assert rewrite.calls == []
