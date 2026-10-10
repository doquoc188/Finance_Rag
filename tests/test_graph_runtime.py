"""Real compiled graph and application composition, with explicit offline models."""

import socket
import sys

import pytest
from fastapi.testclient import TestClient
from langgraph.graph.state import CompiledStateGraph

from app import runtime as composition
from app.core import llm
from app.core.vectorstore import HybridFinancialRetriever
from app.graph.nodes import GradeDocuments
from app.graph.state import FinGraphUpdate
from app.graph.workflow import FallbackStateGraph, build_finrag_graph
from app.main import create_app
from app.runtime import GraphRuntime, compose_runtime
from finrag.config import ChatModelSettings, RetrievalSettings, Settings
from finrag.models import ModelRoles
from finrag.providers import ChatProviderRegistry
from finrag.providers.errors import (
    ProviderConfigurationError,
    ProviderDependencyError,
    ProviderInitializationError,
    ProviderNotRegisteredError,
)
from finrag.retrieval import Retriever
from tests.chat_models import WorkflowChatModel


@pytest.fixture(autouse=True)
def isolate_runtime(monkeypatch: pytest.MonkeyPatch) -> None:
    def reject_legacy_model(*args: object, **kwargs: object) -> None:
        raise AssertionError("The graph must not construct a legacy chat model")

    monkeypatch.setattr(llm, "get_llm", reject_legacy_model)


def role_settings(max_rewrite_attempts: int = 2) -> Settings:
    return Settings(
        generation=ChatModelSettings(provider="generate-test", model="generation", temperature=0.1),
        grading=ChatModelSettings(provider="grade-test", model="grading"),
        query_rewrite=ChatModelSettings(
            provider="rewrite-test", model="query_rewrite", temperature=0.2
        ),
        retrieval=RetrievalSettings(max_rewrite_attempts=max_rewrite_attempts),
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


def initial_state() -> FinGraphUpdate:
    return {
        "question": "Doanh thu FPT?",
        "rewrite_attempts": 0,
        "retry_count": 0,
        "web_search_needed": False,
    }


def test_event_loop_socketpair_does_not_allow_provider_connections() -> None:
    reader, writer = socket.socketpair()
    with reader, writer:
        writer.sendall(b"local wakeup")
        assert reader.recv(64) == b"local wakeup"
    with socket.socket() as connection:
        for address in (("127.0.0.1", 11434), ("192.0.2.1", 443)):
            with pytest.raises(RuntimeError, match="Network access is disabled"):
                connection.connect(address)


def test_real_compiled_graph_routes_distinct_roles() -> None:
    generation = WorkflowChatModel(role="generation", responses=["Role-specific financial answer."])
    grading = WorkflowChatModel(role="grading", responses=["unused"], binary_score="yes")
    rewrite = WorkflowChatModel(role="query_rewrite", responses=["  FPT annual report 2023  "])
    graph = build_finrag_graph(ModelRoles(generation, grading, rewrite))
    assert isinstance(graph, CompiledStateGraph)
    result = graph.invoke(initial_state())

    assert result["generation"] == "Role-specific financial answer."
    assert result["question"] == "Doanh thu FPT?"
    assert result["is_relevant"] == "yes"
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

    assert rewrite.calls == []
    assert result["rewrite_attempts"] == result["retry_count"] == 0
    assert result["web_search_needed"] is False
    assert len(result["documents"]) == len(result["citations"]) == 1


def test_composition_passes_original_configs_and_reuses_models_for_queries() -> None:
    configuration = role_settings()
    records: list[tuple[ChatModelSettings, WorkflowChatModel]] = []
    registry = make_test_registry(records)
    runtime = compose_runtime(configuration, registry=registry)
    assert isinstance(runtime.graph, CompiledStateGraph)
    assert runtime.settings is configuration
    assert isinstance(runtime.retriever, HybridFinancialRetriever)
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


def test_composition_passes_validated_rewrite_budget_to_graph(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: list[tuple[int, int, Retriever | None]] = []
    original_builder = composition.build_finrag_graph

    def record_builder(
        roles: ModelRoles,
        *,
        max_rewrite_attempts: int = 2,
        retriever: Retriever | None = None,
        retrieval_limit: int = 4,
    ) -> CompiledStateGraph:
        captured.append((max_rewrite_attempts, retrieval_limit, retriever))
        graph = original_builder(
            roles,
            max_rewrite_attempts=max_rewrite_attempts,
            retriever=retriever,
            retrieval_limit=retrieval_limit,
        )
        assert isinstance(graph, CompiledStateGraph)
        return graph

    monkeypatch.setattr(composition, "build_finrag_graph", record_builder)
    configuration = role_settings(max_rewrite_attempts=0)
    runtime = compose_runtime(configuration, registry=make_test_registry([]))

    assert runtime.settings is configuration
    assert captured == [(0, configuration.retrieval.final_top_k, runtime.retriever)]


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
            assert "rewrite_attempts" not in body["execution_info"]
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


def test_existing_engine_fallback_honors_zero_rewrite_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    generation = WorkflowChatModel(role="generation", responses=["must not be called"])
    grading = WorkflowChatModel(role="grading", responses=["unused"], binary_score="no")
    rewrite = WorkflowChatModel(role="query_rewrite", responses=["must not be called"])
    monkeypatch.setitem(sys.modules, "langgraph.graph", None)
    graph = build_finrag_graph(ModelRoles(generation, grading, rewrite), max_rewrite_attempts=0)

    assert isinstance(graph, FallbackStateGraph)
    result = graph.invoke(initial_state())
    assert result["generation"] == (
        "Không đủ bằng chứng từ tài liệu được cung cấp để trả lời câu hỏi này."
    )
    assert result["rewrite_attempts"] == 0
    assert generation.calls == rewrite.calls == []
