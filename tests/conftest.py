"""Offline boundaries and opt-in model fixtures for workflow tests."""

import socket
from tempfile import TemporaryDirectory
from typing import Any

import pytest
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.runnables import Runnable, RunnableLambda
from pydantic import BaseModel


class WorkflowChatModel(FakeListChatModel):
    """Real chat-model interface with fixed, test-only answers and relevance grades."""

    def with_structured_output(
        self, schema: Any, *, include_raw: bool = False, **kwargs: Any
    ) -> Runnable[Any, BaseModel]:
        if include_raw or not isinstance(schema, type) or not issubclass(schema, BaseModel):
            raise NotImplementedError("The test model supports Pydantic schemas only")

        def grade(_input: Any) -> BaseModel:
            return schema.model_validate({"binary_score": "yes", "is_useful": "yes"})

        return RunnableLambda(grade)


@pytest.fixture
def workflow_chat_model(monkeypatch: pytest.MonkeyPatch) -> WorkflowChatModel:
    from app.graph import nodes

    model = WorkflowChatModel(responses=["Offline workflow test response."])

    def get_test_llm(temperature: float = 0.0) -> WorkflowChatModel:
        return model

    # Patch the reference consumed by nodes, not the original factory's module.
    monkeypatch.setattr(nodes, "get_llm", get_test_llm)
    monkeypatch.setattr(nodes, "global_retriever", None)
    return model


def pytest_configure(config: pytest.Config) -> None:
    # Run before collection: Settings() reads both the environment and .env during
    # application imports. Test defaults take precedence over either source.
    monkeypatch = pytest.MonkeyPatch()
    config.add_cleanup(monkeypatch.undo)
    model_cache = TemporaryDirectory(prefix="finance-rag-tests-")
    config.add_cleanup(model_cache.cleanup)

    test_environment = {
        "DEBUG": "false",
        "LLM_PROVIDER": "openai",
        "OPENAI_API_KEY": "",
        "GROQ_API_KEY": "",
        "TAVILY_API_KEY": "",
        "QDRANT_LOCATION": ":memory:",
        "EMBEDDING_MODEL": "sentence-transformers/all-MiniLM-L6-v2",
        "LANGCHAIN_TRACING_V2": "false",
        "LANGSMITH_TRACING": "false",
        "LANGCHAIN_TRACING": "false",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "HF_HUB_DISABLE_TELEMETRY": "1",
        "HF_HOME": model_cache.name,
        "HF_HUB_CACHE": model_cache.name,
    }
    for name, value in test_environment.items():
        monkeypatch.setenv(name, value)

    def block_network(*args, **kwargs) -> None:
        raise RuntimeError("Network access is disabled in the offline test suite")

    monkeypatch.setattr(socket.socket, "connect", block_network)
    monkeypatch.setattr(socket.socket, "connect_ex", block_network)
    monkeypatch.setattr(socket, "create_connection", block_network)
