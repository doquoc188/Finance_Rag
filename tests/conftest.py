"""Offline boundaries and opt-in model fixtures for workflow tests."""

import socket
import sys
from tempfile import TemporaryDirectory

import pytest

from tests.chat_models import WorkflowChatModel


@pytest.fixture
def workflow_chat_model() -> WorkflowChatModel:
    model = WorkflowChatModel(responses=["Offline workflow test response."])
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

    if sys.platform == "win32":
        # Windows asyncio builds its wakeup pipe using a loopback socket pair.
        # Only this pair's connection bypasses the guard; normal connects stay blocked.
        original_connect = socket.socket.connect

        def local_socketpair(
            family: int = socket.AF_INET, type: int = socket.SOCK_STREAM, proto: int = 0
        ) -> tuple[socket.socket, socket.socket]:
            if family != socket.AF_INET or type != socket.SOCK_STREAM or proto != 0:
                raise ValueError("Offline event-loop socket pairs support IPv4 TCP only")
            with socket.socket(family, type, proto) as listener:
                listener.bind(("127.0.0.1", 0))
                listener.listen(1)
                client = socket.socket(family, type, proto)
                try:
                    original_connect(client, listener.getsockname())
                    server, _ = listener.accept()
                    if (
                        server.getpeername() != client.getsockname()
                        or server.getsockname() != client.getpeername()
                    ):
                        server.close()
                        raise RuntimeError("Unexpected connection to the offline socket pair")
                    return server, client
                except BaseException:
                    client.close()
                    raise

        monkeypatch.setattr(socket, "socketpair", local_socketpair)

    def block_network(*args: object, **kwargs: object) -> None:
        raise RuntimeError("Network access is disabled in the offline test suite")

    monkeypatch.setattr(socket.socket, "connect", block_network)
    monkeypatch.setattr(socket.socket, "connect_ex", block_network)
    monkeypatch.setattr(socket, "create_connection", block_network)
