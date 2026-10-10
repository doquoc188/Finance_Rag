"""The standalone demo must not substitute a fabricated answer for provider failure."""

import json
from io import BytesIO
from unittest.mock import Mock

import pytest

import simple_app
from app.core.errors import ModelConfigurationError, ProviderUnavailableError


@pytest.fixture(autouse=True)
def demo_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    # GEMINI_API_KEY is used only by this standalone demo, not app.config.Settings.
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)


def test_demo_requires_a_configured_provider() -> None:
    with pytest.raises(ModelConfigurationError, match="GROQ_API_KEY or GEMINI_API_KEY"):
        simple_app.run_llm("Test question")


@pytest.mark.parametrize("provider", ["groq", "gemini"])
def test_demo_provider_failure_is_safe_and_preserves_cause(
    monkeypatch: pytest.MonkeyPatch, provider: str, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv(f"{provider.upper()}_API_KEY", "unit-test-credential")
    cause = OSError("unit-test-private-value")
    monkeypatch.setattr(simple_app.urllib.request, "urlopen", Mock(side_effect=cause))

    with pytest.raises(ProviderUnavailableError, match="No configured demo provider") as caught:
        simple_app.run_llm("Test question")

    provider_error = caught.value.__cause__
    assert isinstance(provider_error, ProviderUnavailableError)
    assert provider_error.__cause__ is cause
    assert "unit-test-private-value" not in str(caught.value)
    assert "unit-test-private-value" not in str(provider_error)
    assert "unit-test-credential" not in str(provider_error)
    captured = capsys.readouterr()
    assert captured.out == captured.err == ""


@pytest.mark.parametrize("provider", ["groq", "gemini"])
def test_demo_success_preserves_request_and_response(
    monkeypatch: pytest.MonkeyPatch, provider: str
) -> None:
    monkeypatch.setenv(f"{provider.upper()}_API_KEY", "unit-test-credential")
    response = (
        {"choices": [{"message": {"content": "Provider answer"}}]}
        if provider == "groq"
        else {"candidates": [{"content": {"parts": [{"text": "Provider answer"}]}}]}
    )
    urlopen = Mock(return_value=BytesIO(json.dumps(response).encode()))
    monkeypatch.setattr(simple_app.urllib.request, "urlopen", urlopen)

    assert simple_app.run_llm("Test question") == "Provider answer"
    request = urlopen.call_args.args[0]
    payload = json.loads(request.data)
    if provider == "groq":
        assert payload["model"] == "llama-3.3-70b-versatile"
        assert payload["messages"] == [{"role": "user", "content": "Test question"}]
        assert payload["temperature"] == 0.1
        assert request.get_header("Authorization") == "Bearer unit-test-credential"
    else:
        assert payload == {"contents": [{"parts": [{"text": "Test question"}]}]}
        assert "gemini-1.5-flash:generateContent" in request.full_url


def test_demo_preserves_existing_real_provider_retry_order(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GROQ_API_KEY", "unit-test-credential")
    monkeypatch.setenv("GEMINI_API_KEY", "unit-test-credential")
    calls: list[str] = []

    def groq_fails(prompt: str, api_key: str) -> str:
        calls.append("groq")
        raise ProviderUnavailableError("Groq unavailable")

    def gemini_succeeds(prompt: str, api_key: str) -> str:
        calls.append("gemini")
        return "Provider answer"

    monkeypatch.setattr(simple_app, "call_groq_api", groq_fails)
    monkeypatch.setattr(simple_app, "call_gemini_api", gemini_succeeds)

    assert simple_app.run_llm("Test question") == "Provider answer"
    assert calls == ["groq", "gemini"]


def test_demo_empty_provider_answer_is_not_replaced(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GROQ_API_KEY", "unit-test-credential")
    monkeypatch.setattr(simple_app, "call_groq_api", Mock(return_value=""))

    with pytest.raises(ProviderUnavailableError, match="No configured demo provider"):
        simple_app.run_llm("Test question")
