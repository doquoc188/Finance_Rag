"""Offline settings validation for the new package, independent of legacy settings."""

import os
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from finrag.config import (
    ChatModelSettings,
    EmbeddingSettings,
    GraphRAGSettings,
    IndexingSettings,
    RetrievalSettings,
    Settings,
)
from finrag.domain import RetrievalStrategy


@pytest.fixture(autouse=True)
def clean_finrag_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in tuple(os.environ):
        if name.upper().startswith("FINRAG_"):
            monkeypatch.delenv(name)


def test_defaults_need_no_credentials_and_roles_are_separate() -> None:
    settings = Settings()

    assert settings.generation.provider == "openai"
    assert settings.generation.model == "gpt-4o-mini"
    assert settings.generation.temperature == 0.0
    assert settings.generation is not settings.grading
    assert settings.grading is not settings.query_rewrite
    assert settings.embedding.provider == "huggingface"
    assert settings.retrieval.strategy is RetrievalStrategy.DENSE
    assert settings.retrieval.final_top_k == 4
    assert settings.retrieval.rrf_k == 60
    assert settings.retrieval.reranker_enabled is False
    assert settings.retrieval.reranker_model == "BAAI/bge-reranker-v2-m3"
    assert settings.retrieval.max_rewrite_attempts == 2
    assert settings.graphrag.enabled is False
    assert settings.graphrag.index_name is None
    assert settings.graphrag.root_dir == Path(".finrag/graphrag")
    assert "temperature" not in EmbeddingSettings.model_fields


def test_explicit_roles_do_not_share_model_selection() -> None:
    settings = Settings(
        generation=ChatModelSettings(provider="alpha", model="generation", temperature=0.2),
        grading=ChatModelSettings(provider="beta", model="grading"),
        query_rewrite=ChatModelSettings(provider="gamma", model="rewrite", temperature=0.1),
        embedding=EmbeddingSettings(provider="local", model="embedding"),
    )

    assert settings.generation.provider == "alpha"
    assert settings.grading.provider == "beta"
    assert settings.query_rewrite.provider == "gamma"
    assert settings.embedding.model == "embedding"
    assert settings.generation.model != settings.grading.model != settings.query_rewrite.model


def test_nested_environment_loading(monkeypatch: pytest.MonkeyPatch) -> None:
    values = {
        "FINRAG_GENERATION__PROVIDER": "alpha",
        "FINRAG_GENERATION__MODEL": "generation-model",
        "FINRAG_GENERATION__TEMPERATURE": "0.3",
        "FINRAG_GRADING__PROVIDER": "beta",
        "FINRAG_GRADING__MODEL": "grading-model",
        "FINRAG_QUERY_REWRITE__PROVIDER": "gamma",
        "FINRAG_QUERY_REWRITE__MODEL": "rewrite-model",
        "FINRAG_EMBEDDING__PROVIDER": "local",
        "FINRAG_EMBEDDING__MODEL": "embedding-model",
        "FINRAG_RETRIEVAL__FUSED_TOP_K": "8",
        "FINRAG_RETRIEVAL__FINAL_TOP_K": "3",
        "FINRAG_RETRIEVAL__MAX_REWRITE_ATTEMPTS": "5",
        "FINRAG_RETRIEVAL__RRF_K": "50",
        "FINRAG_RETRIEVAL__RERANKER_ENABLED": "true",
        "FINRAG_RETRIEVAL__RERANKER_MODEL": "local/reranker",
        "FINRAG_RETRIEVAL__STRATEGY": "graph",
        "FINRAG_GRAPHRAG__ENABLED": "true",
        "FINRAG_GRAPHRAG__INDEX_NAME": "annual-reports",
        "FINRAG_GRAPHRAG__ROOT_DIR": "state/graphrag",
    }
    for name, value in values.items():
        monkeypatch.setenv(name, value)

    settings = Settings()

    assert settings.generation == ChatModelSettings(
        provider="alpha", model="generation-model", temperature=0.3
    )
    assert settings.grading.model == "grading-model"
    assert settings.grading.provider == "beta"
    assert settings.query_rewrite.model == "rewrite-model"
    assert settings.query_rewrite.provider == "gamma"
    assert settings.embedding == EmbeddingSettings(provider="local", model="embedding-model")
    assert settings.retrieval.fused_top_k == 8
    assert settings.retrieval.final_top_k == 3
    assert settings.retrieval.max_rewrite_attempts == 5
    assert settings.retrieval.rrf_k == 50
    assert settings.retrieval.reranker_enabled is True
    assert settings.retrieval.reranker_model == "local/reranker"
    assert settings.retrieval.strategy is RetrievalStrategy.GRAPH
    assert settings.graphrag == GraphRAGSettings(
        enabled=True,
        index_name="annual-reports",
        root_dir=Path("state/graphrag"),
    )


def test_partial_role_override_keeps_defaults_and_other_roles(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("FINRAG_GENERATION__MODEL", "changed")
    settings = Settings()
    assert settings.generation.model == "changed"
    assert settings.generation.provider == "openai"
    assert settings.grading.model == settings.query_rewrite.model == "gpt-4o-mini"


@pytest.mark.parametrize("model_type", [ChatModelSettings, EmbeddingSettings])
@pytest.mark.parametrize("field", ["provider", "model"])
def test_model_identifiers_cannot_be_blank(
    model_type: type[ChatModelSettings] | type[EmbeddingSettings], field: str
) -> None:
    with pytest.raises(ValidationError, match=field):
        model_type.model_validate({field: " \n "})


@pytest.mark.parametrize("temperature", [-0.1, float("inf"), float("nan")])
def test_invalid_temperature(temperature: float) -> None:
    with pytest.raises(ValidationError, match="temperature"):
        ChatModelSettings(temperature=temperature)


@pytest.mark.parametrize(
    "field", ["dense_top_k", "sparse_top_k", "fused_top_k", "final_top_k", "rrf_k"]
)
@pytest.mark.parametrize("value", [-1, 0, True, 1.5])
def test_invalid_retrieval_counts(field: str, value: Any) -> None:
    with pytest.raises(ValidationError, match=field):
        RetrievalSettings.model_validate({field: value})


def test_final_top_k_cannot_exceed_fused_top_k() -> None:
    with pytest.raises(ValidationError, match="final_top_k must not exceed fused_top_k"):
        RetrievalSettings(fused_top_k=3, final_top_k=4)


def test_fused_top_k_cannot_exceed_combined_candidate_limits() -> None:
    with pytest.raises(ValidationError, match="combined candidate limits"):
        RetrievalSettings(dense_top_k=2, sparse_top_k=3, fused_top_k=6, final_top_k=4)


def test_reranker_model_cannot_be_blank() -> None:
    with pytest.raises(ValidationError, match="reranker_model"):
        RetrievalSettings(reranker_model=" \n ")


def test_invalid_environment_count_is_validated(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FINRAG_RETRIEVAL__FINAL_TOP_K", "-1")
    with pytest.raises(ValidationError, match="final_top_k"):
        Settings()


@pytest.mark.parametrize("value", [-1, True, 1.5])
def test_invalid_max_rewrite_attempts(value: Any) -> None:
    with pytest.raises(ValidationError, match="max_rewrite_attempts"):
        RetrievalSettings.model_validate({"max_rewrite_attempts": value})


def test_zero_max_rewrite_attempts_disables_rewriting() -> None:
    assert RetrievalSettings(max_rewrite_attempts=0).max_rewrite_attempts == 0


def test_graph_configuration_is_explicitly_opt_in() -> None:
    with pytest.raises(ValidationError, match="index_name"):
        GraphRAGSettings(enabled=True)
    with pytest.raises(ValidationError, match="requires graphrag.enabled"):
        Settings(retrieval=RetrievalSettings(strategy=RetrievalStrategy.GRAPH))


def test_settings_and_nested_configuration_are_frozen() -> None:
    settings = Settings()
    with pytest.raises(ValidationError, match="frozen"):
        setattr(settings, "generation", ChatModelSettings(model="changed"))
    with pytest.raises(ValidationError, match="frozen"):
        setattr(settings.generation, "model", "changed")


def test_constructor_typos_are_rejected() -> None:
    with pytest.raises(ValidationError, match="extra_forbidden"):
        ChatModelSettings.model_validate({"modle": "typo"})


def test_legacy_environment_and_dotenv_are_not_loaded(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("OPENAI_MODEL", "legacy-model")
    (tmp_path / ".env").write_text("FINRAG_GENERATION__MODEL=dotenv-model\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    assert Settings().generation.model == "gpt-4o-mini"


def test_index_manifest_path_has_local_default_and_nested_environment_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert IndexingSettings().manifest_path == Path(".finrag/index-manifest.json")
    monkeypatch.setenv("FINRAG_INDEXING__MANIFEST_PATH", "state/custom-index.json")
    assert Settings().indexing.manifest_path == Path("state/custom-index.json")
