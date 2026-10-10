"""Offline composition tests for the manual retrieval benchmark runner."""

from pathlib import Path

import pytest

from evals.retrieval import run_benchmark
from finrag.evaluation import RetrievalBenchmark, load_retrieval_benchmark

BENCHMARK_ROOT = Path(__file__).parent.parent / "evals" / "retrieval"


def load_benchmark() -> RetrievalBenchmark:
    return load_retrieval_benchmark(
        BENCHMARK_ROOT / "corpus.jsonl",
        BENCHMARK_ROOT / "queries.jsonl",
        name=run_benchmark.BENCHMARK_NAME,
    )


def test_missing_local_models_leave_only_actual_bm25_measured(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    attempts: list[str] = []

    def missing(model_name: str) -> str:
        attempts.append(model_name)
        raise run_benchmark.LocalModelUnavailableError(f"missing local model: {model_name}")

    monkeypatch.setattr(run_benchmark, "_resolve_local_model", missing)

    report = run_benchmark.run_selected_strategies(
        load_benchmark(), run_benchmark.RETRIEVAL_STRATEGY_NAMES
    )

    assert attempts == [
        run_benchmark.DEFAULT_EMBEDDING_MODEL,
        run_benchmark.DEFAULT_RERANKER_MODEL,
    ]
    assert [result.strategy for result in report.strategies] == ["bm25"]
    assert set(report.unmeasured) == {
        "dense",
        "hybrid_rrf",
        "hybrid_rrf_reranked",
    }
    assert all("missing local model" in reason for reason in report.unmeasured.values())


def test_bm25_only_does_not_inspect_or_construct_models(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def reject_model(model_name: str) -> str:
        raise AssertionError(f"BM25-only run must not inspect {model_name}")

    monkeypatch.setattr(run_benchmark, "_resolve_local_model", reject_model)

    report = run_benchmark.run_selected_strategies(load_benchmark(), ("bm25",))

    assert [result.strategy for result in report.strategies] == ["bm25"]
    assert report.unmeasured == {}
    assert report.configuration["final_top_k"] == 10
    assert report.configuration["fused_top_k"] == 20
    assert report.configuration["model_loading"] == "local_files_only"


@pytest.mark.parametrize(
    ("strategies", "message"),
    [
        ((), "at least one"),
        (("dense", "dense"), "duplicate"),
        (("unknown",), "unknown"),
    ],
)
def test_strategy_selection_fails_explicitly(strategies: tuple[str, ...], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        run_benchmark.run_selected_strategies(load_benchmark(), strategies)


def test_local_model_resolution_rejects_blank_name() -> None:
    with pytest.raises(ValueError, match="must not be blank"):
        run_benchmark._resolve_local_model(" \n ")
