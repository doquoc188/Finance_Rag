"""Offline contract tests for post-RRF reranking."""

from types import MappingProxyType

import pytest

from finrag.domain import RetrievalStrategy, ScoreType
from finrag.retrieval import HybridRetriever
from tests.rerankers import FailingReranker, ScoreReranker
from tests.test_finrag_retrieval import RecordingRetriever, evidence, make_chunk


def branch_results(chunk_ids: list[str], strategy: RetrievalStrategy) -> RecordingRetriever:
    return RecordingRetriever(
        tuple(
            evidence(make_chunk(chunk_id), rank, strategy, 1.0)
            for rank, chunk_id in enumerate(chunk_ids, start=1)
        )
    )


def test_reranking_changes_rrf_order_and_score_semantics() -> None:
    dense = branch_results(["A", "B", "C"], RetrievalStrategy.DENSE)
    sparse = branch_results(["A", "B", "C"], RetrievalStrategy.SPARSE)
    reranker = ScoreReranker({"A": 0.5, "B": -1.0, "C": 4.0})
    hybrid = HybridRetriever(
        dense,
        sparse,
        dense_top_k=3,
        sparse_top_k=3,
        fused_top_k=3,
        reranker=reranker,
    )

    results = hybrid.retrieve("query", limit=3)

    assert [item.chunk.chunk_id for item in results] == ["C", "A", "B"]
    assert [item.score.value for item in results if item.score] == [4.0, 0.5, -1.0]
    assert all(item.score and item.score.score_type is ScoreType.RERANKER for item in results)
    assert reranker.calls == [("query", ("A", "B", "C"), 3)]


def test_reranker_receives_only_fused_candidates_then_applies_final_top_k() -> None:
    ids = [f"chunk-{index}" for index in range(12)]
    dense = branch_results(ids, RetrievalStrategy.DENSE)
    sparse = branch_results([], RetrievalStrategy.SPARSE)
    reranker = ScoreReranker({chunk_id: float(index) for index, chunk_id in enumerate(ids)})
    hybrid = HybridRetriever(
        dense,
        sparse,
        dense_top_k=12,
        sparse_top_k=1,
        fused_top_k=10,
        reranker=reranker,
    )

    results = hybrid.retrieve("query", limit=3)

    assert len(results) == 3
    assert reranker.calls == [("query", tuple(ids[:10]), 3)]
    assert [item.chunk.chunk_id for item in results] == ["chunk-9", "chunk-8", "chunk-7"]


def test_disabled_path_preserves_rrf_order_and_truncates_without_reranker() -> None:
    dense = branch_results(["A", "B", "C"], RetrievalStrategy.DENSE)
    sparse = branch_results(["A", "B", "C"], RetrievalStrategy.SPARSE)
    hybrid = HybridRetriever(
        dense,
        sparse,
        dense_top_k=3,
        sparse_top_k=3,
        fused_top_k=3,
    )

    results = hybrid.retrieve("query", limit=2)

    assert [item.chunk.chunk_id for item in results] == ["A", "B"]
    assert all(item.score and item.score.score_type is ScoreType.RRF for item in results)


def test_enabled_reranker_failure_propagates_without_rrf_fallback() -> None:
    error = RuntimeError("reranker failed")
    hybrid = HybridRetriever(
        branch_results(["A"], RetrievalStrategy.DENSE),
        branch_results(["A"], RetrievalStrategy.SPARSE),
        dense_top_k=1,
        sparse_top_k=1,
        fused_top_k=1,
        reranker=FailingReranker(error),
    )

    with pytest.raises(RuntimeError) as caught:
        hybrid.retrieve("query", limit=1)

    assert caught.value is error


def test_reranking_preserves_chunk_identity_text_and_metadata() -> None:
    chunk = make_chunk("identity", "FPT EPS 2025 was 7,200 VND.")
    original_metadata = chunk.metadata
    assert isinstance(original_metadata, MappingProxyType)
    dense = RecordingRetriever((evidence(chunk, 1, RetrievalStrategy.DENSE, 0.9),))
    sparse = RecordingRetriever((evidence(chunk, 1, RetrievalStrategy.SPARSE, 9.0),))

    (result,) = HybridRetriever(
        dense,
        sparse,
        dense_top_k=1,
        sparse_top_k=1,
        fused_top_k=1,
        reranker=ScoreReranker({"identity": 5.0}),
    ).retrieve("FPT EPS 2025", limit=1)

    assert result.chunk is chunk
    assert result.chunk.chunk_id == "identity"
    assert result.chunk.document_id == chunk.document_id
    assert result.chunk.text == chunk.text
    assert result.chunk.metadata is original_metadata


def test_finance_reranking_promotes_exact_period_and_metric() -> None:
    ids = ["eps-2024", "revenue-2025", "eps-2025"]
    dense = branch_results(ids, RetrievalStrategy.DENSE)
    sparse = branch_results(ids, RetrievalStrategy.SPARSE)
    reranker = ScoreReranker({"eps-2024": 0.2, "revenue-2025": 0.1, "eps-2025": 0.95})

    results = HybridRetriever(
        dense,
        sparse,
        dense_top_k=3,
        sparse_top_k=3,
        fused_top_k=3,
        reranker=reranker,
    ).retrieve("FPT EPS 2025", limit=3)

    assert results[0].chunk.chunk_id == "eps-2025"
    assert results[0].score and results[0].score.score_type is ScoreType.RERANKER
