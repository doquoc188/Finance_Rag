"""Deterministic offline tests for BM25, RRF, and hybrid composition."""

from collections.abc import Sequence

import pytest

from finrag.domain import (
    DocumentChunk,
    RetrievalScore,
    RetrievalStrategy,
    RetrievedEvidence,
    ScoreType,
    SourceReference,
)
from finrag.retrieval import BM25Retriever, HybridRetriever, reciprocal_rank_fusion


def make_chunk(chunk_id: str, text: str | None = None) -> DocumentChunk:
    source = SourceReference("report", "report.pdf", page=1)
    return DocumentChunk(chunk_id, source.document_id, text or chunk_id, source)


def evidence(
    chunk: DocumentChunk,
    rank: int,
    strategy: RetrievalStrategy,
    score: float,
) -> RetrievedEvidence:
    score_type = ScoreType.COSINE if strategy is RetrievalStrategy.DENSE else ScoreType.BM25
    return RetrievedEvidence(chunk, rank, strategy, RetrievalScore(score, score_type))


class RecordingRetriever:
    def __init__(self, results: Sequence[RetrievedEvidence]) -> None:
        self.results = tuple(results)
        self.calls: list[tuple[str, int]] = []

    def retrieve(self, query: str, *, limit: int) -> tuple[RetrievedEvidence, ...]:
        self.calls.append((query, limit))
        return self.results[:limit]


class FailingRetriever:
    def __init__(self, error: RuntimeError) -> None:
        self.error = error

    def retrieve(self, query: str, *, limit: int) -> tuple[RetrievedEvidence, ...]:
        raise self.error


def test_rrf_uses_exact_rank_contributions_and_stable_ties() -> None:
    chunks = {name: make_chunk(name) for name in "ABCD"}
    dense = tuple(
        evidence(chunks[name], rank, RetrievalStrategy.DENSE, 1.0 - rank / 10)
        for rank, name in enumerate("ABC", start=1)
    )
    sparse = tuple(
        evidence(chunks[name], rank, RetrievalStrategy.SPARSE, 40.0 - rank)
        for rank, name in enumerate("CDA", start=1)
    )

    fused = reciprocal_rank_fusion((dense, sparse), k=60)

    assert [item.chunk.chunk_id for item in fused] == ["A", "C", "B", "D"]
    expected = {
        "A": 1 / 61 + 1 / 63,
        "C": 1 / 63 + 1 / 61,
        "B": 1 / 62,
        "D": 1 / 62,
    }
    assert [item.score.value for item in fused if item.score] == pytest.approx(
        [expected[item.chunk.chunk_id] for item in fused]
    )
    assert all(item.strategy is RetrievalStrategy.HYBRID for item in fused)
    assert all(item.score and item.score.score_type is ScoreType.RRF for item in fused)


def test_rrf_ignores_raw_score_values() -> None:
    chunk_a = make_chunk("A")
    chunk_b = make_chunk("B")
    first = (
        (
            evidence(chunk_a, 1, RetrievalStrategy.DENSE, 0.91),
            evidence(chunk_b, 2, RetrievalStrategy.DENSE, 0.90),
        ),
        (
            evidence(chunk_b, 1, RetrievalStrategy.SPARSE, 42.0),
            evidence(chunk_a, 2, RetrievalStrategy.SPARSE, 3.0),
        ),
    )
    changed_scales = (
        (
            evidence(chunk_a, 1, RetrievalStrategy.DENSE, -999.0),
            evidence(chunk_b, 2, RetrievalStrategy.DENSE, 999.0),
        ),
        (
            evidence(chunk_b, 1, RetrievalStrategy.SPARSE, 0.0001),
            evidence(chunk_a, 2, RetrievalStrategy.SPARSE, 1_000_000.0),
        ),
    )

    original = reciprocal_rank_fusion(first)
    changed = reciprocal_rank_fusion(changed_scales)

    assert [(item.chunk.chunk_id, item.score) for item in original] == [
        (item.chunk.chunk_id, item.score) for item in changed
    ]


def test_rrf_fuses_by_chunk_id_and_not_equal_text() -> None:
    shared = make_chunk("shared", "same text")
    distinct = make_chunk("distinct", "same text")

    fused = reciprocal_rank_fusion(
        (
            (evidence(shared, 1, RetrievalStrategy.DENSE, 0.9),),
            (
                evidence(shared, 1, RetrievalStrategy.SPARSE, 10.0),
                evidence(distinct, 2, RetrievalStrategy.SPARSE, 9.0),
            ),
        )
    )

    assert [item.chunk.chunk_id for item in fused] == ["shared", "distinct"]
    assert fused[0].score and fused[0].score.value == pytest.approx(2 / 61)


@pytest.mark.parametrize(
    ("dense_ids", "sparse_ids", "expected"),
    [
        (("A",), (), ["A"]),
        ((), ("B",), ["B"]),
        ((), (), []),
    ],
)
def test_hybrid_accepts_empty_branches(
    dense_ids: tuple[str, ...], sparse_ids: tuple[str, ...], expected: list[str]
) -> None:
    dense = RecordingRetriever(
        [
            evidence(make_chunk(chunk_id), rank, RetrievalStrategy.DENSE, 0.8)
            for rank, chunk_id in enumerate(dense_ids, start=1)
        ]
    )
    sparse = RecordingRetriever(
        [
            evidence(make_chunk(chunk_id), rank, RetrievalStrategy.SPARSE, 8.0)
            for rank, chunk_id in enumerate(sparse_ids, start=1)
        ]
    )

    result = HybridRetriever(dense, sparse, dense_top_k=2, sparse_top_k=3, fused_top_k=5).retrieve(
        "query", limit=4
    )

    assert [item.chunk.chunk_id for item in result] == expected
    assert dense.calls == [("query", 2)]
    assert sparse.calls == [("query", 3)]


@pytest.mark.parametrize("failing_branch", ["dense", "sparse"])
def test_hybrid_propagates_retriever_failures(failing_branch: str) -> None:
    error = RuntimeError(f"{failing_branch} failed")
    empty = RecordingRetriever(())
    dense = FailingRetriever(error) if failing_branch == "dense" else empty
    sparse = FailingRetriever(error) if failing_branch == "sparse" else empty

    with pytest.raises(RuntimeError) as caught:
        HybridRetriever(dense, sparse).retrieve("query", limit=4)

    assert caught.value is error


def test_bm25_ranks_finance_specific_lexical_evidence_first() -> None:
    chunks = (
        make_chunk("revenue", "FPT revenue grew 12.5 percent in 2024."),
        make_chunk("eps", "FPT EPS 2025 reached 7,200 VND per share."),
        make_chunk("other", "VNM profit margin improved in 2025."),
    )

    results = BM25Retriever(chunks).retrieve("FPT EPS 2025", limit=3)

    assert results[0].chunk.chunk_id == "eps"
    assert results[0].strategy is RetrievalStrategy.SPARSE
    assert results[0].score and results[0].score.score_type is ScoreType.BM25


def test_sparse_rank_can_move_relevant_chunk_to_hybrid_first() -> None:
    relevant = make_chunk("relevant", "FPT EPS 2025 reached 7,200 VND.")
    first = make_chunk("dense-first")
    second = make_chunk("dense-second")
    dense = RecordingRetriever(
        (
            evidence(first, 1, RetrievalStrategy.DENSE, 0.99),
            evidence(second, 2, RetrievalStrategy.DENSE, 0.98),
            evidence(relevant, 3, RetrievalStrategy.DENSE, 0.97),
        )
    )
    sparse = RecordingRetriever((evidence(relevant, 1, RetrievalStrategy.SPARSE, 15.0),))

    fused = HybridRetriever(dense, sparse, dense_top_k=3, sparse_top_k=2, fused_top_k=3).retrieve(
        "FPT EPS 2025", limit=3
    )

    assert dense.results[0].chunk is first
    assert [item.chunk.chunk_id for item in fused] == ["relevant", "dense-first", "dense-second"]
