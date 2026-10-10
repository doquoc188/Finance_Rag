"""Offline tests for the optional sentence-transformers BGE adapter."""

from collections.abc import Sequence

import pytest

from finrag.domain import RetrievalStrategy, RetrievedEvidence, ScoreType
from finrag.retrieval.bge import BgeCrossEncoderReranker
from tests.test_finrag_retrieval import evidence, make_chunk


class FakeCrossEncoder:
    def __init__(self, scores: Sequence[float]) -> None:
        self.scores = tuple(scores)
        self.calls: list[tuple[list[tuple[str, str]], bool]] = []

    def predict(
        self,
        inputs: Sequence[tuple[str, str]],
        *,
        show_progress_bar: bool = False,
    ) -> Sequence[float]:
        self.calls.append((list(inputs), show_progress_bar))
        return self.scores


def candidates() -> tuple[RetrievedEvidence, ...]:
    return (
        evidence(make_chunk("A", "EPS 2024"), 1, RetrievalStrategy.HYBRID, 0.03),
        evidence(make_chunk("B", "EPS 2025"), 2, RetrievalStrategy.HYBRID, 0.02),
    )


def test_bge_adapter_scores_query_text_pairs_without_real_model() -> None:
    model = FakeCrossEncoder((0.1, 0.9))
    loaded: list[str] = []

    def load(model_name: str) -> FakeCrossEncoder:
        loaded.append(model_name)
        return model

    reranker = BgeCrossEncoderReranker("BAAI/bge-reranker-v2-m3", model_loader=load)
    results = reranker.rerank("FPT EPS 2025", candidates(), limit=2)

    assert loaded == ["BAAI/bge-reranker-v2-m3"]
    assert model.calls == [([("FPT EPS 2025", "EPS 2024"), ("FPT EPS 2025", "EPS 2025")], False)]
    assert [item.chunk.chunk_id for item in results] == ["B", "A"]
    assert all(item.score and item.score.score_type is ScoreType.RERANKER for item in results)


def test_bge_adapter_rejects_missing_scores() -> None:
    reranker = BgeCrossEncoderReranker(
        "BAAI/bge-reranker-v2-m3", model_loader=lambda _: FakeCrossEncoder((0.5,))
    )

    with pytest.raises(ValueError, match="one score per candidate"):
        reranker.rerank("query", candidates(), limit=2)


def test_bge_adapter_does_not_call_model_for_empty_candidates() -> None:
    model = FakeCrossEncoder(())
    reranker = BgeCrossEncoderReranker("BAAI/bge-reranker-v2-m3", model_loader=lambda _: model)

    assert reranker.rerank("query", (), limit=2) == ()
    assert model.calls == []
