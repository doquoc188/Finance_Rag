"""Typed reranking contract and score-to-evidence conversion."""

from collections.abc import Sequence
from typing import Protocol

from finrag.domain import RetrievalScore, RetrievedEvidence, ScoreType

from .rrf import _positive_integer


class Reranker(Protocol):
    """Reorder a fused candidate set and return at most ``limit`` items."""

    def rerank(
        self,
        query: str,
        candidates: Sequence[RetrievedEvidence],
        *,
        limit: int,
    ) -> Sequence[RetrievedEvidence]: ...


def evidence_from_reranker_scores(
    candidates: Sequence[RetrievedEvidence],
    scores: Sequence[float],
    *,
    limit: int,
) -> tuple[RetrievedEvidence, ...]:
    """Create stable reranked evidence without changing chunks or strategy."""

    _positive_integer(limit, "limit")
    if len(scores) != len(candidates):
        raise ValueError("reranker must return one score per candidate")

    scored = sorted(
        enumerate(zip(candidates, scores, strict=True)),
        key=lambda item: (-float(item[1][1]), item[0]),
    )
    return tuple(
        RetrievedEvidence(
            chunk=candidate.chunk,
            rank=rank,
            strategy=candidate.strategy,
            score=RetrievalScore(float(score), ScoreType.RERANKER),
        )
        for rank, (_, (candidate, score)) in enumerate(scored[:limit], start=1)
    )
