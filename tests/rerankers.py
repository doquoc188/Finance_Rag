"""Deterministic test-only rerankers."""

from collections.abc import Mapping, Sequence

from finrag.domain import RetrievalScore, RetrievedEvidence, ScoreType


class ScoreReranker:
    def __init__(self, scores: Mapping[str, float]) -> None:
        self.scores = dict(scores)
        self.calls: list[tuple[str, tuple[str, ...], int]] = []

    def rerank(
        self,
        query: str,
        candidates: Sequence[RetrievedEvidence],
        *,
        limit: int,
    ) -> tuple[RetrievedEvidence, ...]:
        self.calls.append((query, tuple(item.chunk.chunk_id for item in candidates), limit))
        ordered = sorted(
            enumerate(candidates),
            key=lambda item: (-self.scores[item[1].chunk.chunk_id], item[0]),
        )[:limit]
        return tuple(
            RetrievedEvidence(
                candidate.chunk,
                rank,
                candidate.strategy,
                RetrievalScore(self.scores[candidate.chunk.chunk_id], ScoreType.RERANKER),
            )
            for rank, (_, candidate) in enumerate(ordered, start=1)
        )


class FailingReranker:
    def __init__(self, error: RuntimeError) -> None:
        self.error = error

    def rerank(
        self,
        query: str,
        candidates: Sequence[RetrievedEvidence],
        *,
        limit: int,
    ) -> tuple[RetrievedEvidence, ...]:
        raise self.error
