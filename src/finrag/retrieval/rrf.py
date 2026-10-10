"""Deterministic Reciprocal Rank Fusion for typed retrieval evidence."""

from collections.abc import Sequence

from finrag.domain import (
    RetrievalScore,
    RetrievalStrategy,
    RetrievedEvidence,
    ScoreType,
)


def _positive_integer(value: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")


def reciprocal_rank_fusion(
    rankings: Sequence[Sequence[RetrievedEvidence]],
    *,
    k: int = 60,
    limit: int | None = None,
) -> tuple[RetrievedEvidence, ...]:
    """Fuse rankings by stable chunk ID, using first appearance to break ties.

    Only 1-based ranks within each supplied list contribute to the fused score. Raw
    dense or sparse scores are intentionally ignored. A duplicate ID within one source
    list contributes once, at its first position.
    """

    _positive_integer(k, "k")
    if limit is not None:
        _positive_integer(limit, "limit")

    scores: dict[str, float] = {}
    chunks: dict[str, RetrievedEvidence] = {}
    first_seen: dict[str, int] = {}
    next_seen = 0

    for ranking in rankings:
        seen_in_ranking: set[str] = set()
        for rank, evidence in enumerate(ranking, start=1):
            chunk_id = evidence.chunk.chunk_id
            if chunk_id in seen_in_ranking:
                continue
            seen_in_ranking.add(chunk_id)

            existing = chunks.get(chunk_id)
            if existing is not None and existing.chunk != evidence.chunk:
                raise ValueError(f"chunk_id {chunk_id!r} identifies conflicting chunks")
            if existing is None:
                chunks[chunk_id] = evidence
                first_seen[chunk_id] = next_seen
                next_seen += 1
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + rank)

    ordered_ids = sorted(scores, key=lambda item: (-scores[item], first_seen[item]))
    if limit is not None:
        ordered_ids = ordered_ids[:limit]

    return tuple(
        RetrievedEvidence(
            chunk=chunks[chunk_id].chunk,
            rank=rank,
            strategy=RetrievalStrategy.HYBRID,
            score=RetrievalScore(scores[chunk_id], ScoreType.RRF),
        )
        for rank, chunk_id in enumerate(ordered_ids, start=1)
    )
