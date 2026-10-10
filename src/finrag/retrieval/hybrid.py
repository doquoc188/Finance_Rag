"""Typed retriever contract and dense/sparse composition."""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from finrag.domain import RetrievedEvidence, ScoreType

from .reranking import Reranker
from .rrf import _positive_integer, reciprocal_rank_fusion


class Retriever(Protocol):
    """Small synchronous boundary shared by retrieval implementations."""

    def retrieve(self, query: str, *, limit: int) -> Sequence[RetrievedEvidence]: ...


@dataclass(frozen=True, slots=True)
class HybridRetriever:
    """Run dense and sparse retrieval independently, then fuse their ranks."""

    dense: Retriever
    sparse: Retriever
    dense_top_k: int = 20
    sparse_top_k: int = 20
    fused_top_k: int = 20
    rrf_k: int = 60
    reranker: Reranker | None = None

    def __post_init__(self) -> None:
        _positive_integer(self.dense_top_k, "dense_top_k")
        _positive_integer(self.sparse_top_k, "sparse_top_k")
        _positive_integer(self.fused_top_k, "fused_top_k")
        _positive_integer(self.rrf_k, "rrf_k")
        if self.fused_top_k > self.dense_top_k + self.sparse_top_k:
            raise ValueError("fused_top_k must not exceed the combined candidate limits")

    def retrieve(self, query: str, *, limit: int) -> tuple[RetrievedEvidence, ...]:
        _positive_integer(limit, "limit")
        dense = self.dense.retrieve(query, limit=self.dense_top_k)
        sparse = self.sparse.retrieve(query, limit=self.sparse_top_k)
        fused = reciprocal_rank_fusion(
            (dense, sparse),
            k=self.rrf_k,
            limit=self.fused_top_k,
        )
        final_limit = min(limit, self.fused_top_k)
        if self.reranker is None:
            return fused[:final_limit]

        reranked = tuple(self.reranker.rerank(query, fused, limit=final_limit))
        if len(reranked) > final_limit:
            raise ValueError("reranker returned more than the requested limit")
        candidates = {item.chunk.chunk_id: item for item in fused}
        seen: set[str] = set()
        for expected_rank, item in enumerate(reranked, start=1):
            chunk_id = item.chunk.chunk_id
            original = candidates.get(chunk_id)
            if original is None or item.chunk is not original.chunk:
                raise ValueError("reranker returned evidence outside the fused candidate set")
            if chunk_id in seen:
                raise ValueError("reranker returned a duplicate candidate")
            seen.add(chunk_id)
            if item.rank != expected_rank:
                raise ValueError("reranker must assign sequential ranks")
            if item.strategy is not original.strategy:
                raise ValueError("reranker must preserve the retrieval strategy")
            if item.score is None or item.score.score_type is not ScoreType.RERANKER:
                raise ValueError("reranker must return RERANKER scores")
        return reranked
