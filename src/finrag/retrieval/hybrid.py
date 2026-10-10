"""Typed retriever contract and dense/sparse composition."""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from finrag.domain import RetrievedEvidence

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
        return reciprocal_rank_fusion(
            (dense, sparse),
            k=self.rrf_k,
            limit=min(limit, self.fused_top_k),
        )
