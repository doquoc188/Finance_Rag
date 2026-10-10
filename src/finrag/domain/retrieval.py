"""Retrieval provenance and score labels, without retrieval implementations."""

from dataclasses import dataclass
from enum import StrEnum
from math import isfinite

from .documents import DocumentChunk, _require_positive_integer


class RetrievalStrategy(StrEnum):
    DENSE = "dense"
    SPARSE = "sparse"
    HYBRID = "hybrid"
    GRAPH = "graph"


class ScoreType(StrEnum):
    COSINE = "cosine"
    BM25 = "bm25"
    RRF = "rrf"
    RERANKER = "reranker"


@dataclass(frozen=True, slots=True)
class RetrievalScore:
    """Raw score plus its semantics. Scores intentionally have no ordering operators.

    Even scores with the same type need matching model/index/query context before
    comparison. This value is not a calibrated probability or a normalized score.
    """

    value: float
    score_type: ScoreType

    def __post_init__(self) -> None:
        if isinstance(self.value, bool) or not isinstance(self.value, (int, float)):
            raise ValueError("score value must be a finite number")
        if not isfinite(self.value):
            raise ValueError("score value must be a finite number")
        if not isinstance(self.score_type, ScoreType):
            raise TypeError("score_type must be a ScoreType")


@dataclass(frozen=True, slots=True)
class RetrievedEvidence:
    chunk: DocumentChunk
    rank: int
    strategy: RetrievalStrategy
    score: RetrievalScore | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.chunk, DocumentChunk):
            raise TypeError("chunk must be a DocumentChunk")
        _require_positive_integer(self.rank, "rank")
        if not isinstance(self.strategy, RetrievalStrategy):
            raise TypeError("strategy must be a RetrievalStrategy")
        if self.score is not None and not isinstance(self.score, RetrievalScore):
            raise TypeError("score must pair a value with its ScoreType using RetrievalScore")
