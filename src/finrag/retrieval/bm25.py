"""In-memory BM25 retrieval over an explicit immutable chunk corpus."""

import re
from collections.abc import Callable, Sequence
from importlib import import_module
from typing import Protocol, cast

from finrag.domain import (
    DocumentChunk,
    RetrievalScore,
    RetrievalStrategy,
    RetrievedEvidence,
    ScoreType,
)

from .rrf import _positive_integer

Tokenize = Callable[[str], list[str]]


class _BM25Index(Protocol):
    def get_scores(self, query: list[str]) -> Sequence[float]: ...


def _build_index(corpus: list[list[str]]) -> _BM25Index:
    constructor = cast(
        Callable[[list[list[str]]], _BM25Index],
        getattr(import_module("rank_bm25"), "BM25Okapi"),
    )
    return constructor(corpus)


def tokenize_words(text: str) -> list[str]:
    """Tokenize case-insensitively while retaining Unicode letters and digits."""

    return re.findall(r"\w+", text.casefold())


class BM25Retriever:
    """Rank the supplied corpus independently with BM25."""

    def __init__(
        self,
        chunks: Sequence[DocumentChunk],
        *,
        tokenize: Tokenize = tokenize_words,
    ) -> None:
        self._chunks = tuple(chunks)
        self._tokenize = tokenize
        chunk_ids = [chunk.chunk_id for chunk in self._chunks]
        if len(chunk_ids) != len(set(chunk_ids)):
            raise ValueError("BM25 corpus chunk IDs must be unique")
        corpus = [tokenize(chunk.text) for chunk in self._chunks]
        self._index = _build_index(corpus) if corpus else None

    def retrieve(self, query: str, *, limit: int) -> tuple[RetrievedEvidence, ...]:
        _positive_integer(limit, "limit")
        if self._index is None:
            return ()

        scores = self._index.get_scores(self._tokenize(query))
        ordered = sorted(range(len(self._chunks)), key=lambda index: (-float(scores[index]), index))
        return tuple(
            RetrievedEvidence(
                chunk=self._chunks[index],
                rank=rank,
                strategy=RetrievalStrategy.SPARSE,
                score=RetrievalScore(float(scores[index]), ScoreType.BM25),
            )
            for rank, index in enumerate(ordered[:limit], start=1)
        )
