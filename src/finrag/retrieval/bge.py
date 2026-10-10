"""Optional local BGE cross-encoder reranker adapter."""

from collections.abc import Callable, Iterable, Sequence
from importlib import import_module
from typing import Protocol, cast

from finrag.domain import RetrievedEvidence

from .reranking import evidence_from_reranker_scores
from .rrf import _positive_integer

DEFAULT_BGE_RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"


class CrossEncoderModel(Protocol):
    """The small part of sentence-transformers used by this adapter."""

    def predict(
        self,
        inputs: Sequence[tuple[str, str]],
        *,
        show_progress_bar: bool = False,
    ) -> Iterable[float]: ...


type CrossEncoderLoader = Callable[[str], CrossEncoderModel]


def _load_cross_encoder(model_name: str) -> CrossEncoderModel:
    module = import_module("sentence_transformers")
    constructor = cast(CrossEncoderLoader, getattr(module, "CrossEncoder"))
    return constructor(model_name)


class BgeCrossEncoderReranker:
    """Score fused candidates with a local sentence-transformers CrossEncoder."""

    def __init__(
        self,
        model_name: str = DEFAULT_BGE_RERANKER_MODEL,
        *,
        model_loader: CrossEncoderLoader = _load_cross_encoder,
    ) -> None:
        if not model_name.strip():
            raise ValueError("model_name must not be blank")
        self.model_name = model_name.strip()
        self._model = model_loader(self.model_name)

    def rerank(
        self,
        query: str,
        candidates: Sequence[RetrievedEvidence],
        *,
        limit: int,
    ) -> tuple[RetrievedEvidence, ...]:
        _positive_integer(limit, "limit")
        if not candidates:
            return ()

        inputs = [(query, candidate.chunk.text) for candidate in candidates]
        raw_scores = self._model.predict(inputs, show_progress_bar=False)
        try:
            scores = tuple(float(score) for score in raw_scores)
        except TypeError as error:
            raise ValueError("reranker scores must be an iterable of numbers") from error
        return evidence_from_reranker_scores(candidates, scores, limit=limit)
