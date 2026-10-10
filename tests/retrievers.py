"""Reusable typed retrieval fakes for offline graph tests."""

import math
from collections.abc import Mapping, Sequence

from langchain_core.documents import Document

from finrag.domain import (
    DocumentChunk,
    MetadataValue,
    RetrievalScore,
    RetrievalStrategy,
    RetrievedEvidence,
    ScoreType,
    SourceReference,
)


def document_evidence(
    document: Document,
    chunk_id: str,
    *,
    rank: int = 1,
    strategy: RetrievalStrategy = RetrievalStrategy.HYBRID,
) -> RetrievedEvidence:
    metadata: dict[str, MetadataValue] = {
        key: value
        for key, value in document.metadata.items()
        if isinstance(key, str)
        and key.strip()
        and (
            isinstance(value, (str, int, bool, type(None)))
            or (isinstance(value, float) and math.isfinite(value))
        )
    }
    source_name = metadata.get("source")
    source = source_name if isinstance(source_name, str) and source_name.strip() else "test.pdf"
    page_value = metadata.get("page")
    page = (
        page_value
        if isinstance(page_value, int) and not isinstance(page_value, bool) and page_value > 0
        else None
    )
    document_id = f"document-{chunk_id}"
    reference = SourceReference(document_id, source, page=page)
    chunk = DocumentChunk(chunk_id, document_id, document.page_content, reference, metadata)
    score_type = {
        RetrievalStrategy.DENSE: ScoreType.COSINE,
        RetrievalStrategy.SPARSE: ScoreType.BM25,
        RetrievalStrategy.HYBRID: ScoreType.RRF,
    }.get(strategy)
    score = RetrievalScore(1.0, score_type) if score_type is not None else None
    return RetrievedEvidence(chunk, rank, strategy, score)


class RoutingRetriever:
    """Return configured local evidence by exact query and record every retrieval."""

    def __init__(
        self,
        responses: Mapping[str, Sequence[Document]],
        default: Sequence[Document] = (),
    ) -> None:
        self._responses = {
            query: tuple(
                document_evidence(document, f"response-{response_index}-{document_index}")
                for document_index, document in enumerate(documents)
            )
            for response_index, (query, documents) in enumerate(responses.items())
        }
        self._default = tuple(
            document_evidence(document, f"default-{index}")
            for index, document in enumerate(default)
        )
        self.queries: list[str] = []

    def retrieve(self, query: str, *, limit: int) -> tuple[RetrievedEvidence, ...]:
        self.queries.append(query)
        return self._responses.get(query, self._default)[:limit]


class StaticRetriever:
    def __init__(self, documents: Sequence[Document]) -> None:
        self._evidence = tuple(
            document_evidence(document, f"static-{index}")
            for index, document in enumerate(documents)
        )

    def retrieve(self, query: str, *, limit: int) -> tuple[RetrievedEvidence, ...]:
        return self._evidence[:limit]
