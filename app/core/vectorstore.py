"""Application adapters joining LangChain documents, Qdrant, and clean retrieval types."""

from __future__ import annotations

import hashlib
import math
import uuid
from collections.abc import Sequence
from typing import Protocol, cast

from langchain_core.documents import Document
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from app.config import settings as legacy_settings
from app.core.llm import get_embeddings
from finrag.config import RetrievalSettings
from finrag.domain import (
    DocumentChunk,
    MetadataValue,
    RetrievalScore,
    RetrievalStrategy,
    RetrievedEvidence,
    ScoreType,
    SourceReference,
)
from finrag.retrieval import BM25Retriever, HybridRetriever, Reranker


class EmbeddingModel(Protocol):
    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


class QdrantDenseRetriever:
    """Adapt an explicitly cosine-configured Qdrant collection to typed evidence."""

    def __init__(
        self,
        client: QdrantClient,
        collection_name: str,
        embeddings: EmbeddingModel,
        chunks: Sequence[DocumentChunk],
    ) -> None:
        self._client = client
        self._collection_name = collection_name
        self._embeddings = embeddings
        self._chunks = {chunk.chunk_id: chunk for chunk in chunks}
        if len(self._chunks) != len(chunks):
            raise ValueError("Dense corpus chunk IDs must be unique")

    def retrieve(self, query: str, *, limit: int) -> tuple[RetrievedEvidence, ...]:
        _require_positive_integer(limit, "limit")
        query_vector = [float(value) for value in self._embeddings.embed_query(query)]
        response = self._client.query_points(
            collection_name=self._collection_name,
            query=query_vector,
            limit=limit,
            with_payload=True,
        )
        evidence: list[RetrievedEvidence] = []
        for rank, point in enumerate(response.points, start=1):
            payload = point.payload or {}
            chunk_id = payload.get("chunk_id")
            if not isinstance(chunk_id, str) or chunk_id not in self._chunks:
                raise ValueError("Qdrant result is missing a known stable chunk_id")
            evidence.append(
                RetrievedEvidence(
                    chunk=self._chunks[chunk_id],
                    rank=rank,
                    strategy=RetrievalStrategy.DENSE,
                    score=RetrievalScore(float(point.score), ScoreType.COSINE),
                )
            )
        return tuple(evidence)


class HybridFinancialRetriever:
    """Own one dense/BM25 corpus and expose fused evidence to the application."""

    def __init__(
        self,
        documents: Sequence[Document] | None = None,
        *,
        retrieval_settings: RetrievalSettings | None = None,
        embeddings: EmbeddingModel | None = None,
        qdrant_client: QdrantClient | None = None,
        reranker: Reranker | None = None,
    ) -> None:
        self.settings = retrieval_settings or RetrievalSettings()
        self.documents: list[Document] = []
        self._embeddings = embeddings
        self._client = qdrant_client
        self.reranker = reranker
        self._hybrid: HybridRetriever | None = None
        if documents:
            self.index_documents(documents)

    def index_documents(self, documents: Sequence[Document]) -> None:
        """Rebuild the existing dense and sparse indexes from one prepared corpus.

        Collection recreation intentionally preserves the legacy `force_recreate=True`
        lifecycle. Phase 1.4 owns incremental and failure-atomic indexing.
        """

        prepared_documents, chunks = _prepare_corpus(documents)
        if not chunks:
            self.documents = []
            self._hybrid = None
            return

        embeddings = self._embeddings or cast(EmbeddingModel, get_embeddings())
        vectors = [
            [float(value) for value in vector]
            for vector in embeddings.embed_documents([chunk.text for chunk in chunks])
        ]
        if len(vectors) != len(chunks) or not vectors or not vectors[0]:
            raise ValueError("Embedding output must contain one non-empty vector per chunk")
        dimension = len(vectors[0])
        if any(len(vector) != dimension for vector in vectors):
            raise ValueError("Embedding vectors must all have the same dimension")

        sparse = BM25Retriever(chunks)
        client = self._client or QdrantClient(location=legacy_settings.QDRANT_LOCATION)
        collection_name = legacy_settings.QDRANT_COLLECTION_NAME
        if client.collection_exists(collection_name):
            client.delete_collection(collection_name)
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=dimension, distance=Distance.COSINE),
        )
        client.upsert(
            collection_name=collection_name,
            points=[
                PointStruct(
                    id=str(uuid.uuid5(uuid.NAMESPACE_URL, f"finrag:{chunk.chunk_id}")),
                    vector=vector,
                    payload={"chunk_id": chunk.chunk_id},
                )
                for chunk, vector in zip(chunks, vectors, strict=True)
            ],
        )

        dense = QdrantDenseRetriever(client, collection_name, embeddings, chunks)
        hybrid = HybridRetriever(
            dense,
            sparse,
            dense_top_k=self.settings.dense_top_k,
            sparse_top_k=self.settings.sparse_top_k,
            fused_top_k=self.settings.fused_top_k,
            rrf_k=self.settings.rrf_k,
            reranker=self.reranker,
        )
        self._embeddings = embeddings
        self._client = client
        self.documents = prepared_documents
        self._hybrid = hybrid
        print(f"[VectorStore] Indexed {len(chunks)} document chunks.")

    def retrieve(self, query: str, *, limit: int) -> tuple[RetrievedEvidence, ...]:
        _require_positive_integer(limit, "limit")
        if self._hybrid is None:
            return _legacy_demo_evidence()[:limit]
        return self._hybrid.retrieve(query, limit=limit)

    def get_relevant_documents(self, query: str, top_k: int = 4) -> list[Document]:
        """Compatibility wrapper for callers still consuming LangChain documents."""

        return [evidence_to_document(item) for item in self.retrieve(query, limit=top_k)]


def evidence_to_document(evidence: RetrievedEvidence) -> Document:
    """Restore the graph's established LangChain document shape with provenance."""

    metadata = dict(evidence.chunk.metadata)
    source = evidence.chunk.source_reference
    metadata.update(
        {
            "chunk_id": evidence.chunk.chunk_id,
            "document_id": evidence.chunk.document_id,
            "source": source.source,
            "retrieval_rank": evidence.rank,
            "retrieval_strategy": evidence.strategy.value,
        }
    )
    if source.page is not None:
        metadata["page"] = source.page
    if source.section is not None:
        metadata["section"] = source.section
    if evidence.score is not None:
        metadata["retrieval_score"] = evidence.score.value
        metadata["retrieval_score_type"] = evidence.score.score_type.value
    return Document(page_content=evidence.chunk.text, metadata=metadata)


def _prepare_corpus(
    documents: Sequence[Document],
) -> tuple[list[Document], tuple[DocumentChunk, ...]]:
    prepared: list[Document] = []
    chunks: list[DocumentChunk] = []
    seen_ids: set[str] = set()

    for position, document in enumerate(documents):
        metadata = _scalar_metadata(document.metadata)
        source = _nonblank_string(metadata.get("source")) or "unknown"
        company = _nonblank_string(metadata.get("company")) or ""
        year = _nonblank_string(metadata.get("year")) or ""
        document_id = _nonblank_string(metadata.get("document_id")) or _stable_id(
            "document", source, company, year
        )
        page = _positive_page(metadata.get("page"))
        section = _nonblank_string(metadata.get("section"))
        chunk_id = _nonblank_string(metadata.get("chunk_id")) or _stable_id(
            "chunk", document_id, str(page or ""), section or "", str(position)
        )
        if chunk_id in seen_ids:
            raise ValueError(f"Duplicate chunk_id in retrieval corpus: {chunk_id!r}")
        seen_ids.add(chunk_id)
        metadata.update({"source": source, "document_id": document_id, "chunk_id": chunk_id})
        if page is not None:
            metadata["page"] = page
        else:
            metadata.pop("page", None)
        if section is None:
            metadata.pop("section", None)
        source_reference = SourceReference(document_id, source, page=page, section=section)
        chunk = DocumentChunk(
            chunk_id,
            document_id,
            document.page_content,
            source_reference,
            metadata,
        )
        chunks.append(chunk)
        prepared.append(Document(page_content=document.page_content, metadata=dict(metadata)))

    return prepared, tuple(chunks)


def _scalar_metadata(metadata: dict[str, object]) -> dict[str, MetadataValue]:
    result: dict[str, MetadataValue] = {}
    for key, value in metadata.items():
        if not isinstance(key, str) or not key.strip():
            continue
        if isinstance(value, (str, int, bool, type(None))):
            result[key] = value
        elif isinstance(value, float) and math.isfinite(value):
            result[key] = value
    return result


def _nonblank_string(value: MetadataValue) -> str | None:
    return value.strip() if isinstance(value, str) and value.strip() else None


def _positive_page(value: MetadataValue) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) and value > 0 else None


def _stable_id(kind: str, *parts: str) -> str:
    digest = hashlib.sha256("\x1f".join(parts).encode()).hexdigest()
    return f"{kind}-{digest}"


def _require_positive_integer(value: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")


def _legacy_demo_evidence() -> tuple[RetrievedEvidence, ...]:
    document_id = "document-demo-fpt-2023"
    source = SourceReference(
        document_id,
        "FPT_Annual_Report_2023.pdf",
        page=14,
    )
    metadata: dict[str, MetadataValue] = {
        "source": source.source,
        "page": 14,
        "company": "FPT Corporation",
        "year": "2023",
    }
    chunk = DocumentChunk(
        "chunk-demo-fpt-2023-page-14",
        document_id,
        "[Mock Financial Report] Năm 2023, FPT đạt doanh thu 52.618 tỷ đồng "
        "(tăng 19,6%), Lợi nhuận trước thuế 9.203 tỷ đồng (tăng 20,1%). "
        "Khối Công nghệ đóng góp 45% tổng doanh thu.",
        source,
        metadata,
    )
    return (RetrievedEvidence(chunk, 1, RetrievalStrategy.HYBRID),)
