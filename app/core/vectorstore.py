"""Application adapters joining LangChain documents, Qdrant, and clean retrieval types."""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Protocol, cast

from langchain_core.documents import Document
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointIdsList, PointStruct, VectorParams

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
from finrag.indexing import (
    IDENTITY_VERSION,
    PAYLOAD_SCHEMA_VERSION,
    VECTOR_DISTANCE,
    DocumentIndexState,
    IndexingResult,
    IndexManifest,
    IndexSignature,
    PreparedDocument,
    apply_index_plan,
    canonical_source,
    chunk_text_sha256,
    load_manifest,
    plan_incremental_index,
    prepared_content_sha256,
    qdrant_point_id,
    stable_chunk_id,
    stable_document_id,
    write_manifest,
)
from finrag.retrieval import BM25Retriever, HybridRetriever, Reranker


class EmbeddingModel(Protocol):
    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


class IndexCompatibilityError(RuntimeError):
    """Raised when Qdrant and the committed manifest cannot be reused safely."""


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
    """Own one incrementally indexed dense/BM25 corpus and expose fused evidence."""

    def __init__(
        self,
        documents: Sequence[Document] | None = None,
        *,
        retrieval_settings: RetrievalSettings | None = None,
        embeddings: EmbeddingModel | None = None,
        qdrant_client: QdrantClient | None = None,
        reranker: Reranker | None = None,
        manifest_path: Path | None = None,
        embedding_model_id: str | None = None,
        collection_name: str | None = None,
    ) -> None:
        self.settings = retrieval_settings or RetrievalSettings()
        self.documents: list[Document] = []
        self._embeddings = embeddings
        self._client = qdrant_client
        self.reranker = reranker
        self._hybrid: HybridRetriever | None = None
        self._collection_name = collection_name or legacy_settings.QDRANT_COLLECTION_NAME
        self._manifest_path = manifest_path
        self._embedding_identity_known = embedding_model_id is not None or embeddings is not None
        self._embedding_model_id = embedding_model_id or _embedding_identity(embeddings)
        self._manifest = load_manifest(manifest_path) if manifest_path is not None else None
        self._chunks_by_id: dict[str, DocumentChunk] = {}
        self._store_consistent = True
        if self._manifest is not None:
            self._restore_persisted_state()
        if documents:
            self.index_documents(documents)

    def index_documents(self, documents: Sequence[Document]) -> IndexingResult:
        """Merge a batch into dense and sparse indexes without removing absent documents."""

        incoming = _prepare_corpus(documents)
        plan = plan_incremental_index(self._manifest, incoming)
        counts = Counter(document.state for document in plan.documents)
        if not plan.documents_to_write:
            return self._result(counts, embedded=0, upserted=0, deleted=0)

        embeddings = self._resolve_embeddings()
        chunks_to_write = tuple(
            chunk for document in plan.documents_to_write for chunk in document.current.chunks
        )
        vectors = _embed_chunks(embeddings, chunks_to_write)
        dimension = len(vectors[0])
        signature = self._signature_for_dimension(dimension)
        client = self._client or QdrantClient(location=legacy_settings.QDRANT_LOCATION)
        self._ensure_compatible_collection(client, signature)

        try:
            client.upsert(
                collection_name=self._collection_name,
                points=[
                    PointStruct(
                        id=qdrant_point_id(chunk.chunk_id),
                        vector=vector,
                        payload=_chunk_payload(chunk),
                    )
                    for chunk, vector in zip(chunks_to_write, vectors, strict=True)
                ],
                wait=True,
            )
            if plan.stale_chunk_ids:
                client.delete(
                    collection_name=self._collection_name,
                    points_selector=PointIdsList(
                        points=[qdrant_point_id(chunk_id) for chunk_id in plan.stale_chunk_ids]
                    ),
                    wait=True,
                )
            next_manifest = apply_index_plan(self._manifest, signature, plan)
            if self._manifest_path is not None:
                write_manifest(self._manifest_path, next_manifest)
        except Exception:
            # A retry with the same source is safe, but retrieval must not use a state whose
            # Qdrant mutation and committed manifest may disagree.
            self._store_consistent = False
            raise

        next_chunks = dict(self._chunks_by_id)
        for document in plan.documents_to_write:
            if document.previous is not None:
                for chunk_id in document.previous.chunk_ids:
                    next_chunks.pop(chunk_id, None)
            next_chunks.update((chunk.chunk_id, chunk) for chunk in document.current.chunks)
        self._manifest = next_manifest
        self._chunks_by_id = next_chunks
        self._embeddings = embeddings
        self._client = client
        self._store_consistent = True
        self._install_hybrid(embeddings)
        print(
            f"[VectorStore] Indexed {len(chunks_to_write)} changed chunks; "
            f"corpus now has {len(next_chunks)} chunks."
        )
        return self._result(
            counts,
            embedded=len(chunks_to_write),
            upserted=len(chunks_to_write),
            deleted=len(plan.stale_chunk_ids),
        )

    def retrieve(self, query: str, *, limit: int) -> tuple[RetrievedEvidence, ...]:
        _require_positive_integer(limit, "limit")
        if not self._store_consistent:
            raise IndexCompatibilityError(
                "Index mutation did not commit; retry ingestion before retrieval"
            )
        if self._hybrid is None and self._manifest is not None:
            embeddings = self._resolve_embeddings()
            self._install_hybrid(embeddings)
        if self._hybrid is None:
            return _legacy_demo_evidence()[:limit]
        return self._hybrid.retrieve(query, limit=limit)

    def get_relevant_documents(self, query: str, top_k: int = 4) -> list[Document]:
        """Compatibility wrapper for callers still consuming LangChain documents."""

        return [evidence_to_document(item) for item in self.retrieve(query, limit=top_k)]

    def _restore_persisted_state(self) -> None:
        manifest = cast(IndexManifest, self._manifest)
        self._validate_configured_signature(manifest.signature)
        client = self._client or QdrantClient(location=legacy_settings.QDRANT_LOCATION)
        if not client.collection_exists(self._collection_name):
            raise IndexCompatibilityError(
                "Index manifest exists but its Qdrant collection is missing; "
                "perform an explicit full rebuild"
            )
        _validate_collection_schema(client, self._collection_name, manifest.signature)
        chunks_by_id = _load_payload_chunks(client, self._collection_name)
        expected_ids = set(manifest.chunk_ids)
        actual_ids = set(chunks_by_id)
        if actual_ids != expected_ids:
            raise IndexCompatibilityError(
                "Qdrant payloads do not exactly match the committed manifest; "
                "retry the interrupted ingestion or perform an explicit full rebuild"
            )
        for document in manifest.documents:
            for chunk_id in document.chunk_ids:
                chunk = chunks_by_id[chunk_id]
                if (
                    chunk.document_id != document.document_id
                    or chunk.source_reference.source != document.source
                ):
                    raise IndexCompatibilityError(
                        "Qdrant payload provenance does not match the committed manifest"
                    )
        self._client = client
        self._chunks_by_id = chunks_by_id
        self.documents = [_chunk_to_document(chunk) for chunk in self._ordered_chunks()]

    def _signature_for_dimension(self, dimension: int) -> IndexSignature:
        if self._manifest is not None:
            signature = self._manifest.signature
            self._validate_configured_signature(signature)
            if signature.vector_size != dimension:
                raise IndexCompatibilityError(
                    f"Embedding vector size changed from {signature.vector_size} to {dimension}"
                )
            return signature
        return IndexSignature(
            collection_name=self._collection_name,
            embedding_model=self._embedding_model_id,
            vector_size=dimension,
        )

    def _validate_configured_signature(self, signature: IndexSignature) -> None:
        differences = []
        if signature.collection_name != self._collection_name:
            differences.append("collection name")
        if self._embedding_identity_known and signature.embedding_model != self._embedding_model_id:
            differences.append("embedding model")
        if signature.identity_version != IDENTITY_VERSION:
            differences.append("identity format")
        if signature.payload_schema_version != PAYLOAD_SCHEMA_VERSION:
            differences.append("payload schema")
        if signature.vector_distance != VECTOR_DISTANCE:
            differences.append("vector distance")
        if differences:
            raise IndexCompatibilityError(
                "Index signature is incompatible with configured " + ", ".join(differences)
            )

    def _resolve_embeddings(self) -> EmbeddingModel:
        embeddings = self._embeddings or cast(EmbeddingModel, get_embeddings())
        if not self._embedding_identity_known:
            self._embedding_model_id = _embedding_identity(embeddings)
            self._embedding_identity_known = True
            if self._manifest is not None:
                self._validate_configured_signature(self._manifest.signature)
        self._embeddings = embeddings
        return embeddings

    def _ensure_compatible_collection(
        self,
        client: QdrantClient,
        signature: IndexSignature,
    ) -> None:
        if not client.collection_exists(self._collection_name):
            if self._manifest is not None and self._manifest.documents:
                raise IndexCompatibilityError(
                    "Committed manifest cannot be reused because its Qdrant collection is missing"
                )
            client.create_collection(
                collection_name=self._collection_name,
                vectors_config=VectorParams(
                    size=signature.vector_size,
                    distance=Distance.COSINE,
                ),
            )
            return
        _validate_collection_schema(client, self._collection_name, signature)
        if self._manifest is None:
            info = client.get_collection(self._collection_name)
            if info.points_count not in (None, 0):
                raise IndexCompatibilityError(
                    "Qdrant collection contains points but no compatible manifest; "
                    "perform an explicit migration or rebuild"
                )

    def _install_hybrid(self, embeddings: EmbeddingModel) -> None:
        if self._client is None or self._manifest is None:
            raise IndexCompatibilityError(
                "Cannot initialize retrieval without committed index state"
            )
        chunks = self._ordered_chunks()
        sparse = BM25Retriever(chunks)
        dense = QdrantDenseRetriever(
            self._client,
            self._collection_name,
            embeddings,
            chunks,
        )
        self._hybrid = HybridRetriever(
            dense,
            sparse,
            dense_top_k=self.settings.dense_top_k,
            sparse_top_k=self.settings.sparse_top_k,
            fused_top_k=self.settings.fused_top_k,
            rrf_k=self.settings.rrf_k,
            reranker=self.reranker,
        )
        self.documents = [_chunk_to_document(chunk) for chunk in chunks]

    def _ordered_chunks(self) -> tuple[DocumentChunk, ...]:
        if self._manifest is None:
            return ()
        return tuple(self._chunks_by_id[chunk_id] for chunk_id in self._manifest.chunk_ids)

    def _result(
        self,
        counts: Counter[DocumentIndexState],
        *,
        embedded: int,
        upserted: int,
        deleted: int,
    ) -> IndexingResult:
        total_documents = len(self._manifest.documents) if self._manifest is not None else 0
        total_chunks = len(self._manifest.chunk_ids) if self._manifest is not None else 0
        return IndexingResult(
            new_documents=counts[DocumentIndexState.NEW],
            unchanged_documents=counts[DocumentIndexState.UNCHANGED],
            changed_documents=counts[DocumentIndexState.CHANGED],
            embedded_chunks=embedded,
            upserted_chunks=upserted,
            deleted_chunks=deleted,
            total_documents=total_documents,
            total_chunks=total_chunks,
        )


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


def _prepare_corpus(documents: Sequence[Document]) -> tuple[PreparedDocument, ...]:
    grouped: dict[str, list[DocumentChunk]] = {}
    sources: dict[str, str] = {}
    duplicate_counts: defaultdict[str, Counter[tuple[int | None, str | None, str]]] = defaultdict(
        Counter
    )
    seen_chunk_ids: set[str] = set()

    for document in documents:
        metadata = _scalar_metadata(document.metadata)
        raw_source = _nonblank_string(metadata.get("source"))
        if raw_source is None:
            raise ValueError("Every indexed chunk requires nonblank source metadata")
        source = canonical_source(raw_source)
        company = _nonblank_string(metadata.get("company"))
        period = _nonblank_string(metadata.get("period")) or _nonblank_string(metadata.get("year"))
        document_id = _nonblank_string(metadata.get("document_id")) or stable_document_id(
            source,
            company=company,
            period=period,
        )
        if document_id in sources and sources[document_id] != source:
            raise ValueError(f"document_id {document_id!r} maps to conflicting sources")
        sources[document_id] = source
        page = _positive_page(metadata.get("page"))
        section = _nonblank_string(metadata.get("section"))
        occurrence_key = (page, section, chunk_text_sha256(document.page_content))
        duplicate_counts[document_id][occurrence_key] += 1
        occurrence = duplicate_counts[document_id][occurrence_key]
        chunk_id = _nonblank_string(metadata.get("chunk_id")) or stable_chunk_id(
            document_id,
            document.page_content,
            page=page,
            section=section,
            occurrence=occurrence,
        )
        if chunk_id in seen_chunk_ids:
            raise ValueError(f"Duplicate chunk_id in incoming corpus: {chunk_id!r}")
        seen_chunk_ids.add(chunk_id)
        metadata.update({"source": source, "document_id": document_id, "chunk_id": chunk_id})
        if page is not None:
            metadata["page"] = page
        else:
            metadata.pop("page", None)
        if section is None:
            metadata.pop("section", None)
        reference = SourceReference(document_id, source, page=page, section=section)
        chunk = DocumentChunk(
            chunk_id,
            document_id,
            document.page_content,
            reference,
            metadata,
        )
        grouped.setdefault(document_id, []).append(chunk)

    prepared = []
    for document_id, grouped_chunks in grouped.items():
        chunks = tuple(grouped_chunks)
        prepared.append(
            PreparedDocument(
                document_id,
                sources[document_id],
                prepared_content_sha256(chunks),
                chunks,
            )
        )
    return tuple(prepared)


def _embed_chunks(
    embeddings: EmbeddingModel,
    chunks: tuple[DocumentChunk, ...],
) -> list[list[float]]:
    vectors = [
        [float(value) for value in vector]
        for vector in embeddings.embed_documents([chunk.text for chunk in chunks])
    ]
    if len(vectors) != len(chunks) or not vectors or not vectors[0]:
        raise ValueError("Embedding output must contain one non-empty vector per chunk")
    dimension = len(vectors[0])
    if any(len(vector) != dimension for vector in vectors):
        raise ValueError("Embedding vectors must all have the same dimension")
    if any(not math.isfinite(value) for vector in vectors for value in vector):
        raise ValueError("Embedding vectors must contain only finite numbers")
    return vectors


def _chunk_payload(chunk: DocumentChunk) -> dict[str, object]:
    source = chunk.source_reference
    return {
        "chunk_id": chunk.chunk_id,
        "document_id": chunk.document_id,
        "metadata": dict(chunk.metadata),
        "page": source.page,
        "payload_schema_version": PAYLOAD_SCHEMA_VERSION,
        "section": source.section,
        "source": source.source,
        "text": chunk.text,
    }


def _load_payload_chunks(
    client: QdrantClient,
    collection_name: str,
) -> dict[str, DocumentChunk]:
    chunks: dict[str, DocumentChunk] = {}
    offset: Any = None
    while True:
        records, next_offset = client.scroll(
            collection_name=collection_name,
            limit=256,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        for record in records:
            chunk = _chunk_from_payload(record.payload)
            if chunk.chunk_id in chunks:
                raise IndexCompatibilityError("Qdrant contains duplicate stable chunk IDs")
            if str(record.id) != qdrant_point_id(chunk.chunk_id):
                raise IndexCompatibilityError(
                    "Qdrant point ID does not match its deterministic chunk UUID"
                )
            chunks[chunk.chunk_id] = chunk
        if next_offset is None:
            return chunks
        offset = next_offset


def _chunk_from_payload(payload: Mapping[str, object] | None) -> DocumentChunk:
    if payload is None:
        raise IndexCompatibilityError("Qdrant point is missing its reconstruction payload")
    if payload.get("payload_schema_version") != PAYLOAD_SCHEMA_VERSION:
        raise IndexCompatibilityError("Qdrant payload schema is incompatible")
    chunk_id = _payload_string(payload, "chunk_id")
    document_id = _payload_string(payload, "document_id")
    source = _payload_string(payload, "source")
    text = _payload_string(payload, "text")
    page_value = payload.get("page")
    if page_value is not None and (
        isinstance(page_value, bool) or not isinstance(page_value, int) or page_value < 1
    ):
        raise IndexCompatibilityError("Qdrant payload page must be positive or null")
    section_value = payload.get("section")
    if section_value is not None and (
        not isinstance(section_value, str) or not section_value.strip()
    ):
        raise IndexCompatibilityError("Qdrant payload section must be nonblank or null")
    metadata_value = payload.get("metadata")
    if not isinstance(metadata_value, dict):
        raise IndexCompatibilityError("Qdrant payload metadata must be an object")
    metadata = _strict_payload_metadata(metadata_value)
    if (
        metadata.get("chunk_id") != chunk_id
        or metadata.get("document_id") != document_id
        or metadata.get("source") != source
    ):
        raise IndexCompatibilityError("Qdrant payload metadata identity is inconsistent")
    reference = SourceReference(
        document_id,
        source,
        page=page_value,
        section=section_value,
    )
    return DocumentChunk(chunk_id, document_id, text, reference, metadata)


def _strict_payload_metadata(metadata: dict[object, object]) -> dict[str, MetadataValue]:
    result: dict[str, MetadataValue] = {}
    for key, value in metadata.items():
        if not isinstance(key, str) or not key.strip():
            raise IndexCompatibilityError("Qdrant payload metadata keys must be nonblank strings")
        if not isinstance(value, (str, int, float, bool, type(None))):
            raise IndexCompatibilityError("Qdrant payload metadata values must be scalars")
        if isinstance(value, float) and not math.isfinite(value):
            raise IndexCompatibilityError("Qdrant payload metadata numbers must be finite")
        result[key] = value
    return result


def _payload_string(payload: Mapping[str, object], name: str) -> str:
    value = payload.get(name)
    if not isinstance(value, str) or not value.strip():
        raise IndexCompatibilityError(f"Qdrant payload {name} must be a nonblank string")
    return value


def _validate_collection_schema(
    client: QdrantClient,
    collection_name: str,
    signature: IndexSignature,
) -> None:
    vectors = client.get_collection(collection_name).config.params.vectors
    if not isinstance(vectors, VectorParams):
        raise IndexCompatibilityError("Named or missing Qdrant vectors are not supported")
    distance = vectors.distance.value.casefold()
    if vectors.size != signature.vector_size or distance != signature.vector_distance:
        raise IndexCompatibilityError(
            "Qdrant vector size or distance is incompatible with the index signature"
        )


def _embedding_identity(embeddings: EmbeddingModel | None) -> str:
    if embeddings is None:
        if legacy_settings.OPENAI_API_KEY and not legacy_settings.OPENAI_API_KEY.startswith("mock"):
            return "openai:text-embedding-3-small"
        return f"huggingface:{legacy_settings.EMBEDDING_MODEL}"
    model_name = getattr(embeddings, "model_name", None) or getattr(embeddings, "model", None)
    module_name = type(embeddings).__module__.casefold()
    if isinstance(model_name, str) and "langchain_openai" in module_name:
        return f"openai:{model_name}"
    if isinstance(model_name, str) and "huggingface" in module_name:
        return f"huggingface:{model_name}"
    class_name = f"{type(embeddings).__module__}.{type(embeddings).__qualname__}"
    return f"{class_name}:{model_name}" if isinstance(model_name, str) else class_name


def _chunk_to_document(chunk: DocumentChunk) -> Document:
    return Document(page_content=chunk.text, metadata=dict(chunk.metadata))


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
