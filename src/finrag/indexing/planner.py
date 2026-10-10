"""Pure incremental indexing plan and committed-result values."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from finrag.domain import DocumentChunk

from .manifest import IndexManifest, IndexSignature, ManifestDocument


class DocumentIndexState(StrEnum):
    NEW = "new"
    UNCHANGED = "unchanged"
    CHANGED = "changed"


@dataclass(frozen=True, slots=True)
class PreparedDocument:
    document_id: str
    source: str
    content_sha256: str
    chunks: tuple[DocumentChunk, ...]

    def __post_init__(self) -> None:
        if not self.document_id.strip() or not self.source.strip():
            raise ValueError("prepared document identity must be nonblank")
        if len(self.content_sha256) != 64:
            raise ValueError("content_sha256 must be a SHA-256 digest")
        if not self.chunks:
            raise ValueError("prepared document must contain chunks")
        if any(chunk.document_id != self.document_id for chunk in self.chunks):
            raise ValueError("prepared chunks must match document_id")
        chunk_ids = [chunk.chunk_id for chunk in self.chunks]
        if len(set(chunk_ids)) != len(chunk_ids):
            raise ValueError("prepared document chunk IDs must be unique")

    @property
    def chunk_ids(self) -> tuple[str, ...]:
        return tuple(chunk.chunk_id for chunk in self.chunks)

    def to_manifest_document(self) -> ManifestDocument:
        return ManifestDocument(
            self.document_id,
            self.source,
            self.content_sha256,
            self.chunk_ids,
        )


@dataclass(frozen=True, slots=True)
class PlannedDocument:
    state: DocumentIndexState
    current: PreparedDocument
    previous: ManifestDocument | None

    @property
    def stale_chunk_ids(self) -> tuple[str, ...]:
        if self.previous is None:
            return ()
        current_ids = set(self.current.chunk_ids)
        return tuple(
            chunk_id for chunk_id in self.previous.chunk_ids if chunk_id not in current_ids
        )


@dataclass(frozen=True, slots=True)
class IndexPlan:
    documents: tuple[PlannedDocument, ...]

    @property
    def documents_to_write(self) -> tuple[PlannedDocument, ...]:
        return tuple(
            document
            for document in self.documents
            if document.state is not DocumentIndexState.UNCHANGED
        )

    @property
    def stale_chunk_ids(self) -> tuple[str, ...]:
        return tuple(
            chunk_id for document in self.documents for chunk_id in document.stale_chunk_ids
        )


@dataclass(frozen=True, slots=True)
class IndexingResult:
    new_documents: int
    unchanged_documents: int
    changed_documents: int
    embedded_chunks: int
    upserted_chunks: int
    deleted_chunks: int
    total_documents: int
    total_chunks: int


def plan_incremental_index(
    manifest: IndexManifest | None,
    incoming: tuple[PreparedDocument, ...],
) -> IndexPlan:
    """Classify an incoming batch without mutating storage or manifest state."""

    document_ids = [document.document_id for document in incoming]
    if len(set(document_ids)) != len(document_ids):
        raise ValueError("incoming document IDs must be unique")
    chunk_ids = [chunk_id for document in incoming for chunk_id in document.chunk_ids]
    if len(set(chunk_ids)) != len(chunk_ids):
        raise ValueError("incoming chunk IDs must be globally unique")
    previous_by_id = manifest.documents_by_id if manifest is not None else {}
    incoming_document_ids = set(document_ids)
    retained_chunk_ids = {
        chunk_id
        for document_id, document in previous_by_id.items()
        if document_id not in incoming_document_ids
        for chunk_id in document.chunk_ids
    }
    collisions = retained_chunk_ids.intersection(chunk_ids)
    if collisions:
        raise ValueError("incoming chunk IDs collide with retained manifest documents")
    planned = []
    for document in incoming:
        previous = previous_by_id.get(document.document_id)
        if previous is None:
            state = DocumentIndexState.NEW
        elif (
            previous.content_sha256 == document.content_sha256
            and previous.source == document.source
            and previous.chunk_ids == document.chunk_ids
        ):
            state = DocumentIndexState.UNCHANGED
        else:
            state = DocumentIndexState.CHANGED
        planned.append(PlannedDocument(state, document, previous))
    return IndexPlan(tuple(planned))


def apply_index_plan(
    manifest: IndexManifest | None,
    signature: IndexSignature,
    plan: IndexPlan,
) -> IndexManifest:
    """Merge the plan into committed state while retaining absent documents."""

    documents = manifest.documents_by_id if manifest is not None else {}
    for planned in plan.documents:
        documents[planned.current.document_id] = planned.current.to_manifest_document()
    return IndexManifest(
        signature,
        tuple(sorted(documents.values(), key=lambda document: document.document_id)),
    )
