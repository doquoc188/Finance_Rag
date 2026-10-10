"""Stable identity and manifest contracts for incremental indexing."""

from .identity import (
    IDENTITY_VERSION,
    canonical_source,
    chunk_text_sha256,
    prepared_content_sha256,
    qdrant_point_id,
    stable_chunk_id,
    stable_document_id,
)
from .manifest import (
    MANIFEST_VERSION,
    PAYLOAD_SCHEMA_VERSION,
    VECTOR_DISTANCE,
    IndexManifest,
    IndexManifestError,
    IndexSignature,
    ManifestDocument,
    load_manifest,
    write_manifest,
)
from .planner import (
    DocumentIndexState,
    IndexingResult,
    IndexPlan,
    PlannedDocument,
    PreparedDocument,
    apply_index_plan,
    plan_incremental_index,
)

__all__ = [
    "IDENTITY_VERSION",
    "MANIFEST_VERSION",
    "PAYLOAD_SCHEMA_VERSION",
    "VECTOR_DISTANCE",
    "DocumentIndexState",
    "IndexingResult",
    "IndexManifest",
    "IndexManifestError",
    "IndexPlan",
    "IndexSignature",
    "ManifestDocument",
    "PlannedDocument",
    "PreparedDocument",
    "apply_index_plan",
    "canonical_source",
    "chunk_text_sha256",
    "load_manifest",
    "plan_incremental_index",
    "prepared_content_sha256",
    "qdrant_point_id",
    "stable_chunk_id",
    "stable_document_id",
    "write_manifest",
]
