"""Deterministic document, chunk, point, and content identities."""

from __future__ import annotations

import hashlib
import json
import unicodedata
import uuid
from collections.abc import Mapping, Sequence
from pathlib import PurePosixPath, PureWindowsPath

from finrag.domain import DocumentChunk, MetadataValue

IDENTITY_VERSION = 1
QDRANT_POINT_NAMESPACE = uuid.UUID("ba3d5508-9577-4e26-9037-f1ed7078e349")
_IDENTITY_METADATA_KEYS = frozenset({"chunk_id", "document_id", "source"})


def canonical_source(source: str) -> str:
    """Return a stable logical source value, rejecting ambiguous placeholders."""

    value = _canonical_string(source, "source").replace("\\", "/")
    if value.casefold() in {"unknown", "unknown source", "n/a", "none"}:
        raise ValueError("source must identify a logical document, not a placeholder")
    if PurePosixPath(value).is_absolute() or PureWindowsPath(value).is_absolute():
        raise ValueError("source must be logical identity metadata, not an absolute path")
    return value


def stable_document_id(
    source: str,
    *,
    company: str | None = None,
    period: str | None = None,
) -> str:
    """Hash logical source identity without using content, position, or random values."""

    payload = {
        "company": _optional_canonical_string(company),
        "identity_version": IDENTITY_VERSION,
        "period": _optional_canonical_string(period),
        "source": canonical_source(source),
    }
    return f"document-{_sha256_json(payload)}"


def stable_chunk_id(
    document_id: str,
    text: str,
    *,
    page: int | None = None,
    section: str | None = None,
    occurrence: int = 1,
) -> str:
    """Hash a chunk's document-local structure and duplicate occurrence."""

    document_id = _canonical_string(document_id, "document_id")
    if page is not None and (isinstance(page, bool) or not isinstance(page, int) or page < 1):
        raise ValueError("page must be a positive integer or None")
    if isinstance(occurrence, bool) or not isinstance(occurrence, int) or occurrence < 1:
        raise ValueError("occurrence must be a positive integer")
    normalized_text = _canonical_text(text)
    payload = {
        "document_id": document_id,
        "identity_version": IDENTITY_VERSION,
        "occurrence": occurrence,
        "page": page,
        "section": _optional_canonical_string(section),
        "text_sha256": hashlib.sha256(normalized_text.encode("utf-8")).hexdigest(),
    }
    return f"chunk-{_sha256_json(payload)}"


def prepared_content_sha256(chunks: Sequence[DocumentChunk]) -> str:
    """Fingerprint ordered, prepared content without transient identity fields."""

    if not chunks:
        raise ValueError("prepared document must contain at least one chunk")
    document_id = chunks[0].document_id
    if any(chunk.document_id != document_id for chunk in chunks):
        raise ValueError("prepared content chunks must belong to one document")
    records = []
    for chunk in chunks:
        source = chunk.source_reference
        records.append(
            {
                "metadata": _content_metadata(chunk.metadata),
                "page": source.page,
                "section": source.section,
                "text": _canonical_text(chunk.text),
            }
        )
    return _sha256_json(records)


def qdrant_point_id(chunk_id: str) -> str:
    """Map a stable chunk ID to Qdrant's deterministic UUID point identifier."""

    return str(uuid.uuid5(QDRANT_POINT_NAMESPACE, _canonical_string(chunk_id, "chunk_id")))


def chunk_text_sha256(text: str) -> str:
    """Return the normalized text digest used to count duplicate occurrences."""

    return hashlib.sha256(_canonical_text(text).encode("utf-8")).hexdigest()


def _content_metadata(metadata: Mapping[str, MetadataValue]) -> dict[str, MetadataValue]:
    return {key: metadata[key] for key in sorted(metadata) if key not in _IDENTITY_METADATA_KEYS}


def _canonical_text(value: str) -> str:
    value = _canonical_string(value, "text")
    return value.replace("\r\n", "\n").replace("\r", "\n")


def _canonical_string(value: str, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a nonblank string")
    return unicodedata.normalize("NFKC", value).strip()


def _optional_canonical_string(value: str | None) -> str | None:
    return None if value is None else _canonical_string(value, "identity value")


def _sha256_json(value: object) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
