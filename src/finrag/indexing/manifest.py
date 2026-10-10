"""Strict, deterministic JSON manifest for committed index state."""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

MANIFEST_VERSION = 1
PAYLOAD_SCHEMA_VERSION = 1
VECTOR_DISTANCE = "cosine"


class IndexManifestError(ValueError):
    """Raised when persisted index metadata is malformed or incompatible."""


@dataclass(frozen=True, slots=True)
class IndexSignature:
    collection_name: str
    embedding_model: str
    vector_size: int
    vector_distance: str = VECTOR_DISTANCE
    identity_version: int = 1
    payload_schema_version: int = PAYLOAD_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _require_nonblank(self.collection_name, "collection_name")
        _require_nonblank(self.embedding_model, "embedding_model")
        _require_positive_integer(self.vector_size, "vector_size")
        if self.vector_distance != VECTOR_DISTANCE:
            raise ValueError(f"vector_distance must be {VECTOR_DISTANCE!r}")
        _require_positive_integer(self.identity_version, "identity_version")
        _require_positive_integer(self.payload_schema_version, "payload_schema_version")


@dataclass(frozen=True, slots=True)
class ManifestDocument:
    document_id: str
    source: str
    content_sha256: str
    chunk_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        _require_nonblank(self.document_id, "document_id")
        _require_nonblank(self.source, "source")
        if not _is_sha256(self.content_sha256):
            raise ValueError("content_sha256 must be a lowercase SHA-256 digest")
        if not self.chunk_ids:
            raise ValueError("manifest document must contain at least one chunk_id")
        if len(set(self.chunk_ids)) != len(self.chunk_ids):
            raise ValueError("manifest document chunk_ids must be unique")
        for chunk_id in self.chunk_ids:
            _require_nonblank(chunk_id, "chunk_id")


@dataclass(frozen=True, slots=True)
class IndexManifest:
    signature: IndexSignature
    documents: tuple[ManifestDocument, ...]
    manifest_version: int = MANIFEST_VERSION

    def __post_init__(self) -> None:
        if self.manifest_version != MANIFEST_VERSION:
            raise ValueError(f"manifest_version must be {MANIFEST_VERSION}")
        document_ids = [document.document_id for document in self.documents]
        if len(set(document_ids)) != len(document_ids):
            raise ValueError("manifest document_ids must be unique")
        chunk_ids = [chunk_id for document in self.documents for chunk_id in document.chunk_ids]
        if len(set(chunk_ids)) != len(chunk_ids):
            raise ValueError("manifest chunk_ids must be globally unique")

    @property
    def documents_by_id(self) -> dict[str, ManifestDocument]:
        return {document.document_id: document for document in self.documents}

    @property
    def chunk_ids(self) -> tuple[str, ...]:
        return tuple(chunk_id for document in self.documents for chunk_id in document.chunk_ids)


def load_manifest(path: Path) -> IndexManifest | None:
    """Load a manifest, returning None only when it does not exist."""

    try:
        raw = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return None
    except OSError as exc:
        raise IndexManifestError(f"Could not read index manifest: {path}") from exc
    try:
        value = json.loads(raw)
        return _manifest_from_json(value)
    except (json.JSONDecodeError, TypeError, ValueError) as exc:
        raise IndexManifestError(f"Invalid index manifest: {path}") from exc


def write_manifest(path: Path, manifest: IndexManifest) -> None:
    """Atomically replace a manifest with deterministic UTF-8 JSON."""

    path.parent.mkdir(parents=True, exist_ok=True)
    rendered = (
        json.dumps(
            _manifest_to_json(manifest),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            newline="\n",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            temporary.write(rendered)
            temporary.flush()
            os.fsync(temporary.fileno())
        os.replace(temporary_path, path)
    except OSError as exc:
        raise IndexManifestError(f"Could not atomically write index manifest: {path}") from exc
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def _manifest_to_json(manifest: IndexManifest) -> dict[str, object]:
    signature = manifest.signature
    return {
        "documents": [
            {
                "chunk_ids": list(document.chunk_ids),
                "content_sha256": document.content_sha256,
                "document_id": document.document_id,
                "source": document.source,
            }
            for document in sorted(manifest.documents, key=lambda item: item.document_id)
        ],
        "index_signature": {
            "collection_name": signature.collection_name,
            "embedding_model": signature.embedding_model,
            "identity_version": signature.identity_version,
            "payload_schema_version": signature.payload_schema_version,
            "vector_distance": signature.vector_distance,
            "vector_size": signature.vector_size,
        },
        "manifest_version": manifest.manifest_version,
    }


def _manifest_from_json(value: object) -> IndexManifest:
    root = _object(value, "manifest")
    _require_keys(root, {"documents", "index_signature", "manifest_version"}, "manifest")
    version = _integer(root["manifest_version"], "manifest_version")
    if version != MANIFEST_VERSION:
        raise IndexManifestError(
            f"Unsupported manifest_version {version}; expected {MANIFEST_VERSION}"
        )
    signature_value = _object(root["index_signature"], "index_signature")
    signature_keys = {
        "collection_name",
        "embedding_model",
        "identity_version",
        "payload_schema_version",
        "vector_distance",
        "vector_size",
    }
    _require_keys(signature_value, signature_keys, "index_signature")
    signature = IndexSignature(
        collection_name=_string(signature_value["collection_name"], "collection_name"),
        embedding_model=_string(signature_value["embedding_model"], "embedding_model"),
        vector_size=_integer(signature_value["vector_size"], "vector_size"),
        vector_distance=_string(signature_value["vector_distance"], "vector_distance"),
        identity_version=_integer(signature_value["identity_version"], "identity_version"),
        payload_schema_version=_integer(
            signature_value["payload_schema_version"], "payload_schema_version"
        ),
    )
    documents_value = root["documents"]
    if not isinstance(documents_value, list):
        raise IndexManifestError("documents must be a JSON array")
    documents = []
    for position, raw_document in enumerate(documents_value):
        document = _object(raw_document, f"documents[{position}]")
        _require_keys(
            document,
            {"chunk_ids", "content_sha256", "document_id", "source"},
            f"documents[{position}]",
        )
        raw_chunk_ids = document["chunk_ids"]
        if not isinstance(raw_chunk_ids, list):
            raise IndexManifestError(f"documents[{position}].chunk_ids must be a JSON array")
        documents.append(
            ManifestDocument(
                document_id=_string(document["document_id"], "document_id"),
                source=_string(document["source"], "source"),
                content_sha256=_string(document["content_sha256"], "content_sha256"),
                chunk_ids=tuple(_string(item, "chunk_id") for item in raw_chunk_ids),
            )
        )
    return IndexManifest(signature, tuple(documents), version)


def _object(value: object, name: str) -> dict[str, Any]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise IndexManifestError(f"{name} must be a JSON object")
    return value


def _require_keys(value: dict[str, Any], expected: set[str], name: str) -> None:
    if set(value) != expected:
        raise IndexManifestError(f"{name} must contain exactly: {', '.join(sorted(expected))}")


def _string(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise IndexManifestError(f"{name} must be a nonblank string")
    return value


def _integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise IndexManifestError(f"{name} must be an integer")
    return value


def _require_nonblank(value: str, name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonblank string")


def _require_positive_integer(value: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and value == value.lower()
        and all(character in "0123456789abcdef" for character in value)
    )
