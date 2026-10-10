"""Immutable source evidence without framework-specific document objects."""

from collections.abc import Mapping
from dataclasses import dataclass, field
from math import isfinite
from types import MappingProxyType

type MetadataValue = str | int | float | bool | None


def _require_nonblank(value: str, field_name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a nonblank string")


def _require_positive_integer(value: int, field_name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{field_name} must be a positive integer")


@dataclass(frozen=True, slots=True)
class SourceReference:
    document_id: str
    source: str
    page: int | None = None
    section: str | None = None

    def __post_init__(self) -> None:
        _require_nonblank(self.document_id, "document_id")
        _require_nonblank(self.source, "source")
        if self.page is not None:
            _require_positive_integer(self.page, "page")
        if self.section is not None:
            _require_nonblank(self.section, "section")


@dataclass(frozen=True, slots=True)
class DocumentChunk:
    chunk_id: str
    document_id: str
    text: str
    source_reference: SourceReference
    # Metadata participates in equality, but not hashing: mappings are unhashable.
    metadata: Mapping[str, MetadataValue] = field(default_factory=dict, hash=False)

    def __post_init__(self) -> None:
        _require_nonblank(self.chunk_id, "chunk_id")
        _require_nonblank(self.document_id, "document_id")
        _require_nonblank(self.text, "text")
        if not isinstance(self.source_reference, SourceReference):
            raise TypeError("source_reference must be a SourceReference")
        if self.source_reference.document_id != self.document_id:
            raise ValueError("source_reference.document_id must match document_id")
        metadata = dict(self.metadata)
        for key, value in metadata.items():
            _require_nonblank(key, "metadata key")
            if not isinstance(value, (str, int, float, bool, type(None))):
                raise TypeError("metadata values must be immutable scalars")
            if isinstance(value, float) and not isfinite(value):
                raise ValueError("metadata numbers must be finite")
        object.__setattr__(self, "metadata", MappingProxyType(metadata))
