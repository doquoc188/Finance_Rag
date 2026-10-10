"""Framework-independent contracts for explicit graph queries."""

from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol


def _require_nonblank(value: str, field_name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a nonblank string")


class GraphQueryMode(StrEnum):
    """Microsoft GraphRAG query modes supported by the vertical slice."""

    LOCAL = "local"
    GLOBAL = "global"


@dataclass(frozen=True, slots=True)
class GraphProvenance:
    """Stable fixture records linked to source documents used in query context."""

    context_type: str
    record_id: str
    document_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        _require_nonblank(self.context_type, "context_type")
        _require_nonblank(self.record_id, "record_id")
        if not self.document_ids:
            raise ValueError("document_ids must not be empty")
        for document_id in self.document_ids:
            _require_nonblank(document_id, "document_id")


@dataclass(frozen=True, slots=True)
class GraphQueryResult:
    """Graph answer translated away from Microsoft-native objects."""

    answer: str
    mode: GraphQueryMode
    provenance: tuple[GraphProvenance, ...] = ()

    def __post_init__(self) -> None:
        _require_nonblank(self.answer, "answer")
        if not isinstance(self.mode, GraphQueryMode):
            raise TypeError("mode must be a GraphQueryMode")
        if any(not isinstance(item, GraphProvenance) for item in self.provenance):
            raise TypeError("provenance must contain GraphProvenance values")


class GraphQueryBackend(Protocol):
    """Explicit asynchronous graph-query boundary."""

    async def query(self, question: str, *, mode: GraphQueryMode) -> GraphQueryResult:
        """Run one caller-selected graph query without automatic routing."""
        ...
