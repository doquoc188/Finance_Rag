"""Framework-independent document evidence and retrieval value objects."""

from .documents import DocumentChunk, MetadataValue, SourceReference
from .retrieval import RetrievalScore, RetrievalStrategy, RetrievedEvidence, ScoreType

__all__ = [
    "DocumentChunk",
    "MetadataValue",
    "RetrievedEvidence",
    "RetrievalScore",
    "RetrievalStrategy",
    "ScoreType",
    "SourceReference",
]
