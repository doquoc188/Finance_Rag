"""Optional GraphRAG contracts and lazy Microsoft adapter."""

from .artifacts import GraphArtifacts, load_graph_artifacts
from .contracts import GraphProvenance, GraphQueryBackend, GraphQueryMode, GraphQueryResult
from .errors import (
    GraphRAGArtifactError,
    GraphRAGConfigurationError,
    GraphRAGDependencyError,
    GraphRAGError,
    GraphRAGQueryError,
)
from .microsoft import MicrosoftGraphRAGBackend

__all__ = [
    "GraphArtifacts",
    "GraphProvenance",
    "GraphQueryBackend",
    "GraphQueryMode",
    "GraphQueryResult",
    "GraphRAGArtifactError",
    "GraphRAGConfigurationError",
    "GraphRAGDependencyError",
    "GraphRAGError",
    "GraphRAGQueryError",
    "MicrosoftGraphRAGBackend",
    "load_graph_artifacts",
]
