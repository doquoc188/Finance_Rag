"""Project-level errors for the optional GraphRAG boundary."""


class GraphRAGError(RuntimeError):
    """Base error for explicit GraphRAG operations."""


class GraphRAGArtifactError(GraphRAGError):
    """The curated graph artifacts are missing or malformed."""


class GraphRAGConfigurationError(GraphRAGError):
    """The live GraphRAG workspace is not configured safely."""


class GraphRAGDependencyError(GraphRAGError):
    """The optional Microsoft GraphRAG dependency is unavailable."""


class GraphRAGQueryError(GraphRAGError):
    """Microsoft GraphRAG could not complete a query."""
