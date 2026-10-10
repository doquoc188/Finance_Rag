"""Offline application-boundary tests for Qdrant dense score semantics."""

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from app.core.vectorstore import QdrantDenseRetriever
from finrag.domain import (
    DocumentChunk,
    RetrievalStrategy,
    ScoreType,
    SourceReference,
)


class QueryEmbeddings:
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        raise AssertionError("Dense querying must not rebuild embeddings")

    def embed_query(self, text: str) -> list[float]:
        return [1.0, 0.0]


def test_qdrant_dense_adapter_reports_explicit_cosine_scores() -> None:
    client = QdrantClient(location=":memory:")
    client.create_collection(
        collection_name="test",
        vectors_config=VectorParams(size=2, distance=Distance.COSINE),
    )
    source = SourceReference("report", "report.pdf", page=1)
    matching = DocumentChunk("matching", "report", "FPT EPS", source)
    other = DocumentChunk("other", "report", "other evidence", source)
    client.upsert(
        collection_name="test",
        points=[
            PointStruct(id=1, vector=[1.0, 0.0], payload={"chunk_id": matching.chunk_id}),
            PointStruct(id=2, vector=[0.0, 1.0], payload={"chunk_id": other.chunk_id}),
        ],
    )
    retriever = QdrantDenseRetriever(client, "test", QueryEmbeddings(), (matching, other))

    results = retriever.retrieve("FPT EPS", limit=2)

    assert results[0].chunk is matching
    assert results[0].strategy is RetrievalStrategy.DENSE
    assert results[0].score and results[0].score.score_type is ScoreType.COSINE
    assert results[0].score.value == 1.0
