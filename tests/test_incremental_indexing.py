"""Offline Qdrant and API regressions for non-destructive incremental indexing."""

from collections.abc import Sequence
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
from fastapi.testclient import TestClient
from langchain_core.documents import Document
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from app.core.pdf_parser import FinancialPDFParser
from app.core.vectorstore import HybridFinancialRetriever, IndexCompatibilityError
from app.main import create_app
from app.runtime import GraphRuntime
from finrag.config import Settings
from finrag.indexing import load_manifest, qdrant_point_id


class RecordingEmbeddings:
    model_name = "recording-v1"

    def __init__(self) -> None:
        self.document_calls: list[list[str]] = []
        self.query_calls: list[str] = []
        self.fail_documents = False

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        self.document_calls.append(list(texts))
        if self.fail_documents:
            raise RuntimeError("controlled embedding failure")
        return [self._vector(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        self.query_calls.append(text)
        return self._vector(text)

    @staticmethod
    def _vector(text: str) -> list[float]:
        lowered = text.casefold()
        if "alpha" in lowered:
            return [1.0, 0.0]
        if "beta" in lowered:
            return [0.0, 1.0]
        return [0.7, 0.7]


class RecordingQdrantClient(QdrantClient):
    def __init__(self) -> None:
        super().__init__(location=":memory:")
        self.upsert_sizes: list[int] = []
        self.delete_sizes: list[int] = []
        self.delete_collection_calls = 0
        self.fail_upsert = False

    def upsert(  # type: ignore[override]
        self, collection_name: str, points: Any, **kwargs: Any
    ) -> Any:
        point_list = list(points)
        self.upsert_sizes.append(len(point_list))
        if self.fail_upsert:
            raise RuntimeError("controlled upsert failure")
        return super().upsert(collection_name, point_list, **kwargs)

    def delete(  # type: ignore[override]
        self, collection_name: str, points_selector: Any, **kwargs: Any
    ) -> Any:
        self.delete_sizes.append(len(points_selector.points))
        return super().delete(collection_name, points_selector, **kwargs)

    def delete_collection(  # type: ignore[override]
        self, collection_name: str, **kwargs: Any
    ) -> bool:
        self.delete_collection_calls += 1
        raise AssertionError(f"collection deletion is forbidden: {collection_name}")


def document(
    source: str,
    text: str,
    *,
    page: int = 1,
    document_id: str | None = None,
    chunk_id: str | None = None,
) -> Document:
    metadata: dict[str, object] = {
        "source": source,
        "page": page,
        "company": source.removesuffix(".pdf").upper(),
        "year": "2025",
    }
    if document_id is not None:
        metadata["document_id"] = document_id
    if chunk_id is not None:
        metadata["chunk_id"] = chunk_id
    return Document(page_content=text, metadata=metadata)


def retriever(
    tmp_path: Path,
    *,
    client: RecordingQdrantClient | None = None,
    embeddings: RecordingEmbeddings | None = None,
) -> tuple[HybridFinancialRetriever, RecordingQdrantClient, RecordingEmbeddings, Path]:
    active_client = client or RecordingQdrantClient()
    active_embeddings = embeddings or RecordingEmbeddings()
    manifest_path = tmp_path / "index-manifest.json"
    return (
        HybridFinancialRetriever(
            embeddings=active_embeddings,
            qdrant_client=active_client,
            manifest_path=manifest_path,
            embedding_model_id="test:recording-v1",
            collection_name="incremental-test",
        ),
        active_client,
        active_embeddings,
        manifest_path,
    )


def payloads(client: QdrantClient) -> list[dict[str, Any]]:
    records, offset = client.scroll(
        "incremental-test", limit=100, with_payload=True, with_vectors=False
    )
    assert offset is None
    return [dict(record.payload or {}) for record in records]


def assert_sources_are_consistent(
    active: HybridFinancialRetriever,
    client: QdrantClient,
    expected_sources: set[str],
) -> None:
    dense_sources = {str(payload["source"]) for payload in payloads(client)}
    sparse_sources = {str(item.metadata["source"]) for item in active.documents}
    assert dense_sources == sparse_sources == expected_sources


def test_unchanged_ingestion_is_a_zero_write_no_op(tmp_path: Path) -> None:
    active, client, embeddings, manifest_path = retriever(tmp_path)
    corpus = [document("a.pdf", "alpha revenue")]
    first = active.index_documents(corpus)
    manifest_bytes = manifest_path.read_bytes()
    embeddings.document_calls.clear()
    client.upsert_sizes.clear()
    client.delete_sizes.clear()

    second = active.index_documents(corpus)

    assert first.new_documents == first.embedded_chunks == first.upserted_chunks == 1
    assert second.unchanged_documents == 1
    assert second.embedded_chunks == second.upserted_chunks == second.deleted_chunks == 0
    assert second.total_documents == second.total_chunks == 1
    assert embeddings.document_calls == []
    assert client.upsert_sizes == client.delete_sizes == []
    assert manifest_path.read_bytes() == manifest_bytes
    assert client.delete_collection_calls == 0


def test_unrelated_insertion_and_reorder_keep_existing_ids_stable(tmp_path: Path) -> None:
    active, client, embeddings, manifest_path = retriever(tmp_path)
    alpha = document("a.pdf", "alpha revenue")
    beta = document("b.pdf", "beta margin")
    active.index_documents([alpha])
    alpha_manifest = load_manifest(manifest_path)
    assert alpha_manifest is not None
    alpha_ids = alpha_manifest.documents[0].chunk_ids
    embeddings.document_calls.clear()

    result = active.index_documents([beta, alpha])
    updated = load_manifest(manifest_path)

    assert result.new_documents == result.unchanged_documents == 1
    assert embeddings.document_calls == [["beta margin"]]
    assert updated is not None
    alpha_record = next(item for item in updated.documents if item.source == "a.pdf")
    assert alpha_record.chunk_ids == alpha_ids
    assert_sources_are_consistent(active, client, {"a.pdf", "b.pdf"})


def test_inserting_an_unrelated_chunk_keeps_existing_chunk_id(tmp_path: Path) -> None:
    active, _, _, manifest_path = retriever(tmp_path)
    existing = document("a.pdf", "alpha revenue", page=2)
    active.index_documents([existing])
    original = load_manifest(manifest_path)
    assert original is not None
    original_id = original.documents[0].chunk_ids[0]

    result = active.index_documents([document("a.pdf", "new cover disclosure", page=1), existing])
    updated = load_manifest(manifest_path)

    assert result.changed_documents == 1
    assert result.deleted_chunks == 0
    assert updated is not None
    assert original_id in updated.documents[0].chunk_ids


def test_equal_text_across_documents_and_duplicates_within_one_document_are_distinct(
    tmp_path: Path,
) -> None:
    active, client, _, _ = retriever(tmp_path)
    documents = [
        document("a.pdf", "same financial text"),
        document("a.pdf", "same financial text"),
        document("b.pdf", "same financial text"),
    ]

    result = active.index_documents(documents)
    chunk_ids = [str(payload["chunk_id"]) for payload in payloads(client)]

    assert result.total_documents == 2
    assert result.total_chunks == 3
    assert len(set(chunk_ids)) == 3
    assert {str(payload["document_id"]) for payload in payloads(client)} == {
        str(item.metadata["document_id"]) for item in active.documents
    }


def test_explicit_ids_are_preserved_and_point_ids_are_deterministic(tmp_path: Path) -> None:
    active, client, _, _ = retriever(tmp_path)
    active.index_documents(
        [
            document(
                "explicit.pdf",
                "alpha disclosure",
                document_id="document-explicit",
                chunk_id="chunk-explicit",
            )
        ]
    )
    records, _ = client.scroll("incremental-test", limit=10, with_payload=True)

    assert len(records) == 1
    assert records[0].payload is not None
    assert records[0].payload["document_id"] == "document-explicit"
    assert records[0].payload["chunk_id"] == "chunk-explicit"
    assert str(records[0].id) == qdrant_point_id("chunk-explicit")


def test_changed_document_replaces_only_its_points_and_keeps_other_corpus(
    tmp_path: Path,
) -> None:
    active, client, embeddings, _ = retriever(tmp_path)
    alpha_old = document("a.pdf", "alpha old revenue")
    beta = document("b.pdf", "beta margin")
    active.index_documents([alpha_old, beta])
    beta_payload = next(payload for payload in payloads(client) if payload["source"] == "b.pdf")
    embeddings.document_calls.clear()
    client.upsert_sizes.clear()
    client.delete_sizes.clear()

    result = active.index_documents([document("a.pdf", "alpha new revenue")])

    assert result.changed_documents == 1
    assert result.embedded_chunks == result.upserted_chunks == result.deleted_chunks == 1
    assert embeddings.document_calls == [["alpha new revenue"]]
    assert client.upsert_sizes == client.delete_sizes == [1]
    assert (
        next(payload for payload in payloads(client) if payload["source"] == "b.pdf")
        == beta_payload
    )
    assert_sources_are_consistent(active, client, {"a.pdf", "b.pdf"})
    assert client.delete_collection_calls == 0


def test_restart_reconstructs_complete_hybrid_corpus_from_safe_payloads(tmp_path: Path) -> None:
    first, client, _, manifest_path = retriever(tmp_path)
    corpus = [document("a.pdf", "alpha revenue"), document("b.pdf", "beta margin")]
    first.index_documents(corpus)
    restarted_embeddings = RecordingEmbeddings()

    restarted = HybridFinancialRetriever(
        embeddings=restarted_embeddings,
        qdrant_client=client,
        manifest_path=manifest_path,
        embedding_model_id="test:recording-v1",
        collection_name="incremental-test",
    )
    results = restarted.get_relevant_documents("alpha", top_k=2)
    unchanged = restarted.index_documents([corpus[0]])

    assert {item.metadata["source"] for item in restarted.documents} == {"a.pdf", "b.pdf"}
    assert {item.metadata["source"] for item in results} == {"a.pdf", "b.pdf"}
    assert restarted_embeddings.document_calls == []
    assert restarted_embeddings.query_calls == ["alpha"]
    assert unchanged.unchanged_documents == 1
    assert unchanged.total_documents == unchanged.total_chunks == 2


def test_embedding_failure_does_not_mutate_qdrant_or_manifest(tmp_path: Path) -> None:
    active, client, embeddings, manifest_path = retriever(tmp_path)
    active.index_documents([document("a.pdf", "alpha old revenue")])
    before_manifest = manifest_path.read_bytes()
    before_payloads = payloads(client)
    client.upsert_sizes.clear()
    client.delete_sizes.clear()
    embeddings.fail_documents = True

    with pytest.raises(RuntimeError, match="embedding failure"):
        active.index_documents([document("a.pdf", "alpha new revenue")])

    assert payloads(client) == before_payloads
    assert manifest_path.read_bytes() == before_manifest
    assert client.upsert_sizes == client.delete_sizes == []


def test_upsert_failure_keeps_old_points_and_manifest_and_blocks_stale_retrieval(
    tmp_path: Path,
) -> None:
    active, client, _, manifest_path = retriever(tmp_path)
    active.index_documents([document("a.pdf", "alpha old revenue")])
    before_manifest = manifest_path.read_bytes()
    before_payloads = payloads(client)
    client.upsert_sizes.clear()
    client.delete_sizes.clear()
    client.fail_upsert = True

    with pytest.raises(RuntimeError, match="upsert failure"):
        active.index_documents([document("a.pdf", "alpha new revenue")])

    assert payloads(client) == before_payloads
    assert manifest_path.read_bytes() == before_manifest
    assert client.upsert_sizes == [1]
    assert client.delete_sizes == []
    with pytest.raises(IndexCompatibilityError, match="retry ingestion"):
        active.retrieve("alpha", limit=1)


def test_incompatible_existing_collection_is_refused_without_deletion(tmp_path: Path) -> None:
    client = RecordingQdrantClient()
    client.create_collection(
        "incremental-test",
        vectors_config=VectorParams(size=3, distance=Distance.EUCLID),
    )
    active, _, _, manifest_path = retriever(tmp_path, client=client)

    with pytest.raises(IndexCompatibilityError, match="size or distance"):
        active.index_documents([document("a.pdf", "alpha revenue")])

    assert client.collection_exists("incremental-test")
    assert client.delete_collection_calls == 0
    assert not manifest_path.exists()


def test_nonempty_collection_without_manifest_is_refused(tmp_path: Path) -> None:
    client = RecordingQdrantClient()
    client.create_collection(
        "incremental-test",
        vectors_config=VectorParams(size=2, distance=Distance.COSINE),
    )
    client.upsert(
        "incremental-test",
        [PointStruct(id=1, vector=[1.0, 0.0], payload={"legacy": True})],
    )
    client.upsert_sizes.clear()
    active, _, _, _ = retriever(tmp_path, client=client)

    with pytest.raises(IndexCompatibilityError, match="no compatible manifest"):
        active.index_documents([document("a.pdf", "alpha revenue")])

    assert client.delete_collection_calls == 0


def test_manifest_embedding_signature_change_is_refused_on_restart(tmp_path: Path) -> None:
    first, client, _, manifest_path = retriever(tmp_path)
    first.index_documents([document("a.pdf", "alpha revenue")])

    with pytest.raises(IndexCompatibilityError, match="embedding model"):
        HybridFinancialRetriever(
            embeddings=RecordingEmbeddings(),
            qdrant_client=client,
            manifest_path=manifest_path,
            embedding_model_id="test:different-model",
            collection_name="incremental-test",
        )


def test_restart_refuses_manifest_collection_payload_mismatch(tmp_path: Path) -> None:
    first, client, _, manifest_path = retriever(tmp_path)
    first.index_documents([document("a.pdf", "alpha revenue")])
    extra_payload = payloads(client)[0]
    extra_payload.update(
        {
            "chunk_id": "chunk-extra",
            "document_id": "document-extra",
            "source": "extra.pdf",
            "text": "extra disclosure",
        }
    )
    extra_payload["metadata"] = {
        "chunk_id": "chunk-extra",
        "document_id": "document-extra",
        "source": "extra.pdf",
    }
    client.upsert(
        "incremental-test",
        [
            PointStruct(
                id=qdrant_point_id("chunk-extra"),
                vector=[0.7, 0.7],
                payload=extra_payload,
            )
        ],
    )

    with pytest.raises(IndexCompatibilityError, match="exactly match"):
        HybridFinancialRetriever(
            embeddings=RecordingEmbeddings(),
            qdrant_client=client,
            manifest_path=manifest_path,
            embedding_model_id="test:recording-v1",
            collection_name="incremental-test",
        )


def test_restart_refuses_manifest_without_collection(tmp_path: Path) -> None:
    first, _, _, manifest_path = retriever(tmp_path)
    first.index_documents([document("a.pdf", "alpha revenue")])

    with pytest.raises(IndexCompatibilityError, match="collection is missing"):
        HybridFinancialRetriever(
            embeddings=RecordingEmbeddings(),
            qdrant_client=RecordingQdrantClient(),
            manifest_path=manifest_path,
            embedding_model_id="test:recording-v1",
            collection_name="incremental-test",
        )


def test_two_separate_api_ingestions_preserve_both_documents(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    active, client, _, _ = retriever(tmp_path)

    def parse_upload(
        self: FinancialPDFParser,
        file_path: str,
        company_name: str = "FPT Corporation",
        year: str = "2023",
    ) -> Sequence[Document]:
        del self
        source = Path(file_path).name
        token = "alpha" if source == "a.pdf" else "beta"
        return [
            Document(
                page_content=f"{token} revenue {year}",
                metadata={"source": source, "company": company_name, "year": year, "page": 1},
            )
        ]

    monkeypatch.setattr(FinancialPDFParser, "parse_pdf", parse_upload)
    monkeypatch.chdir(tmp_path)
    runtime = SimpleNamespace(retriever=active, settings=Settings())
    application = create_app(lambda: cast(GraphRuntime, runtime))

    with TestClient(application) as api:
        first = api.post(
            "/api/v1/ingest",
            files={"file": ("a.pdf", b"first", "application/pdf")},
            data={"company_name": "A", "year": "2025"},
        )
        second = api.post(
            "/api/v1/ingest",
            files={"file": ("b.pdf", b"second", "application/pdf")},
            data={"company_name": "B", "year": "2025"},
        )

    assert first.status_code == second.status_code == 200
    assert first.json()["chunks_created"] == second.json()["chunks_created"] == 1
    assert_sources_are_consistent(active, client, {"a.pdf", "b.pdf"})
    assert {item.metadata["source"] for item in active.get_relevant_documents("alpha", 2)} == {
        "a.pdf",
        "b.pdf",
    }
