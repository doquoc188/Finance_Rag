"""Pure stable-identity, manifest, and incremental-plan contracts."""

import json
import uuid
from pathlib import Path

import pytest

from finrag.domain import DocumentChunk, SourceReference
from finrag.indexing import (
    DocumentIndexState,
    IndexManifest,
    IndexManifestError,
    IndexSignature,
    PreparedDocument,
    apply_index_plan,
    load_manifest,
    plan_incremental_index,
    prepared_content_sha256,
    qdrant_point_id,
    stable_chunk_id,
    stable_document_id,
    write_manifest,
)


def make_chunk(
    chunk_id: str,
    document_id: str,
    text: str,
    *,
    source: str = "report.pdf",
    page: int = 1,
) -> DocumentChunk:
    reference = SourceReference(document_id, source, page=page)
    return DocumentChunk(
        chunk_id,
        document_id,
        text,
        reference,
        {
            "chunk_id": chunk_id,
            "document_id": document_id,
            "source": source,
            "page": page,
            "year": "2025",
        },
    )


def make_prepared(document_id: str, text: str, *, source: str = "report.pdf") -> PreparedDocument:
    chunk = make_chunk(f"chunk-{document_id}-{text}", document_id, text, source=source)
    chunks = (chunk,)
    return PreparedDocument(document_id, source, prepared_content_sha256(chunks), chunks)


def test_generated_document_ids_use_logical_source_identity() -> None:
    first = stable_document_id("reports\\FPT-2025.pdf", company="FPT", period="2025")
    assert first == stable_document_id("reports/FPT-2025.pdf", company="FPT", period="2025")
    assert first != stable_document_id("reports/VNM-2025.pdf", company="FPT", period="2025")
    assert first != stable_document_id("reports/FPT-2025.pdf", company="FPT", period="2024")
    with pytest.raises(ValueError, match="logical document"):
        stable_document_id("unknown")
    with pytest.raises(ValueError, match="absolute path"):
        stable_document_id("C:\\temp\\upload-123.pdf")


def test_generated_chunk_and_qdrant_ids_obey_document_local_invariants() -> None:
    first = stable_chunk_id("document-a", "same text", page=3, occurrence=1)
    assert first == stable_chunk_id("document-a", "same text", page=3, occurrence=1)
    assert first != stable_chunk_id("document-b", "same text", page=3, occurrence=1)
    assert first != stable_chunk_id("document-a", "same text", page=3, occurrence=2)
    assert first != stable_chunk_id("document-a", "same text", page=4, occurrence=1)
    assert qdrant_point_id(first) == qdrant_point_id(first)
    assert str(uuid.UUID(qdrant_point_id(first))) == qdrant_point_id(first)


def test_content_fingerprint_tracks_order_text_and_structure_but_not_ids() -> None:
    first = make_chunk("explicit-a", "document-a", "Revenue 50", page=1)
    second = make_chunk("explicit-b", "document-a", "Growth 12.5%", page=2)
    equivalent_first = make_chunk("different-id", "document-a", "Revenue 50", page=1)

    original = prepared_content_sha256((first, second))
    assert original == prepared_content_sha256((equivalent_first, second))
    assert original != prepared_content_sha256((second, first))
    assert original != prepared_content_sha256(
        (make_chunk("explicit-a", "document-a", "Revenue 51", page=1), second)
    )
    assert original != prepared_content_sha256(
        (make_chunk("explicit-a", "document-a", "Revenue 50", page=2), second)
    )


def test_planner_classifies_new_unchanged_and_changed_and_retains_absent_documents() -> None:
    signature = IndexSignature("finance", "test:embedding", 2)
    document_a = make_prepared("document-a", "old", source="a.pdf")
    document_b = make_prepared("document-b", "other", source="b.pdf")
    first_plan = plan_incremental_index(None, (document_a, document_b))
    assert [item.state for item in first_plan.documents] == [
        DocumentIndexState.NEW,
        DocumentIndexState.NEW,
    ]
    manifest = apply_index_plan(None, signature, first_plan)

    changed_a = make_prepared("document-a", "new", source="a.pdf")
    second_plan = plan_incremental_index(manifest, (document_b, changed_a))
    assert [item.state for item in second_plan.documents] == [
        DocumentIndexState.UNCHANGED,
        DocumentIndexState.CHANGED,
    ]
    assert second_plan.stale_chunk_ids == document_a.chunk_ids
    updated = apply_index_plan(manifest, signature, second_plan)
    assert set(updated.documents_by_id) == {"document-a", "document-b"}
    assert updated.documents_by_id["document-b"] == manifest.documents_by_id["document-b"]


def test_manifest_json_is_deterministic_round_trip_and_atomic(tmp_path: Path) -> None:
    signature = IndexSignature("finance", "test:embedding", 2)
    documents = (
        make_prepared("document-b", "beta", source="b.pdf"),
        make_prepared("document-a", "alpha", source="a.pdf"),
    )
    manifest = apply_index_plan(None, signature, plan_incremental_index(None, documents))
    path = tmp_path / "nested" / "manifest.json"

    write_manifest(path, manifest)
    first_bytes = path.read_bytes()
    write_manifest(path, manifest)

    assert path.read_bytes() == first_bytes
    assert load_manifest(path) == manifest
    assert list(json.loads(first_bytes)["documents"])[0]["document_id"] == "document-a"
    assert list(path.parent.glob("*.tmp")) == []


@pytest.mark.parametrize(
    "content",
    [
        "not-json",
        '{"manifest_version": 999, "index_signature": {}, "documents": []}',
        '{"manifest_version": 1, "index_signature": {}, "documents": []}',
    ],
)
def test_manifest_rejects_malformed_or_incompatible_json(tmp_path: Path, content: str) -> None:
    path = tmp_path / "manifest.json"
    path.write_text(content, encoding="utf-8")
    with pytest.raises(IndexManifestError, match="manifest"):
        load_manifest(path)


def test_manifest_rejects_global_chunk_id_collisions() -> None:
    signature = IndexSignature("finance", "test:embedding", 2)
    first = make_prepared("document-a", "same")
    second_chunk = make_chunk(first.chunk_ids[0], "document-b", "different", source="b.pdf")
    second = PreparedDocument(
        "document-b",
        "b.pdf",
        prepared_content_sha256((second_chunk,)),
        (second_chunk,),
    )
    with pytest.raises(ValueError, match="globally unique"):
        IndexManifest(signature, (first.to_manifest_document(), second.to_manifest_document()))


def test_planner_rejects_chunk_collision_with_an_absent_retained_document() -> None:
    signature = IndexSignature("finance", "test:embedding", 2)
    retained = make_prepared("document-a", "same", source="a.pdf")
    manifest = apply_index_plan(
        None,
        signature,
        plan_incremental_index(None, (retained,)),
    )
    colliding_chunk = make_chunk(
        retained.chunk_ids[0],
        "document-b",
        "different",
        source="b.pdf",
    )
    colliding = PreparedDocument(
        "document-b",
        "b.pdf",
        prepared_content_sha256((colliding_chunk,)),
        (colliding_chunk,),
    )

    with pytest.raises(ValueError, match="retained manifest"):
        plan_incremental_index(manifest, (colliding,))
