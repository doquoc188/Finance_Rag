"""Evidence value semantics, immutability, and explicit retrieval scores."""

import json
import operator
from collections.abc import MutableMapping
from dataclasses import FrozenInstanceError
from typing import Any, cast

import pytest

from finrag.domain import (
    DocumentChunk,
    MetadataValue,
    RetrievalScore,
    RetrievalStrategy,
    RetrievedEvidence,
    ScoreType,
    SourceReference,
)


@pytest.fixture
def source() -> SourceReference:
    return SourceReference("report-2023", "annual-report.pdf", page=14, section="Revenue")


@pytest.fixture
def chunk(source: SourceReference) -> DocumentChunk:
    return DocumentChunk("report-2023:14:1", "report-2023", "Evidence text", source)


def test_source_reference_value_semantics(source: SourceReference) -> None:
    same = SourceReference("report-2023", "annual-report.pdf", page=14, section="Revenue")
    assert source == same
    assert hash(source) == hash(same)
    assert SourceReference("web-document", "https://example.invalid/report").page is None
    with pytest.raises(FrozenInstanceError):
        setattr(source, "page", 2)


@pytest.mark.parametrize("page", [0, -1, True, 1.5, "1"])
def test_invalid_source_page(page: Any) -> None:
    with pytest.raises(ValueError, match="page"):
        SourceReference("document", "report.pdf", page=page)


@pytest.mark.parametrize("changes", [{"document_id": ""}, {"source": "  "}, {"section": "\n"}])
def test_invalid_source_identifiers(changes: dict[str, Any]) -> None:
    values: dict[str, Any] = {"document_id": "document", "source": "report.pdf"}
    values.update(changes)
    with pytest.raises(ValueError):
        SourceReference(**values)


@pytest.mark.parametrize("changes", [{"chunk_id": ""}, {"document_id": " "}, {"text": " \n "}])
def test_invalid_chunk_fields(source: SourceReference, changes: dict[str, Any]) -> None:
    values: dict[str, Any] = {
        "chunk_id": "chunk",
        "document_id": source.document_id,
        "text": "Evidence",
        "source_reference": source,
    }
    values.update(changes)
    with pytest.raises(ValueError):
        DocumentChunk(**values)


def test_chunk_rejects_conflicting_provenance(source: SourceReference) -> None:
    with pytest.raises(ValueError, match="must match document_id"):
        DocumentChunk("chunk", "other-document", "Evidence", source)


def test_chunk_metadata_is_copied_and_immutable(source: SourceReference) -> None:
    metadata: dict[str, MetadataValue] = {"year": 2023, "has_table": True, "unit": "VND"}
    text = "  Evidence\nwith formatting  "
    chunk = DocumentChunk("chunk", source.document_id, text, source, metadata)
    same = DocumentChunk("chunk", source.document_id, text, source, dict(metadata))

    metadata["year"] = 2024
    assert chunk.metadata["year"] == 2023
    assert chunk.text == text
    assert chunk == same
    assert hash(chunk) == hash(same)
    with pytest.raises(FrozenInstanceError):
        setattr(chunk, "text", "Changed")
    with pytest.raises(TypeError):
        cast(MutableMapping[str, MetadataValue], chunk.metadata)["year"] = 2025


@pytest.mark.parametrize("value", [[], {}, set(), float("nan"), float("inf")])
def test_metadata_rejects_mutable_or_nonfinite_values(source: SourceReference, value: Any) -> None:
    with pytest.raises((TypeError, ValueError)):
        DocumentChunk("chunk", source.document_id, "Evidence", source, {"invalid": value})


def test_evidence_without_score_and_with_typed_score(chunk: DocumentChunk) -> None:
    unscored = RetrievedEvidence(chunk, 1, RetrievalStrategy.GRAPH)
    assert unscored.score is None
    score = RetrievalScore(0.8, ScoreType.COSINE)
    evidence = RetrievedEvidence(chunk, 2, RetrievalStrategy.DENSE, score)
    assert evidence.chunk is chunk
    assert evidence.rank == 2
    assert evidence.score == score
    assert evidence == RetrievedEvidence(chunk, 2, RetrievalStrategy.DENSE, score)
    with pytest.raises(FrozenInstanceError):
        setattr(evidence, "rank", 0)
    with pytest.raises(FrozenInstanceError):
        setattr(score, "value", 0.0)


@pytest.mark.parametrize("rank", [0, -1, True, 1.5, "1"])
def test_evidence_rank_must_be_a_positive_integer(chunk: DocumentChunk, rank: Any) -> None:
    with pytest.raises(ValueError, match="rank"):
        RetrievedEvidence(chunk, rank, RetrievalStrategy.DENSE)


@pytest.mark.parametrize("score_type", list(ScoreType))
def test_score_identity_preserves_semantics(score_type: ScoreType) -> None:
    score = RetrievalScore(0.5, score_type)
    assert score.score_type is score_type
    for other_type in ScoreType:
        if other_type is not score_type:
            assert score != RetrievalScore(0.5, other_type)


def test_scores_have_no_implicit_ordering() -> None:
    # Deliberately bypass static rejection to verify runtime rejects ordering too.
    cosine: Any = RetrievalScore(0.5, ScoreType.COSINE)
    bm25: Any = RetrievalScore(10, ScoreType.BM25)
    with pytest.raises(TypeError):
        operator.lt(cosine, bm25)
    # Raw scores are not probabilities: retain negative/large values without normalization.
    assert RetrievalScore(-0.5, ScoreType.COSINE).value == -0.5
    assert RetrievalScore(12.5, ScoreType.RERANKER).value == 12.5


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf"), True, "0.5"])
def test_invalid_score_values(value: Any) -> None:
    with pytest.raises(ValueError, match="finite number"):
        RetrievalScore(value, ScoreType.COSINE)


def test_unlabelled_score_is_rejected(chunk: DocumentChunk) -> None:
    values: dict[str, Any] = {
        "chunk": chunk,
        "rank": 1,
        "strategy": RetrievalStrategy.DENSE,
        "score": 0.5,
    }
    with pytest.raises(TypeError, match="ScoreType"):
        RetrievedEvidence(**values)


@pytest.mark.parametrize("strategy", list(RetrievalStrategy))
def test_strategy_values_round_trip_through_json(strategy: RetrievalStrategy) -> None:
    encoded = json.dumps({"strategy": strategy})
    assert RetrievalStrategy(json.loads(encoded)["strategy"]) is strategy
    assert str(strategy) == strategy.value


def test_only_named_strategy_and_score_categories_are_accepted(chunk: DocumentChunk) -> None:
    with pytest.raises(ValueError):
        RetrievalStrategy("speculative")
    with pytest.raises(ValueError):
        ScoreType("probability")
