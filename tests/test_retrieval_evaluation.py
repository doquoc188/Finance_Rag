"""Framework-independent tests for the retrieval evaluation harness."""

import json
from collections.abc import Mapping, Sequence
from pathlib import Path

import pytest

from finrag.domain import DocumentChunk, RetrievalStrategy, RetrievedEvidence, SourceReference
from finrag.evaluation import (
    RETRIEVAL_STRATEGY_NAMES,
    BenchmarkQuery,
    RetrievalBenchmark,
    build_evaluation_report,
    evaluate_retriever,
    format_markdown_report,
    load_retrieval_benchmark,
    recall_at_k,
    reciprocal_rank_at_k,
    render_evaluation_json,
)
from finrag.retrieval import BM25Retriever

BENCHMARK_ROOT = Path(__file__).parent.parent / "evals" / "retrieval"


@pytest.mark.parametrize(
    ("retrieved", "relevant", "cutoff", "expected"),
    [
        (("A", "B"), ("A",), 5, 1.0),
        (("X1", "X2", "X3", "X4", "A"), ("A",), 5, 1.0),
        (("X1", "X2", "X3", "X4", "X5", "A"), ("A",), 5, 0.0),
        (("A", "X", "B"), ("A", "B"), 3, 1.0),
        (("X", "Y"), ("A",), 10, 0.0),
        (("X", "A"), ("A",), 10, 1.0),
    ],
)
def test_recall_at_k_exact_math(
    retrieved: tuple[str, ...],
    relevant: tuple[str, ...],
    cutoff: int,
    expected: float,
) -> None:
    assert recall_at_k(retrieved, relevant, cutoff) == expected


@pytest.mark.parametrize(
    ("retrieved", "relevant", "cutoff", "expected"),
    [
        (("A", "B"), ("A",), 10, 1.0),
        (("X1", "X2", "X3", "X4", "A"), ("A",), 10, 0.2),
        (("X1", "X2", "X3", "X4", "X5", "A"), ("A",), 5, 0.0),
        (("X", "B", "A"), ("A", "B"), 10, 0.5),
        (("X", "Y"), ("A",), 10, 0.0),
        (("X", "A"), ("A",), 10, 0.5),
    ],
)
def test_reciprocal_rank_at_k_exact_math(
    retrieved: tuple[str, ...],
    relevant: tuple[str, ...],
    cutoff: int,
    expected: float,
) -> None:
    assert reciprocal_rank_at_k(retrieved, relevant, cutoff) == expected


def test_metrics_reject_invalid_cutoffs_and_empty_gold() -> None:
    with pytest.raises(ValueError, match="cutoff"):
        recall_at_k(("A",), ("A",), 0)
    with pytest.raises(ValueError, match="relevant_chunk_ids"):
        recall_at_k(("A",), (), 5)
    with pytest.raises(ValueError, match="relevant_chunk_ids"):
        reciprocal_rank_at_k(("A",), (), 10)


def test_curated_benchmark_validates_explicit_stable_ids_and_composition() -> None:
    benchmark = load_retrieval_benchmark(
        BENCHMARK_ROOT / "corpus.jsonl",
        BENCHMARK_ROOT / "queries.jsonl",
        name="finance-retrieval-curated-v1",
    )

    assert len(benchmark.chunks) == 47
    assert len(benchmark.queries) == 27
    assert all(chunk.chunk_id.startswith("bench_") for chunk in benchmark.chunks)
    assert all(query.query_id.startswith("q_") for query in benchmark.queries)
    assert {query.category for query in benchmark.queries} == {
        "bilingual_terminology",
        "exact_company_metric",
        "hard_metric",
        "hard_period",
        "hard_year",
        "multi_relevant",
        "semantic_paraphrase",
    }
    assert sum(len(query.relevant_chunk_ids) > 1 for query in benchmark.queries) == 2
    assert all(chunk.metadata["synthetic"] is True for chunk in benchmark.chunks)


def write_jsonl(path: Path, records: Sequence[object]) -> None:
    path.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )


def valid_corpus_record(chunk_id: str = "bench_a") -> dict[str, object]:
    return {
        "chunk_id": chunk_id,
        "document_id": "bench_document",
        "text": "Synthetic evidence.",
        "metadata": {"source": "fixture", "synthetic": True},
    }


def valid_query_record(query_id: str = "q_a") -> dict[str, object]:
    return {
        "query_id": query_id,
        "question": "What is the evidence?",
        "relevant_chunk_ids": ["bench_a"],
    }


@pytest.mark.parametrize(
    ("corpus", "queries", "message"),
    [
        (
            [valid_corpus_record(), valid_corpus_record()],
            [valid_query_record()],
            "duplicate chunk_id",
        ),
        (
            [valid_corpus_record()],
            [valid_query_record(), valid_query_record()],
            "duplicate query_id",
        ),
        (
            [valid_corpus_record()],
            [{**valid_query_record(), "question": "  "}],
            "question",
        ),
        (
            [valid_corpus_record()],
            [{**valid_query_record(), "relevant_chunk_ids": []}],
            "relevant_chunk_ids",
        ),
        (
            [valid_corpus_record()],
            [{**valid_query_record(), "relevant_chunk_ids": ["missing"]}],
            "unknown relevant chunk_id",
        ),
        (
            [{**valid_corpus_record(), "metadata": []}],
            [valid_query_record()],
            "metadata",
        ),
    ],
)
def test_benchmark_validation_rejects_invalid_data(
    tmp_path: Path,
    corpus: list[dict[str, object]],
    queries: list[dict[str, object]],
    message: str,
) -> None:
    corpus_path = tmp_path / "corpus.jsonl"
    query_path = tmp_path / "queries.jsonl"
    write_jsonl(corpus_path, corpus)
    write_jsonl(query_path, queries)

    with pytest.raises(ValueError, match=message):
        load_retrieval_benchmark(corpus_path, query_path, name="invalid")


def test_benchmark_validation_rejects_malformed_json(tmp_path: Path) -> None:
    corpus_path = tmp_path / "corpus.jsonl"
    query_path = tmp_path / "queries.jsonl"
    corpus_path.write_text("{not-json}\n", encoding="utf-8")
    write_jsonl(query_path, [valid_query_record()])

    with pytest.raises(ValueError, match="invalid JSON.*corpus.jsonl:1"):
        load_retrieval_benchmark(corpus_path, query_path, name="invalid")


def make_chunk(chunk_id: str) -> DocumentChunk:
    source = SourceReference("benchmark-document", "fixture")
    return DocumentChunk(chunk_id, source.document_id, f"Evidence {chunk_id}", source)


class RecordingRetriever:
    def __init__(self, results: Mapping[str, Sequence[RetrievedEvidence]]) -> None:
        self.results = {query: tuple(items) for query, items in results.items()}
        self.calls: list[tuple[str, int]] = []

    def retrieve(self, query: str, *, limit: int) -> tuple[RetrievedEvidence, ...]:
        self.calls.append((query, limit))
        return self.results[query][:limit]


def ranked(*chunk_ids: str) -> tuple[RetrievedEvidence, ...]:
    return tuple(
        RetrievedEvidence(make_chunk(chunk_id), rank, RetrievalStrategy.DENSE)
        for rank, chunk_id in enumerate(chunk_ids, start=1)
    )


def small_benchmark() -> RetrievalBenchmark:
    chunks = tuple(make_chunk(chunk_id) for chunk_id in ("A", "B", "C"))
    queries = (
        BenchmarkQuery("q1", "first question", ("A",), "unit", "en"),
        BenchmarkQuery("q2", "second question", ("B", "C"), "unit", "en"),
    )
    return RetrievalBenchmark("unit-benchmark", chunks, queries)


def test_runner_evaluates_each_query_once_and_respects_returned_rank_order() -> None:
    benchmark = small_benchmark()
    retriever = RecordingRetriever(
        {
            "first question": ranked("X", "A"),
            "second question": ranked("C", "X", "B"),
        }
    )

    result = evaluate_retriever("dense", retriever, benchmark.queries)

    assert retriever.calls == [("first question", 10), ("second question", 10)]
    assert result.query_count == 2
    assert result.recall_at_5 == 1.0
    assert result.recall_at_10 == 1.0
    assert result.mrr_at_10 == 0.75
    assert result.per_query[0].retrieved_chunk_ids == ("X", "A")
    assert result.per_query[0].relevant_hits[0].rank == 2
    assert [(hit.chunk_id, hit.rank) for hit in result.per_query[1].relevant_hits] == [
        ("C", 1),
        ("B", 3),
    ]


def test_runner_rejects_unknown_strategy_and_invalid_result_ranks() -> None:
    benchmark = small_benchmark()
    retriever = RecordingRetriever(
        {
            "first question": (RetrievedEvidence(make_chunk("A"), 2, RetrievalStrategy.DENSE),),
            "second question": ranked("B"),
        }
    )

    with pytest.raises(ValueError, match="strategy name"):
        evaluate_retriever("renamed_dense", retriever, benchmark.queries)
    with pytest.raises(ValueError, match="sequential ranks"):
        evaluate_retriever("dense", retriever, benchmark.queries)


def test_strategy_names_and_result_serialization_are_stable() -> None:
    assert RETRIEVAL_STRATEGY_NAMES == (
        "dense",
        "bm25",
        "hybrid_rrf",
        "hybrid_rrf_reranked",
    )
    benchmark = small_benchmark()
    result = evaluate_retriever(
        "bm25",
        RecordingRetriever({"first question": ranked("A"), "second question": ranked("B", "C")}),
        benchmark.queries,
    )
    report = build_evaluation_report(
        benchmark,
        (result,),
        unmeasured={"dense": "local embedding model unavailable"},
        configuration={"retrieval_limit": 10, "rrf_k": 60},
    )

    first = render_evaluation_json(report)
    second = render_evaluation_json(report)

    assert first == second
    assert "timestamp" not in first.lower()
    decoded = json.loads(first)
    assert decoded["strategies"]["bm25"]["recall_at_5"] == 1.0
    assert decoded["unmeasured"] == {"dense": "local embedding model unavailable"}
    markdown = format_markdown_report(report)
    assert "| BM25 | 1.0000 | 1.0000 | 1.0000 | MEASURED |" in markdown
    assert "| Dense | — | — | — | NOT MEASURED" in markdown


def test_actual_bm25_implementation_runs_over_curated_benchmark() -> None:
    benchmark = load_retrieval_benchmark(
        BENCHMARK_ROOT / "corpus.jsonl",
        BENCHMARK_ROOT / "queries.jsonl",
        name="finance-retrieval-curated-v1",
    )

    result = evaluate_retriever("bm25", BM25Retriever(benchmark.chunks), benchmark.queries)

    assert result.strategy == "bm25"
    assert result.query_count == 27
    assert 0.0 <= result.recall_at_5 <= result.recall_at_10 <= 1.0
    assert 0.0 <= result.mrr_at_10 <= 1.0
