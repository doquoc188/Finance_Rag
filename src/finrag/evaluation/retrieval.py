"""Typed data loading, metrics, and reporting for retrieval-only evaluation."""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from json import JSONDecodeError
from math import fsum, isfinite
from pathlib import Path
from types import MappingProxyType
from typing import Literal

from finrag.domain import DocumentChunk, MetadataValue, SourceReference
from finrag.retrieval import Retriever

type RetrievalStrategyName = Literal[
    "dense",
    "bm25",
    "hybrid_rrf",
    "hybrid_rrf_reranked",
]

RETRIEVAL_STRATEGY_NAMES: tuple[RetrievalStrategyName, ...] = (
    "dense",
    "bm25",
    "hybrid_rrf",
    "hybrid_rrf_reranked",
)
EVALUATION_RETRIEVAL_LIMIT = 10

_STRATEGY_LABELS: Mapping[RetrievalStrategyName, str] = MappingProxyType(
    {
        "dense": "Dense",
        "bm25": "BM25",
        "hybrid_rrf": "Hybrid RRF",
        "hybrid_rrf_reranked": "Hybrid RRF + reranker",
    }
)


@dataclass(frozen=True, slots=True)
class BenchmarkQuery:
    query_id: str
    question: str
    relevant_chunk_ids: tuple[str, ...]
    category: str | None = None
    language: str | None = None

    def __post_init__(self) -> None:
        _require_nonblank(self.query_id, "query_id")
        _require_nonblank(self.question, "question")
        if not self.relevant_chunk_ids:
            raise ValueError("relevant_chunk_ids must contain at least one chunk ID")
        if len(set(self.relevant_chunk_ids)) != len(self.relevant_chunk_ids):
            raise ValueError("relevant_chunk_ids must not contain duplicates")
        for chunk_id in self.relevant_chunk_ids:
            _require_nonblank(chunk_id, "relevant chunk_id")
        if self.category is not None:
            _require_nonblank(self.category, "category")
        if self.language is not None:
            _require_nonblank(self.language, "language")


@dataclass(frozen=True, slots=True)
class RetrievalBenchmark:
    name: str
    chunks: tuple[DocumentChunk, ...]
    queries: tuple[BenchmarkQuery, ...]

    def __post_init__(self) -> None:
        _require_nonblank(self.name, "benchmark name")
        if not self.chunks:
            raise ValueError("benchmark corpus must contain at least one chunk")
        if not self.queries:
            raise ValueError("benchmark queries must contain at least one query")

        chunk_ids = [chunk.chunk_id for chunk in self.chunks]
        duplicate_chunk_id = _first_duplicate(chunk_ids)
        if duplicate_chunk_id is not None:
            raise ValueError(f"duplicate chunk_id: {duplicate_chunk_id}")
        query_ids = [query.query_id for query in self.queries]
        duplicate_query_id = _first_duplicate(query_ids)
        if duplicate_query_id is not None:
            raise ValueError(f"duplicate query_id: {duplicate_query_id}")

        known_chunk_ids = set(chunk_ids)
        for query in self.queries:
            for relevant_id in query.relevant_chunk_ids:
                if relevant_id not in known_chunk_ids:
                    raise ValueError(
                        f"query {query.query_id!r} references unknown relevant chunk_id: "
                        f"{relevant_id}"
                    )


@dataclass(frozen=True, slots=True)
class RelevantHit:
    chunk_id: str
    rank: int


@dataclass(frozen=True, slots=True)
class QueryEvaluation:
    query_id: str
    question: str
    relevant_chunk_ids: tuple[str, ...]
    retrieved_chunk_ids: tuple[str, ...]
    relevant_hits: tuple[RelevantHit, ...]
    recall_at_5: float
    recall_at_10: float
    reciprocal_rank_at_10: float


@dataclass(frozen=True, slots=True)
class RetrievalEvaluation:
    strategy: RetrievalStrategyName
    query_count: int
    recall_at_5: float
    recall_at_10: float
    mrr_at_10: float
    per_query: tuple[QueryEvaluation, ...]


@dataclass(frozen=True, slots=True)
class EvaluationReport:
    benchmark: str
    corpus_size: int
    query_count: int
    configuration: Mapping[str, object]
    strategies: tuple[RetrievalEvaluation, ...]
    unmeasured: Mapping[RetrievalStrategyName, str]

    def __post_init__(self) -> None:
        _require_nonblank(self.benchmark, "benchmark")
        _require_positive_integer(self.corpus_size, "corpus_size")
        _require_positive_integer(self.query_count, "query_count")
        strategy_names = [result.strategy for result in self.strategies]
        duplicate_strategy = _first_duplicate(strategy_names)
        if duplicate_strategy is not None:
            raise ValueError(f"duplicate measured strategy: {duplicate_strategy}")
        for result in self.strategies:
            if result.query_count != self.query_count:
                raise ValueError("all strategy results must use the benchmark query count")
        for strategy, reason in self.unmeasured.items():
            _validate_strategy_name(strategy)
            _require_nonblank(reason, "unmeasured reason")
            if strategy in strategy_names:
                raise ValueError(f"strategy cannot be measured and unmeasured: {strategy}")
        object.__setattr__(self, "configuration", MappingProxyType(dict(self.configuration)))
        object.__setattr__(self, "unmeasured", MappingProxyType(dict(self.unmeasured)))


def load_retrieval_benchmark(
    corpus_path: Path,
    queries_path: Path,
    *,
    name: str,
) -> RetrievalBenchmark:
    """Load and validate strict JSONL corpus/query records."""

    chunks = tuple(
        _parse_chunk(record, corpus_path, line_number)
        for line_number, record in _read_jsonl(corpus_path)
    )
    queries = tuple(
        _parse_query(record, queries_path, line_number)
        for line_number, record in _read_jsonl(queries_path)
    )
    return RetrievalBenchmark(name, chunks, queries)


def recall_at_k(
    retrieved_chunk_ids: Sequence[str],
    relevant_chunk_ids: Sequence[str],
    cutoff: int,
) -> float:
    """Return relevant IDs retrieved in top K divided by all relevant IDs."""

    _require_positive_integer(cutoff, "cutoff")
    relevant = _validated_relevant_ids(relevant_chunk_ids)
    retrieved = set(retrieved_chunk_ids[:cutoff])
    return len(retrieved & relevant) / len(relevant)


def reciprocal_rank_at_k(
    retrieved_chunk_ids: Sequence[str],
    relevant_chunk_ids: Sequence[str],
    cutoff: int,
) -> float:
    """Return reciprocal rank of the first relevant top-K result, or zero."""

    _require_positive_integer(cutoff, "cutoff")
    relevant = _validated_relevant_ids(relevant_chunk_ids)
    for rank, chunk_id in enumerate(retrieved_chunk_ids[:cutoff], start=1):
        if chunk_id in relevant:
            return 1.0 / rank
    return 0.0


def evaluate_retriever(
    strategy: str,
    retriever: Retriever,
    queries: Sequence[BenchmarkQuery],
) -> RetrievalEvaluation:
    """Evaluate one named retriever once per query at the fixed maximum cutoff."""

    strategy_name = _validate_strategy_name(strategy)
    if not queries:
        raise ValueError("queries must contain at least one benchmark query")
    duplicate_query_id = _first_duplicate(query.query_id for query in queries)
    if duplicate_query_id is not None:
        raise ValueError(f"duplicate query_id: {duplicate_query_id}")

    per_query: list[QueryEvaluation] = []
    for query in queries:
        evidence = tuple(retriever.retrieve(query.question, limit=EVALUATION_RETRIEVAL_LIMIT))
        if len(evidence) > EVALUATION_RETRIEVAL_LIMIT:
            raise ValueError("retriever returned more than the requested evaluation limit")
        for expected_rank, item in enumerate(evidence, start=1):
            if item.rank != expected_rank:
                raise ValueError("retriever results must have sequential ranks")
        retrieved_ids = tuple(item.chunk.chunk_id for item in evidence)
        duplicate_chunk_id = _first_duplicate(retrieved_ids)
        if duplicate_chunk_id is not None:
            raise ValueError(f"retriever returned duplicate chunk_id: {duplicate_chunk_id}")
        relevant = set(query.relevant_chunk_ids)
        hits = tuple(
            RelevantHit(chunk_id, rank)
            for rank, chunk_id in enumerate(retrieved_ids, start=1)
            if chunk_id in relevant
        )
        per_query.append(
            QueryEvaluation(
                query_id=query.query_id,
                question=query.question,
                relevant_chunk_ids=query.relevant_chunk_ids,
                retrieved_chunk_ids=retrieved_ids,
                relevant_hits=hits,
                recall_at_5=recall_at_k(retrieved_ids, query.relevant_chunk_ids, 5),
                recall_at_10=recall_at_k(retrieved_ids, query.relevant_chunk_ids, 10),
                reciprocal_rank_at_10=reciprocal_rank_at_k(
                    retrieved_ids, query.relevant_chunk_ids, 10
                ),
            )
        )

    count = len(per_query)
    return RetrievalEvaluation(
        strategy=strategy_name,
        query_count=count,
        recall_at_5=fsum(result.recall_at_5 for result in per_query) / count,
        recall_at_10=fsum(result.recall_at_10 for result in per_query) / count,
        mrr_at_10=fsum(result.reciprocal_rank_at_10 for result in per_query) / count,
        per_query=tuple(per_query),
    )


def build_evaluation_report(
    benchmark: RetrievalBenchmark,
    strategies: Sequence[RetrievalEvaluation],
    *,
    unmeasured: Mapping[RetrievalStrategyName, str],
    configuration: Mapping[str, object],
) -> EvaluationReport:
    return EvaluationReport(
        benchmark=benchmark.name,
        corpus_size=len(benchmark.chunks),
        query_count=len(benchmark.queries),
        configuration=configuration,
        strategies=tuple(strategies),
        unmeasured=unmeasured,
    )


def render_evaluation_json(report: EvaluationReport) -> str:
    """Render deterministic machine-readable output without rounding or timestamps."""

    return (
        json.dumps(
            _report_to_dict(report),
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        + "\n"
    )


def format_markdown_report(report: EvaluationReport) -> str:
    """Render a small human table, rounding metrics only for display."""

    measured = {result.strategy: result for result in report.strategies}
    lines = [
        f"# Retrieval baseline — {report.benchmark}",
        "",
        f"Corpus chunks: {report.corpus_size}  ",
        f"Queries: {report.query_count}",
        "",
        "| Strategy | Recall@5 | Recall@10 | MRR@10 | Status |",
        "| --- | ---: | ---: | ---: | --- |",
    ]
    for strategy in RETRIEVAL_STRATEGY_NAMES:
        label = _STRATEGY_LABELS[strategy]
        result = measured.get(strategy)
        if result is not None:
            lines.append(
                f"| {label} | {result.recall_at_5:.4f} | {result.recall_at_10:.4f} | "
                f"{result.mrr_at_10:.4f} | MEASURED |"
            )
        elif strategy in report.unmeasured:
            lines.append(f"| {label} | — | — | — | NOT MEASURED: {report.unmeasured[strategy]} |")
        else:
            lines.append(f"| {label} | — | — | — | NOT REQUESTED |")
    lines.extend(
        [
            "",
            "Metrics are macro averages over manual query relevance labels. This small",
            "curated synthetic benchmark is not a production-quality claim.",
            "",
        ]
    )
    return "\n".join(lines)


def write_evaluation_outputs(
    report: EvaluationReport,
    *,
    json_path: Path,
    markdown_path: Path,
) -> None:
    json_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(render_evaluation_json(report), encoding="utf-8")
    markdown_path.write_text(format_markdown_report(report), encoding="utf-8")


def _read_jsonl(path: Path) -> tuple[tuple[int, object], ...]:
    records: list[tuple[int, object]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            raise ValueError(f"blank JSONL record at {path}:{line_number}")
        try:
            value: object = json.loads(line)
        except JSONDecodeError as error:
            raise ValueError(f"invalid JSON at {path}:{line_number}") from error
        records.append((line_number, value))
    return tuple(records)


def _parse_chunk(value: object, path: Path, line_number: int) -> DocumentChunk:
    context = f"{path}:{line_number}"
    record = _record(value, context)
    _require_keys(record, {"chunk_id", "document_id", "text", "metadata"}, context)
    chunk_id = _record_string(record, "chunk_id", context)
    document_id = _record_string(record, "document_id", context)
    text = _record_string(record, "text", context)
    metadata = _metadata(record["metadata"], context)
    source = metadata.get("source")
    if not isinstance(source, str) or not source.strip():
        raise ValueError(f"metadata.source must be a nonblank string at {context}")
    page_value = metadata.get("page")
    if page_value is not None and (
        isinstance(page_value, bool) or not isinstance(page_value, int) or page_value < 1
    ):
        raise ValueError(f"metadata.page must be a positive integer at {context}")
    section_value = metadata.get("section")
    if section_value is not None and (
        not isinstance(section_value, str) or not section_value.strip()
    ):
        raise ValueError(f"metadata.section must be a nonblank string at {context}")
    reference = SourceReference(
        document_id,
        source,
        page=page_value,
        section=section_value,
    )
    return DocumentChunk(chunk_id, document_id, text, reference, metadata)


def _parse_query(value: object, path: Path, line_number: int) -> BenchmarkQuery:
    context = f"{path}:{line_number}"
    record = _record(value, context)
    required = {"query_id", "question", "relevant_chunk_ids"}
    optional = {"category", "language"}
    _require_keys(record, required, context, optional=optional)
    relevant_value = record["relevant_chunk_ids"]
    if not isinstance(relevant_value, list):
        raise ValueError(f"relevant_chunk_ids must be a list at {context}")
    relevant_ids = tuple(
        _string_value(item, f"relevant_chunk_ids[{index}]", context)
        for index, item in enumerate(relevant_value)
    )
    category = _optional_record_string(record, "category", context)
    language = _optional_record_string(record, "language", context)
    return BenchmarkQuery(
        _record_string(record, "query_id", context),
        _record_string(record, "question", context),
        relevant_ids,
        category,
        language,
    )


def _record(value: object, context: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ValueError(f"record must be a JSON object at {context}")
    result: dict[str, object] = {}
    for key, item in value.items():
        if not isinstance(key, str):
            raise ValueError(f"record keys must be strings at {context}")
        result[key] = item
    return result


def _require_keys(
    record: Mapping[str, object],
    required: set[str],
    context: str,
    *,
    optional: set[str] | None = None,
) -> None:
    missing = required - record.keys()
    if missing:
        raise ValueError(f"missing fields {sorted(missing)} at {context}")
    allowed = required | (optional or set())
    unexpected = record.keys() - allowed
    if unexpected:
        raise ValueError(f"unexpected fields {sorted(unexpected)} at {context}")


def _record_string(record: Mapping[str, object], key: str, context: str) -> str:
    return _string_value(record[key], key, context)


def _optional_record_string(record: Mapping[str, object], key: str, context: str) -> str | None:
    if key not in record:
        return None
    return _string_value(record[key], key, context)


def _string_value(value: object, field_name: str, context: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a nonblank string at {context}")
    return value.strip()


def _metadata(value: object, context: str) -> dict[str, MetadataValue]:
    if not isinstance(value, dict):
        raise ValueError(f"metadata must be a JSON object at {context}")
    result: dict[str, MetadataValue] = {}
    for key, item in value.items():
        if not isinstance(key, str) or not key.strip():
            raise ValueError(f"metadata keys must be nonblank strings at {context}")
        if not isinstance(item, (str, int, float, bool, type(None))):
            raise ValueError(f"metadata values must be scalar at {context}")
        if isinstance(item, float) and not isfinite(item):
            raise ValueError(f"metadata numbers must be finite at {context}")
        result[key] = item
    return result


def _validated_relevant_ids(relevant_chunk_ids: Sequence[str]) -> set[str]:
    if not relevant_chunk_ids:
        raise ValueError("relevant_chunk_ids must contain at least one chunk ID")
    result = set(relevant_chunk_ids)
    if len(result) != len(relevant_chunk_ids):
        raise ValueError("relevant_chunk_ids must not contain duplicates")
    return result


def _validate_strategy_name(strategy: str) -> RetrievalStrategyName:
    if strategy not in RETRIEVAL_STRATEGY_NAMES:
        choices = ", ".join(RETRIEVAL_STRATEGY_NAMES)
        raise ValueError(f"strategy name must be one of: {choices}")
    return strategy


def _first_duplicate(values: Iterable[str]) -> str | None:
    seen: set[str] = set()
    for value in values:
        if value in seen:
            return value
        seen.add(value)
    return None


def _require_nonblank(value: str, field_name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a nonblank string")


def _require_positive_integer(value: int, field_name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{field_name} must be a positive integer")


def _report_to_dict(report: EvaluationReport) -> dict[str, object]:
    strategies = {
        result.strategy: {
            "strategy": result.strategy,
            "query_count": result.query_count,
            "recall_at_5": result.recall_at_5,
            "recall_at_10": result.recall_at_10,
            "mrr_at_10": result.mrr_at_10,
            "per_query": [
                {
                    "query_id": query.query_id,
                    "question": query.question,
                    "relevant_chunk_ids": list(query.relevant_chunk_ids),
                    "retrieved_chunk_ids": list(query.retrieved_chunk_ids),
                    "relevant_hits": [
                        {"chunk_id": hit.chunk_id, "rank": hit.rank} for hit in query.relevant_hits
                    ],
                    "recall_at_5": query.recall_at_5,
                    "recall_at_10": query.recall_at_10,
                    "reciprocal_rank_at_10": query.reciprocal_rank_at_10,
                }
                for query in result.per_query
            ],
        }
        for result in report.strategies
    }
    return {
        "benchmark": report.benchmark,
        "corpus_size": report.corpus_size,
        "query_count": report.query_count,
        "configuration": dict(report.configuration),
        "strategies": strategies,
        "unmeasured": dict(report.unmeasured),
    }
