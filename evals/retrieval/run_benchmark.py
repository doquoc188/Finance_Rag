"""Compose and run the curated benchmark with actual retrieval implementations."""

from __future__ import annotations

import argparse
import os
import uuid
from collections.abc import Sequence
from importlib.metadata import version
from pathlib import Path
from platform import python_version
from typing import cast

from finrag.evaluation import (
    RETRIEVAL_STRATEGY_NAMES,
    EvaluationReport,
    RetrievalBenchmark,
    RetrievalEvaluation,
    RetrievalStrategyName,
    build_evaluation_report,
    evaluate_retriever,
    format_markdown_report,
    load_retrieval_benchmark,
    write_evaluation_outputs,
)
from finrag.retrieval import BM25Retriever, HybridRetriever, Reranker, Retriever

BENCHMARK_NAME = "finance-retrieval-curated-v1"
DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"
DENSE_TOP_K = 20
SPARSE_TOP_K = 20
FUSED_TOP_K = 20
FINAL_TOP_K = 10
RRF_K = 60

BENCHMARK_ROOT = Path(__file__).resolve().parent


class LocalModelUnavailableError(RuntimeError):
    """A required model snapshot is absent from local storage."""


def run_selected_strategies(
    benchmark: RetrievalBenchmark,
    strategies: Sequence[str],
    *,
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    reranker_model: str = DEFAULT_RERANKER_MODEL,
) -> EvaluationReport:
    """Run available actual strategies and mark missing local models explicitly."""

    selected = _validated_strategies(strategies)
    measured: list[RetrievalEvaluation] = []
    unmeasured: dict[RetrievalStrategyName, str] = {}
    sparse = BM25Retriever(benchmark.chunks)

    if "bm25" in selected:
        measured.append(evaluate_retriever("bm25", sparse, benchmark.queries))

    dense: Retriever | None = None
    dense_error: str | None = None
    if any(strategy in selected for strategy in ("dense", "hybrid_rrf", "hybrid_rrf_reranked")):
        try:
            dense = _build_dense_retriever(benchmark, embedding_model)
        except LocalModelUnavailableError as error:
            dense_error = str(error)

    if "dense" in selected:
        if dense is None:
            unmeasured["dense"] = cast(str, dense_error)
        else:
            measured.append(evaluate_retriever("dense", dense, benchmark.queries))

    if "hybrid_rrf" in selected:
        if dense is None:
            unmeasured["hybrid_rrf"] = cast(str, dense_error)
        else:
            measured.append(
                evaluate_retriever(
                    "hybrid_rrf",
                    _hybrid(dense, sparse),
                    benchmark.queries,
                )
            )

    if "hybrid_rrf_reranked" in selected:
        reranker_path: str | None = None
        reranker_error: str | None = None
        try:
            reranker_path = _resolve_local_model(reranker_model)
        except LocalModelUnavailableError as error:
            reranker_error = str(error)
        missing = "; ".join(reason for reason in (dense_error, reranker_error) if reason)
        if dense is None or reranker_path is None:
            unmeasured["hybrid_rrf_reranked"] = missing
        else:
            from finrag.retrieval.bge import BgeCrossEncoderReranker

            reranker = BgeCrossEncoderReranker(reranker_path)
            measured.append(
                evaluate_retriever(
                    "hybrid_rrf_reranked",
                    _hybrid(dense, sparse, reranker=reranker),
                    benchmark.queries,
                )
            )

    ordered_results = tuple(
        next(result for result in measured if result.strategy == strategy)
        for strategy in RETRIEVAL_STRATEGY_NAMES
        if any(result.strategy == strategy for result in measured)
    )
    return build_evaluation_report(
        benchmark,
        ordered_results,
        unmeasured=unmeasured,
        configuration={
            "embedding_model": embedding_model,
            "reranker_model": reranker_model,
            "dense_top_k": DENSE_TOP_K,
            "sparse_top_k": SPARSE_TOP_K,
            "fused_top_k": FUSED_TOP_K,
            "final_top_k": FINAL_TOP_K,
            "rrf_k": RRF_K,
            "model_loading": "local_files_only",
            "requested_strategies": list(selected),
            "python_version": python_version(),
            "rank_bm25_version": version("rank-bm25"),
            "qdrant_client_version": version("qdrant-client"),
            "sentence_transformers_version": version("sentence-transformers"),
        },
    )


def _build_dense_retriever(
    benchmark: RetrievalBenchmark,
    embedding_model: str,
) -> Retriever:
    """Build the production dense adapter against in-memory Qdrant and local embeddings."""

    model_path = _resolve_local_model(embedding_model)
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from qdrant_client import QdrantClient
    from qdrant_client.models import Distance, PointStruct, VectorParams

    from app.core.vectorstore import QdrantDenseRetriever

    embeddings = HuggingFaceEmbeddings(
        model_name=model_path,
        model_kwargs={"local_files_only": True},
        show_progress=False,
    )
    vectors = [
        [float(value) for value in vector]
        for vector in embeddings.embed_documents([chunk.text for chunk in benchmark.chunks])
    ]
    if len(vectors) != len(benchmark.chunks) or not vectors or not vectors[0]:
        raise ValueError("embedding output must contain one non-empty vector per chunk")
    dimension = len(vectors[0])
    if any(len(vector) != dimension for vector in vectors):
        raise ValueError("embedding vectors must all have the same dimension")

    client = QdrantClient(location=":memory:")
    collection_name = "finrag_retrieval_evaluation"
    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=dimension, distance=Distance.COSINE),
    )
    client.upsert(
        collection_name=collection_name,
        points=[
            PointStruct(
                id=str(uuid.uuid5(uuid.NAMESPACE_URL, f"finrag-eval:{chunk.chunk_id}")),
                vector=vector,
                payload={"chunk_id": chunk.chunk_id},
            )
            for chunk, vector in zip(benchmark.chunks, vectors, strict=True)
        ],
    )
    return QdrantDenseRetriever(
        client,
        collection_name,
        embeddings,
        benchmark.chunks,
    )


def _hybrid(
    dense: Retriever,
    sparse: Retriever,
    *,
    reranker: Reranker | None = None,
) -> HybridRetriever:
    return HybridRetriever(
        dense,
        sparse,
        dense_top_k=DENSE_TOP_K,
        sparse_top_k=SPARSE_TOP_K,
        fused_top_k=FUSED_TOP_K,
        rrf_k=RRF_K,
        reranker=reranker,
    )


def _resolve_local_model(model_name_or_path: str) -> str:
    if not model_name_or_path.strip():
        raise ValueError("model name must not be blank")
    candidate = Path(model_name_or_path)
    if candidate.is_dir():
        return str(candidate.resolve())

    _force_local_model_mode()
    from huggingface_hub import snapshot_download

    try:
        return snapshot_download(repo_id=model_name_or_path, local_files_only=True)
    except Exception as error:
        raise LocalModelUnavailableError(
            f"local model snapshot unavailable for {model_name_or_path!r}; "
            "download it explicitly before running this strategy"
        ) from error


def _force_local_model_mode() -> None:
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"


def _validated_strategies(strategies: Sequence[str]) -> tuple[RetrievalStrategyName, ...]:
    if not strategies:
        raise ValueError("at least one retrieval strategy is required")
    selected: list[RetrievalStrategyName] = []
    for strategy in strategies:
        if strategy not in RETRIEVAL_STRATEGY_NAMES:
            raise ValueError(f"unknown retrieval strategy: {strategy}")
        typed = strategy
        if typed in selected:
            raise ValueError(f"duplicate retrieval strategy: {strategy}")
        selected.append(typed)
    return tuple(selected)


def _parse_args(arguments: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--strategies",
        nargs="+",
        choices=RETRIEVAL_STRATEGY_NAMES,
        default=list(RETRIEVAL_STRATEGY_NAMES),
    )
    parser.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL)
    parser.add_argument("--reranker-model", default=DEFAULT_RERANKER_MODEL)
    parser.add_argument(
        "--json-output",
        type=Path,
        default=BENCHMARK_ROOT / "results" / "baseline.json",
    )
    parser.add_argument(
        "--markdown-output",
        type=Path,
        default=BENCHMARK_ROOT / "results" / "baseline.md",
    )
    return parser.parse_args(arguments)


def main(arguments: Sequence[str] | None = None) -> int:
    args = _parse_args(arguments)
    benchmark = load_retrieval_benchmark(
        BENCHMARK_ROOT / "corpus.jsonl",
        BENCHMARK_ROOT / "queries.jsonl",
        name=BENCHMARK_NAME,
    )
    report = run_selected_strategies(
        benchmark,
        args.strategies,
        embedding_model=args.embedding_model,
        reranker_model=args.reranker_model,
    )
    write_evaluation_outputs(
        report,
        json_path=args.json_output,
        markdown_path=args.markdown_output,
    )
    print(format_markdown_report(report), end="")
    print(f"JSON: {args.json_output}")
    print(f"Markdown: {args.markdown_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
