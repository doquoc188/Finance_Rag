"""Real compiled-graph coverage for hybrid retrieval followed by reranking."""

from langgraph.graph.state import CompiledStateGraph

from app.graph.state import FinGraphUpdate
from app.graph.workflow import build_finrag_graph
from finrag.domain import RetrievalStrategy, ScoreType
from finrag.models import ModelRoles
from finrag.retrieval import HybridRetriever
from tests.chat_models import WorkflowChatModel
from tests.rerankers import ScoreReranker
from tests.test_finrag_retrieval import RecordingRetriever, evidence, make_chunk


def test_reranked_hybrid_evidence_order_reaches_real_graph_generation() -> None:
    chunks = (
        make_chunk("eps-2024", "FPT EPS 2024 was 6,000 VND."),
        make_chunk("revenue-2025", "FPT revenue 2025 was 50 trillion VND."),
        make_chunk("eps-2025", "FPT EPS 2025 was 7,200 VND."),
    )
    dense = RecordingRetriever(
        tuple(
            evidence(chunk, rank, RetrievalStrategy.DENSE, 1.0 - rank / 10)
            for rank, chunk in enumerate(chunks, start=1)
        )
    )
    sparse = RecordingRetriever(
        tuple(
            evidence(chunk, rank, RetrievalStrategy.SPARSE, 10.0 - rank)
            for rank, chunk in enumerate(chunks, start=1)
        )
    )
    reranker = ScoreReranker({"eps-2024": 0.2, "revenue-2025": 0.1, "eps-2025": 0.95})
    hybrid = HybridRetriever(
        dense,
        sparse,
        dense_top_k=3,
        sparse_top_k=3,
        fused_top_k=3,
        reranker=reranker,
    )
    generation = WorkflowChatModel(role="generation", responses=["Evidence order received."])
    grading = WorkflowChatModel(role="grading", responses=["unused"], binary_score="yes")
    rewrite = WorkflowChatModel(role="query_rewrite", responses=["must not be called"])
    graph = build_finrag_graph(
        ModelRoles(generation, grading, rewrite), retriever=hybrid, retrieval_limit=3
    )
    assert isinstance(graph, CompiledStateGraph)
    initial: FinGraphUpdate = {
        "question": "FPT EPS 2025",
        "rewrite_attempts": 0,
        "retry_count": 0,
        "web_search_needed": False,
    }

    result = graph.invoke(initial)

    assert dense.calls == sparse.calls == [("FPT EPS 2025", 3)]
    assert reranker.calls == [("FPT EPS 2025", ("eps-2024", "revenue-2025", "eps-2025"), 3)]
    assert [document.metadata["chunk_id"] for document in result["documents"]] == [
        "eps-2025",
        "eps-2024",
        "revenue-2025",
    ]
    assert all(
        document.metadata["retrieval_score_type"] == ScoreType.RERANKER.value
        for document in result["documents"]
    )
    generation_input = generation.calls[0].input
    assert generation_input.index(chunks[2].text) < generation_input.index(chunks[0].text)
    assert generation_input.index(chunks[0].text) < generation_input.index(chunks[1].text)
    assert rewrite.calls == []
