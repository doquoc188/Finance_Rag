"""Real compiled-graph coverage for injected hybrid retrieval."""

from collections.abc import Mapping

from langchain_core.documents import Document
from langgraph.graph.state import CompiledStateGraph

from app.graph.state import FinGraphUpdate
from app.graph.workflow import build_finrag_graph
from finrag.domain import RetrievalScore, RetrievalStrategy, RetrievedEvidence, ScoreType
from finrag.models import ModelRoles
from finrag.retrieval import HybridRetriever
from finrag.verification import VerificationStatus
from tests.chat_models import WorkflowChatModel
from tests.retrievers import document_evidence


class QueryRetriever:
    def __init__(
        self,
        strategy: RetrievalStrategy,
        responses: Mapping[str, RetrievedEvidence],
    ) -> None:
        self.strategy = strategy
        self.responses = dict(responses)
        self.calls: list[tuple[str, int]] = []

    def retrieve(self, query: str, *, limit: int) -> tuple[RetrievedEvidence, ...]:
        self.calls.append((query, limit))
        result = self.responses.get(query)
        if result is None:
            return ()
        score_type = (
            ScoreType.COSINE if self.strategy is RetrievalStrategy.DENSE else ScoreType.BM25
        )
        return (
            RetrievedEvidence(
                result.chunk,
                1,
                self.strategy,
                RetrievalScore(0.9 if score_type is ScoreType.COSINE else 12.0, score_type),
            ),
        )


def test_hybrid_evidence_reaches_real_graph_after_one_bounded_rewrite() -> None:
    question = "What was revenue?"
    rewritten = "FPT revenue 2025"
    irrelevant = document_evidence(
        Document(page_content="Employee headcount was 10."), "irrelevant"
    )
    relevant = document_evidence(
        Document(
            page_content="Revenue reached 50 billion VND.",
            metadata={"source": "fpt-2025.pdf", "page": 7, "year": "2025"},
        ),
        "relevant",
    )
    responses = {question: irrelevant, rewritten: relevant}
    dense = QueryRetriever(RetrievalStrategy.DENSE, responses)
    sparse = QueryRetriever(RetrievalStrategy.SPARSE, responses)
    hybrid = HybridRetriever(
        dense,
        sparse,
        dense_top_k=2,
        sparse_top_k=2,
        fused_top_k=2,
    )
    generation = WorkflowChatModel(role="generation", responses=["Revenue reached 50 billion VND."])
    grading = WorkflowChatModel(role="grading", responses=["unused"], binary_scores=("no", "yes"))
    rewrite = WorkflowChatModel(role="query_rewrite", responses=[rewritten])
    graph = build_finrag_graph(
        ModelRoles(generation, grading, rewrite),
        max_rewrite_attempts=2,
        retriever=hybrid,
        retrieval_limit=2,
    )
    assert isinstance(graph, CompiledStateGraph)
    initial: FinGraphUpdate = {
        "question": question,
        "rewrite_attempts": 0,
        "retry_count": 0,
        "web_search_needed": False,
    }

    result = graph.invoke(initial)

    expected_calls = [(question, 2), (rewritten, 2)]
    assert dense.calls == sparse.calls == expected_calls
    assert result["rewrite_attempts"] == result["retry_count"] == 1
    assert [document.page_content for document in result["documents"]] == [relevant.chunk.text]
    assert result["documents"][0].metadata["retrieval_strategy"] == "hybrid"
    assert result["documents"][0].metadata["retrieval_score_type"] == "rrf"
    assert result["documents"][0].metadata["retrieval_score"] == 2 / 61
    assert relevant.chunk.text in generation.calls[0].input
    assert irrelevant.chunk.text not in generation.calls[0].input
    assert result["numeric_verification"][0].status is VerificationStatus.SUPPORTED
