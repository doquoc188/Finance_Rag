"""Offline real-graph regressions for bounded retrieval correction policy."""

from typing import cast

import pytest
from langchain_core.documents import Document
from langgraph.graph.state import CompiledStateGraph

from app.graph.state import FinGraphUpdate
from app.graph.workflow import build_finrag_graph
from finrag.models import ModelRoles
from finrag.verification import VerificationStatus
from tests.chat_models import ModelCall, WorkflowChatModel
from tests.retrievers import RoutingRetriever

QUESTION = "What was revenue?"
INSUFFICIENT_EVIDENCE = "Không đủ bằng chứng từ tài liệu được cung cấp để trả lời câu hỏi này."


def initial_state() -> FinGraphUpdate:
    return {
        "question": QUESTION,
        "rewrite_attempts": 0,
        "retry_count": 0,
        "web_search_needed": False,
    }


def graph_models(
    *,
    answer: str = "Revenue was 50 billion VND.",
    grades: tuple[str, ...] = ("yes",),
    rewrites: list[str] | None = None,
) -> tuple[ModelRoles, WorkflowChatModel, WorkflowChatModel, WorkflowChatModel]:
    generation = WorkflowChatModel(role="generation", responses=[answer])
    grading = WorkflowChatModel(
        role="grading", responses=["unused"], binary_scores=grades, binary_score=grades[-1]
    )
    rewrite = WorkflowChatModel(
        role="query_rewrite", responses=rewrites or ["rewrite must not be called"]
    )
    return ModelRoles(generation, grading, rewrite), generation, grading, rewrite


def test_relevant_evidence_generates_without_rewrite_and_runs_numeric_verification() -> None:
    relevant = Document(page_content="Revenue was 50 billion VND.")
    rejected = Document(page_content="The company employed 10 people.")
    retriever = RoutingRetriever({QUESTION: [relevant, rejected]})
    roles, generation, grading, rewrite = graph_models(grades=("yes", "no"))

    graph = build_finrag_graph(roles, max_rewrite_attempts=2, retriever=retriever)
    assert isinstance(graph, CompiledStateGraph)
    assert "web_search" not in graph.get_graph().nodes
    result = graph.invoke(initial_state())

    assert retriever.queries == [QUESTION]
    assert len(grading.calls) == 2
    assert len(generation.calls) == 1
    assert rewrite.calls == []
    assert [document.page_content for document in result["documents"]] == [relevant.page_content]
    assert rejected.page_content not in generation.calls[0].input
    assert result["rewrite_attempts"] == result["retry_count"] == 0
    assert result["numeric_verification"][0].status is VerificationStatus.SUPPORTED


def test_one_rewrite_retrieves_relevant_evidence_then_generates() -> None:
    initial = Document(page_content="Employee headcount was 10.")
    relevant = Document(page_content="Revenue was 50 billion VND.")
    rewritten = "FPT revenue evidence"
    retriever = RoutingRetriever({QUESTION: [initial], rewritten: [relevant]})
    roles, generation, grading, rewrite = graph_models(
        grades=("no", "yes"), rewrites=[f"  {rewritten}  "]
    )

    result = build_finrag_graph(roles, max_rewrite_attempts=2, retriever=retriever).invoke(
        initial_state()
    )

    assert retriever.queries == [QUESTION, rewritten]
    assert len(grading.calls) == 2
    assert rewrite.calls == [ModelCall("query_rewrite", f"Rewrite query: {QUESTION}")]
    assert len(generation.calls) == 1
    assert relevant.page_content in generation.calls[0].input
    assert initial.page_content not in generation.calls[0].input
    assert [document.page_content for document in result["documents"]] == [relevant.page_content]
    assert result["transformed_question"] == rewritten
    assert result["rewrite_attempts"] == result["retry_count"] == 1


def test_retry_exhaustion_stops_after_two_rewrites_and_uses_safe_empty_generation() -> None:
    rejected = Document(page_content="No relevant financial evidence.")
    retriever = RoutingRetriever({}, default=[rejected])
    roles, generation, grading, rewrite = graph_models(
        grades=("no",), rewrites=["rewrite one", "rewrite two"]
    )

    result = build_finrag_graph(roles, max_rewrite_attempts=2, retriever=retriever).invoke(
        initial_state()
    )

    assert retriever.queries == [QUESTION, "rewrite one", "rewrite two"]
    assert len(grading.calls) == 3
    assert len(rewrite.calls) == 2
    assert generation.calls == []
    assert result["documents"] == []
    assert result["generation"] == INSUFFICIENT_EVIDENCE
    assert result["citations"] == []
    assert result["numeric_verification"] == ()
    assert result["rewrite_attempts"] == result["retry_count"] == 2
    assert result["web_search_needed"] is False


def test_zero_rewrite_budget_terminates_without_rewrite_or_generation_model() -> None:
    rejected = Document(page_content="No relevant financial evidence.")
    retriever = RoutingRetriever({QUESTION: [rejected]})
    roles, generation, grading, rewrite = graph_models(grades=("no",))

    result = build_finrag_graph(roles, max_rewrite_attempts=0, retriever=retriever).invoke(
        initial_state()
    )

    assert retriever.queries == [QUESTION]
    assert len(grading.calls) == 1
    assert rewrite.calls == generation.calls == []
    assert result["generation"] == INSUFFICIENT_EVIDENCE
    assert result["rewrite_attempts"] == result["retry_count"] == 0


def test_identical_rewrites_still_terminate_at_per_execution_limit() -> None:
    rejected = Document(page_content="No relevant financial evidence.")
    retriever = RoutingRetriever({QUESTION: [rejected]})
    roles, generation, grading, rewrite = graph_models(
        grades=("no",), rewrites=[QUESTION, QUESTION]
    )
    graph = build_finrag_graph(roles, max_rewrite_attempts=2, retriever=retriever)

    first = graph.invoke(initial_state())
    second = graph.invoke(initial_state())

    assert retriever.queries == [QUESTION] * 6
    assert len(grading.calls) == 6
    assert len(rewrite.calls) == 4
    assert generation.calls == []
    assert first["generation"] == second["generation"] == INSUFFICIENT_EVIDENCE
    assert first["rewrite_attempts"] == second["rewrite_attempts"] == 2


def test_configured_budget_is_not_cut_off_by_langgraph_default_recursion_limit() -> None:
    rejected = Document(page_content="No relevant financial evidence.")
    retriever = RoutingRetriever({QUESTION: [rejected]})
    roles, generation, grading, rewrite = graph_models(grades=("no",), rewrites=[QUESTION])

    result = build_finrag_graph(roles, max_rewrite_attempts=8, retriever=retriever).invoke(
        initial_state()
    )

    assert len(retriever.queries) == len(grading.calls) == 9
    assert len(rewrite.calls) == 8
    assert generation.calls == []
    assert result["generation"] == INSUFFICIENT_EVIDENCE
    assert result["rewrite_attempts"] == 8


def test_numeric_contradiction_remains_terminal_and_does_not_rewrite() -> None:
    evidence = Document(page_content="Revenue was 50 billion VND.")
    retriever = RoutingRetriever({QUESTION: [evidence]})
    roles, generation, grading, rewrite = graph_models(
        answer="Revenue was 55 billion VND.", grades=("yes",)
    )

    result = build_finrag_graph(roles, max_rewrite_attempts=2, retriever=retriever).invoke(
        initial_state()
    )

    assert len(generation.calls) == len(grading.calls) == 1
    assert rewrite.calls == []
    assert result["numeric_verification"][0].status is VerificationStatus.CONTRADICTED
    assert result["is_hallucinated"] == "yes"
    assert result["rewrite_attempts"] == 0


@pytest.mark.parametrize("invalid_limit", [-1, True, 1.5])
def test_graph_builder_rejects_invalid_rewrite_limits(invalid_limit: object) -> None:
    roles, _, _, _ = graph_models()
    with pytest.raises(ValueError, match="non-negative integer"):
        build_finrag_graph(roles, max_rewrite_attempts=cast(int, invalid_limit))
