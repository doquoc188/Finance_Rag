"""Real fact-check node and compiled graph, with explicit local upstream fixtures."""

from decimal import Decimal

import pytest
from langchain_core.documents import Document
from langgraph.graph.state import CompiledStateGraph

from app.graph.nodes import grade_generation_node
from app.graph.workflow import build_finrag_graph
from finrag.models import ModelRoles
from finrag.verification import VerificationStatus
from tests.chat_models import WorkflowChatModel
from tests.retrievers import StaticRetriever
from tests.test_grounded_inputs import node_state


@pytest.mark.parametrize(
    ("answer", "status", "hallucinated", "useful"),
    [
        ("Revenue was 50 billion VND.", VerificationStatus.SUPPORTED, "no", "yes"),
        ("Revenue was 55 billion VND.", VerificationStatus.CONTRADICTED, "yes", "no"),
        ("Profit was 55 billion VND.", VerificationStatus.NOT_FOUND, "no", "no"),
    ],
)
def test_real_compiled_graph_propagates_numeric_report_and_conservative_flags(
    answer: str,
    status: VerificationStatus,
    hallucinated: str,
    useful: str,
) -> None:
    document = Document(
        page_content="Revenue reached VND 50 billion.",
        metadata={"source": "local-report.txt", "page": 2, "company": "Test", "year": "2025"},
    )
    generation = WorkflowChatModel(role="generation", responses=[answer])
    grading = WorkflowChatModel(role="grading", responses=["unused"])
    rewrite = WorkflowChatModel(role="query_rewrite", responses=["must not be called"])
    graph = build_finrag_graph(
        ModelRoles(generation, grading, rewrite), retriever=StaticRetriever([document])
    )
    assert isinstance(graph, CompiledStateGraph)
    result = graph.invoke(
        {
            "question": "What was revenue?",
            "rewrite_attempts": 0,
            "retry_count": 0,
            "web_search_needed": False,
        }
    )
    assert result["generation"] == answer
    (verification,) = result["numeric_verification"]
    assert verification.status is status
    assert verification.mention.raw in answer
    assert isinstance(verification.mention.value, Decimal)
    if status is not VerificationStatus.NOT_FOUND:
        assert verification.evidence[0].sentence == document.page_content
        assert (
            result["documents"][verification.evidence[0].evidence_index].page_content
            == document.page_content
        )
    assert result["is_hallucinated"] == hallucinated
    assert result["is_useful"] == useful
    assert result["retry_count"] == 0
    assert result["rewrite_attempts"] == 0
    assert result["web_search_needed"] is False
    assert len(generation.calls) == len(grading.calls) == 1
    assert rewrite.calls == []


def test_fact_check_node_has_independent_claim_results_and_calls_no_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def reject_chat_call(*args: object, **kwargs: object) -> None:
        raise AssertionError("Numeric verification must not invoke a model")

    monkeypatch.setattr(WorkflowChatModel, "invoke", reject_chat_call)
    state = node_state("Revenue and margin?", [Document(page_content="Revenue: 50 billion VND.")])
    state["generation"] = "Revenue was 50 billion VND and margin was 12.5%."
    result = grade_generation_node(state)
    assert [claim.status for claim in result["numeric_verification"]] == [
        VerificationStatus.SUPPORTED,
        VerificationStatus.NOT_FOUND,
    ]
    assert result["is_hallucinated"] == "no"
    assert result["is_useful"] == "no"


def test_numeric_free_answer_has_empty_report_without_truth_certification() -> None:
    state = node_state("Question?", [])
    state["generation"] = "Không đủ bằng chứng để trả lời."
    result = grade_generation_node(state)
    assert result["numeric_verification"] == ()
    assert result["is_hallucinated"] == "no"
    assert result["is_useful"] == "yes"  # Compatibility, not an actual usefulness grade.
