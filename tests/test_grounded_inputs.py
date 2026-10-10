"""Observable prompt-input regressions, not claims of live model accuracy."""

import pytest
from langchain_core.documents import Document

from app.graph.nodes import GradeDocuments, create_generate_node, create_grade_documents_node
from app.graph.state import FinGraphState
from tests.chat_models import WorkflowChatModel


def node_state(question: str, documents: list[Document]) -> FinGraphState:
    return {
        "question": question,
        "transformed_question": None,
        "documents": documents,
        "generation": "",
        "web_search_needed": False,
        "is_relevant": "yes",
        "is_hallucinated": "no",
        "is_useful": "yes",
        "retry_count": 0,
        "citations": [],
    }


def test_generation_receives_question_financial_text_and_available_metadata() -> None:
    text = "Revenue: 1,250 billion VND.\nMargin: 12.5%. Sales: USD 25 million."
    document = Document(
        page_content=text,
        metadata={"year": "2023", "source": "report.pdf", "page": 14, "company": "FPT"},
    )
    model = WorkflowChatModel(role="generation", responses=["Evidence-based test response."])
    result = create_generate_node(model)(node_state("What was the revenue?", [document]))

    assert result["generation"] == "Evidence-based test response."
    assert result["citations"][0]["source"] == "report.pdf"
    request = model.calls[0].input
    assert "What was the revenue?" in request
    assert text in request
    assert "[SOURCE 1]\nsource: report.pdf\npage: 14\ncompany: FPT\nyear: 2023" in request
    assert "EVIDENCE:\n" in request and "QUESTION:\n" in request
    assert "only the supplied evidence" in request
    assert "insufficient" in request and "do not invent facts" in request


def test_different_evidence_changes_generation_request_without_changing_question() -> None:
    model = WorkflowChatModel(responses=["local"])
    node = create_generate_node(model)
    for text in ("Revenue: VND 1,250 billion.", "Revenue: USD 25 million."):
        node(node_state("What was the revenue?", [Document(page_content=text)]))
    first, second = (call.input for call in model.calls)
    assert first != second
    assert "What was the revenue?" in first and "What was the revenue?" in second
    assert "VND 1,250 billion." in first and "USD 25 million." not in first
    assert "USD 25 million." in second and "VND 1,250 billion." not in second


def test_context_omits_missing_and_complex_metadata_without_fabricating_values() -> None:
    model = WorkflowChatModel(responses=["local"])
    document = Document(
        page_content="Supplied evidence.",
        metadata={"source": None, "page": None, "company": {"private": "not prompt text"}},
    )
    create_generate_node(model)(node_state("Question?", [document]))
    request = model.calls[0].input
    assert "[SOURCE 1]\ncontent:\nSupplied evidence." in request
    assert "source:" not in request and "page:" not in request and "company:" not in request
    assert "private" not in request and "Unknown" not in request and "N/A" not in request


def test_context_serialization_is_stable_and_preserves_document_order() -> None:
    model = WorkflowChatModel(responses=["local"])
    node = create_generate_node(model)
    for metadata in (
        {"page": 2, "source": "a.pdf", "has_table": True},
        {"has_table": True, "source": "a.pdf", "page": 2},
    ):
        node(
            node_state(
                "Question?",
                [
                    Document(page_content="First evidence.", metadata=metadata),
                    Document(page_content="Second evidence.", metadata={"source": "b.pdf"}),
                ],
            )
        )
    assert model.calls[0].input == model.calls[1].input
    request = model.calls[0].input
    assert request.index("[SOURCE 1]") < request.index("First evidence.")
    assert request.index("First evidence.") < request.index("[SOURCE 2]")
    assert request.index("[SOURCE 2]") < request.index("Second evidence.")
    assert "has_table: True" in request


@pytest.mark.parametrize("texts", [[], [""], [" \n\t"]])
def test_empty_evidence_returns_explicit_insufficiency_without_model_call(texts: list[str]) -> None:
    model = WorkflowChatModel(responses=["An unsupported answer must never be returned."])
    documents = [Document(page_content=text, metadata={"source": "empty.pdf"}) for text in texts]
    result = create_generate_node(model)(node_state("What was the revenue?", documents))
    assert result["generation"] == (
        "Không đủ bằng chứng từ tài liệu được cung cấp để trả lời câu hỏi này."
    )
    assert result["citations"] == []
    assert model.calls == []


def test_grading_receives_question_and_candidate_with_existing_schema() -> None:
    document = Document(page_content="Revenue: VND 1,250 billion.\nMargin: 12.5%.")
    model = WorkflowChatModel(role="grading", responses=["unused"])
    node = create_grade_documents_node(model)
    for question in ("What was the revenue?", "What was the margin?"):
        result = node(node_state(question, [document]))
        assert result["documents"] == [document]
        assert result["is_relevant"] == "yes"
        assert result["web_search_needed"] is False
    first, second = model.calls
    assert first.schema is second.schema is GradeDocuments
    assert first.role == second.role == "grading"
    assert first.input != second.input
    assert "QUESTION:\nWhat was the revenue?" in first.input
    assert "QUESTION:\nWhat was the margin?" in second.input
    for call in model.calls:
        assert f"CANDIDATE DOCUMENT:\n{document.page_content}" in call.input
        assert "relevant to answering" in call.input


def test_each_candidate_is_graded_individually_with_question_and_negative_grade() -> None:
    documents = [
        Document(page_content="Candidate Alpha."),
        Document(page_content="Candidate Beta."),
    ]
    model = WorkflowChatModel(role="grading", responses=["unused"], binary_score="no")
    result = create_grade_documents_node(model)(node_state("What was the revenue?", documents))
    assert len(model.calls) == 2
    assert (
        "Candidate Alpha." in model.calls[0].input and "Candidate Beta." not in model.calls[0].input
    )
    assert (
        "Candidate Beta." in model.calls[1].input and "Candidate Alpha." not in model.calls[1].input
    )
    assert all("What was the revenue?" in call.input for call in model.calls)
    assert all(call.schema is GradeDocuments for call in model.calls)
    # Preserve existing policy: retain original docs if all candidates are rejected.
    assert result["documents"] == documents
    assert result["is_relevant"] == "no"
    assert result["web_search_needed"] is True
