from langchain_core.documents import Document
from langgraph.graph.state import CompiledStateGraph

from app.core.pdf_parser import FinancialPDFParser
from app.core.vectorstore import HybridFinancialRetriever
from app.graph.state import FinGraphUpdate
from app.graph.workflow import build_finrag_graph
from finrag.models import ModelRoles
from tests.chat_models import WorkflowChatModel


class FinanceEmbeddings:
    def __init__(self) -> None:
        self.document_calls: list[list[str]] = []
        self.query_calls: list[str] = []

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        self.document_calls.append(texts)
        return [[1.0, 0.0] if "FPT" in text else [0.0, 1.0] for text in texts]

    def embed_query(self, text: str) -> list[float]:
        self.query_calls.append(text)
        return [1.0, 0.0] if "FPT" in text else [0.0, 1.0]


def test_financial_pdf_parser_markdown() -> None:
    parser = FinancialPDFParser()
    sample_content = """# Báo cáo FPT 2023
| Chỉ tiêu | 2023 |
| Doanh thu | 52.618 |
"""
    docs = parser.parse_text_or_markdown(
        sample_content, source_name="test.md", company_name="FPT", year="2023"
    )
    assert len(docs) > 0
    assert docs[0].metadata["company"] == "FPT"
    assert docs[0].metadata["has_table"] is True


def test_hybrid_retriever() -> None:
    docs = [
        Document(
            page_content="FPT đạt doanh thu 52.618 tỷ đồng năm 2023.",
            metadata={"source": "fpt.pdf", "page": 1},
        ),
        Document(
            page_content="Vinamilk đạt doanh thu 60.369 tỷ đồng năm 2023.",
            metadata={"source": "vnm.pdf", "page": 1},
        ),
    ]
    embeddings = FinanceEmbeddings()
    retriever = HybridFinancialRetriever(documents=docs, embeddings=embeddings)
    results = retriever.get_relevant_documents("doanh thu FPT", top_k=1)
    assert len(results) == 1
    assert "FPT" in results[0].page_content
    assert embeddings.document_calls == [[document.page_content for document in docs]]
    assert embeddings.query_calls == ["doanh thu FPT"]
    assert results[0].metadata["retrieval_strategy"] == "hybrid"
    assert results[0].metadata["retrieval_score_type"] == "rrf"
    assert results[0].metadata["retrieval_score"] == 2 / 61


def test_repeated_ordered_corpus_keeps_equal_text_chunks_distinct_and_stable() -> None:
    documents = [
        Document(page_content="FPT EPS 2025", metadata={"source": "fpt.pdf", "page": 1}),
        Document(page_content="FPT EPS 2025", metadata={"source": "fpt.pdf", "page": 1}),
    ]
    retriever = HybridFinancialRetriever(documents=documents, embeddings=FinanceEmbeddings())

    first_ids = {
        result.metadata["chunk_id"]
        for result in retriever.get_relevant_documents("FPT EPS 2025", top_k=2)
    }
    retriever.index_documents(documents)
    second_ids = {
        result.metadata["chunk_id"]
        for result in retriever.get_relevant_documents("FPT EPS 2025", top_k=2)
    }

    assert len(first_ids) == 2
    assert second_ids == first_ids


def test_finrag_langgraph_invocation(workflow_chat_model: WorkflowChatModel) -> None:
    finrag_app = build_finrag_graph(
        ModelRoles(workflow_chat_model, workflow_chat_model, workflow_chat_model)
    )
    assert isinstance(finrag_app, CompiledStateGraph)
    initial_state: FinGraphUpdate = {
        "question": "Doanh thu năm 2023 của FPT là bao nhiêu?",
        "rewrite_attempts": 0,
        "retry_count": 0,
        "web_search_needed": False,
    }
    result = finrag_app.invoke(initial_state)
    assert "generation" in result
    assert result["generation"] is not None
    assert result["generation"] == "Offline workflow test response."
    assert result["question"] == initial_state["question"]
    assert result["is_relevant"] == "yes"
    assert result["retry_count"] == 0
    assert result["web_search_needed"] is False
    assert result["is_hallucinated"] == "no"
    assert result["is_useful"] == "yes"
    assert result["citations"]
    assert result["citations"][0]["source"] == result["documents"][0].metadata["source"]
    assert result["citations"][0]["page"] == result["documents"][0].metadata["page"]
