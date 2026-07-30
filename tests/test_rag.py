import pytest
from langchain_core.documents import Document
from app.core.pdf_parser import FinancialPDFParser
from app.core.vectorstore import HybridFinancialRetriever
from app.graph.workflow import finrag_app

def test_financial_pdf_parser_markdown():
    parser = FinancialPDFParser()
    sample_content = """# Báo cáo FPT 2023
| Chỉ tiêu | 2023 |
| Doanh thu | 52.618 |
"""
    docs = parser.parse_text_or_markdown(sample_content, source_name="test.md", company_name="FPT", year="2023")
    assert len(docs) > 0
    assert docs[0].metadata["company"] == "FPT"
    assert docs[0].metadata["has_table"] is True

def test_hybrid_retriever():
    docs = [
        Document(page_content="FPT đạt doanh thu 52.618 tỷ đồng năm 2023.", metadata={"source": "fpt.pdf", "page": 1}),
        Document(page_content="Vinamilk đạt doanh thu 60.369 tỷ đồng năm 2023.", metadata={"source": "vnm.pdf", "page": 1}),
    ]
    retriever = HybridFinancialRetriever(documents=docs)
    results = retriever.get_relevant_documents("doanh thu FPT", top_k=1)
    assert len(results) == 1
    assert "FPT" in results[0].page_content

def test_finrag_langgraph_invocation():
    initial_state = {
        "question": "Doanh thu năm 2023 của FPT là bao nhiêu?",
        "retry_count": 0,
        "web_search_needed": False
    }
    result = finrag_app.invoke(initial_state)
    assert "generation" in result
    assert result["generation"] is not None
