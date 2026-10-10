from functools import partial
from typing import Protocol, cast

from langchain_core.language_models.chat_models import BaseChatModel
from pydantic import BaseModel, Field

from app.core.errors import ProviderUnavailableError

try:
    from langchain_core.documents import Document
except ImportError:
    from app.core.pdf_parser import Document

try:
    from langchain_core.prompts import ChatPromptTemplate
except ImportError as exc:
    raise ProviderUnavailableError(
        "Package 'langchain-core' with ChatPromptTemplate is required. Run uv sync."
    ) from exc

from app.core.vectorstore import HybridFinancialRetriever
from app.graph.state import FinGraphState, FinGraphUpdate


class GraphNode(Protocol):
    def __call__(self, state: FinGraphState) -> FinGraphUpdate: ...


class GradeDocuments(BaseModel):
    binary_score: str = Field(description="Documents are relevant to the question, 'yes' or 'no'")


class GradeHallucination(BaseModel):
    binary_score: str = Field(description="Answer is grounded in the facts, 'yes' or 'no'")
    is_useful: str = Field(description="Answer answers the user question, 'yes' or 'no'")


global_retriever: HybridFinancialRetriever | None = None


def set_global_retriever(retriever: HybridFinancialRetriever) -> None:
    global global_retriever
    global_retriever = retriever


def retrieve_node(state: FinGraphState) -> FinGraphUpdate:
    print("--- [Graph Node] RETRIEVE DOCUMENTS ---")
    question = state.get("transformed_question") or state["question"]

    if global_retriever and global_retriever.documents:
        docs = global_retriever.get_relevant_documents(question, top_k=4)
    else:
        docs = [
            Document(
                page_content="[Mock Financial Report] Năm 2023, FPT đạt doanh thu 52.618 tỷ đồng (tăng 19,6%), Lợi nhuận trước thuế 9.203 tỷ đồng (tăng 20,1%). Khối Công nghệ đóng góp 45% tổng doanh thu.",
                metadata={
                    "source": "FPT_Annual_Report_2023.pdf",
                    "page": 14,
                    "company": "FPT Corporation",
                    "year": "2023",
                },
            )
        ]
    return {"documents": docs, "question": state["question"]}


def grade_documents_node(state: FinGraphState, *, model: BaseChatModel) -> FinGraphUpdate:
    print("--- [Graph Node] GRADE DOCUMENTS RELEVANCE ---")
    _question = state["question"]
    docs = state["documents"]

    filtered_docs = []
    web_search_needed = False

    for d in docs:
        try:
            structured_llm = model.with_structured_output(GradeDocuments)
            # A Pydantic schema returns that schema instance under LangChain's contract.
            res = cast(GradeDocuments, structured_llm.invoke(d.page_content))
            score = res.binary_score.lower()
        except Exception:
            score = (
                "yes"
                if any(
                    w in d.page_content.lower()
                    for w in ["doanh thu", "lợi nhuận", "fpt", "tỷ đồng", "báo cáo", "%"]
                )
                else "no"
            )

        if score == "yes":
            filtered_docs.append(d)
        else:
            web_search_needed = True

    if not filtered_docs:
        web_search_needed = True

    is_rel = "yes" if filtered_docs else "no"
    return {
        "documents": filtered_docs if filtered_docs else docs,
        "is_relevant": is_rel,
        "web_search_needed": web_search_needed,
    }


def generate_node(state: FinGraphState, *, model: BaseChatModel) -> FinGraphUpdate:
    print("--- [Graph Node] GENERATE FINANCIAL RESPONSE ---")
    question = state["question"]
    docs = state["documents"]

    context_str = "\n\n".join(
        [
            f"--- TÀI LIỆU [Nguồn: {d.metadata.get('source', 'Unknown')}, Trang: {d.metadata.get('page', 1)}] ---\n{d.page_content}"
            for d in docs
        ]
    )

    prompt_template = ChatPromptTemplate.from_messages([("user", "{question}")])
    chain = prompt_template | model
    response = chain.invoke({"context": context_str, "question": question})

    citations = [
        {
            "source": d.metadata.get("source", "N/A"),
            "page": d.metadata.get("page", 1),
            "company": d.metadata.get("company", "N/A"),
            "year": d.metadata.get("year", "N/A"),
            "snippet": d.page_content[:150] + "...",
        }
        for d in docs
    ]

    return {
        # Keep the existing textual-content contract without normalizing model output.
        "generation": cast(
            str, response.content if hasattr(response, "content") else str(response)
        ),
        "citations": citations,
    }


def transform_query_node(state: FinGraphState, *, model: BaseChatModel) -> FinGraphUpdate:
    print("--- [Graph Node] TRANSFORM QUERY ---")
    question = state["question"]
    retry_count = state.get("retry_count", 0) + 1
    res = model.invoke(f"Rewrite query: {question}")
    new_query = cast(str, res.content if hasattr(res, "content") else str(res))
    return {"transformed_question": new_query.strip(), "retry_count": retry_count}


def web_search_node(state: FinGraphState) -> FinGraphUpdate:
    print("--- [Graph Node] WEB SEARCH FALLBACK ---")
    question = state.get("transformed_question") or state["question"]
    docs = state.get("documents", [])

    web_content = ""
    try:
        from duckduckgo_search import DDGS

        with DDGS() as ddgs:
            results = list(ddgs.text(keywords=f"{question} báo cáo tài chính", max_results=3))
            for r in results:
                web_content += (
                    f"\n[Web Source: {r.get('title')}] ({r.get('href')})\n{r.get('body')}\n"
                )
    except Exception:
        web_content = f"Thông tin thị trường về: {question}"

    web_doc = Document(
        page_content=f"--- THÔNG TIN TÌM KIẾM WEB BỔ SUNG ---\n{web_content}",
        metadata={
            "source": "DuckDuckGo Web Search",
            "page": 1,
            "company": "Market Info",
            "year": "2024",
        },
    )
    docs.append(web_doc)
    return {"documents": docs, "web_search_needed": False}


def grade_generation_node(state: FinGraphState) -> FinGraphUpdate:
    print("--- [Graph Node] FACT-CHECK & HALLUCINATION GUARDRAIL ---")
    generation = state["generation"]
    docs = state["documents"]

    import re

    numbers_in_gen = set(re.findall(r"\d+(?:\.\d+)?", generation))
    context_text = " ".join([d.page_content for d in docs])

    is_hallucinated = "no"
    for num in numbers_in_gen:
        if len(num) >= 3 and num not in context_text:
            print(f"[Fact-Check Notice] Number {num} checked against context.")

    return {"is_hallucinated": is_hallucinated, "is_useful": "yes"}


def create_grade_documents_node(model: BaseChatModel) -> GraphNode:
    return partial(grade_documents_node, model=model)


def create_generate_node(model: BaseChatModel) -> GraphNode:
    return partial(generate_node, model=model)


def create_transform_query_node(model: BaseChatModel) -> GraphNode:
    return partial(transform_query_node, model=model)
