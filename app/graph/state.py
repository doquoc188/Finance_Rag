from typing import Any, Dict, List, Optional

from typing_extensions import NotRequired, TypedDict

from finrag.verification import NumericVerification

try:
    from langchain_core.documents import Document
except ImportError:
    from app.core.pdf_parser import Document


class FinGraphState(TypedDict):
    """
    State dictionary representing the complete memory and lifecycle of the Self-Corrective Financial RAG Graph.
    """

    question: str  # Original user question
    transformed_question: Optional[str]  # Rewritten query optimized for search
    documents: List[Document]  # Retrieved financial document chunks
    generation: str  # Final generated response text
    web_search_needed: bool  # Flag whether web search was triggered
    is_relevant: str  # "yes" or "no" (grade of documents)
    is_hallucinated: str  # Legacy name: numeric contradiction, not general hallucination
    is_useful: str  # Legacy name: numeric coverage, not a general usefulness grade
    retry_count: int  # Number of transform query retries (prevent infinite loops)
    citations: List[Dict[str, Any]]  # Structured citation metadata (File, Page, Company, Year)
    numeric_verification: NotRequired[tuple[NumericVerification, ...]]


class FinGraphUpdate(TypedDict, total=False):
    """Partial node updates / initial input; existing state fields stay unchanged."""

    question: str
    transformed_question: str | None
    documents: list[Document]
    generation: str
    web_search_needed: bool
    is_relevant: str
    is_hallucinated: str
    is_useful: str
    retry_count: int
    citations: list[dict[str, Any]]
    numeric_verification: tuple[NumericVerification, ...]
