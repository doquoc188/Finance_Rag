from typing import List, Dict, Any, Optional
from typing_extensions import TypedDict

try:
    from langchain_core.documents import Document
except ImportError:
    from app.core.pdf_parser import Document

class FinGraphState(TypedDict):
    """
    State dictionary representing the complete memory and lifecycle of the Self-Corrective Financial RAG Graph.
    """
    question: str                         # Original user question
    transformed_question: Optional[str]   # Rewritten query optimized for search
    documents: List[Document]             # Retrieved financial document chunks
    generation: str                       # Final generated response text
    web_search_needed: bool               # Flag whether web search was triggered
    is_relevant: str                      # "yes" or "no" (grade of documents)
    is_hallucinated: str                  # "yes" or "no" (grade of answer faithfulness)
    is_useful: str                        # "yes" or "no" (grade of answer usefulness)
    retry_count: int                      # Number of transform query retries (prevent infinite loops)
    citations: List[Dict[str, Any]]       # Structured citation metadata (File, Page, Company, Year)
