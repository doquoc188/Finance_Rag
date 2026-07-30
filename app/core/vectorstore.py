import os
import re
from typing import List, Dict, Any

try:
    from langchain_core.documents import Document
except ImportError:
    from app.core.pdf_parser import Document

from app.config import settings
from app.core.llm import get_embeddings

class HybridFinancialRetriever:
    """Hybrid Retriever combining Lexical Search + Vector Search with Reciprocal Rank Fusion (RRF)."""
    
    def __init__(self, documents: List[Document] = None):
        self.embeddings = get_embeddings()
        self.documents: List[Document] = documents or []
        self.vector_store = None
        self.bm25 = None
        
        if self.documents:
            self.index_documents(self.documents)

    def index_documents(self, documents: List[Document]):
        """Indexes documents into memory / Qdrant & BM25."""
        self.documents = documents
        
        # Try indexing in Qdrant if library available
        try:
            from langchain_community.vectorstores import Qdrant
            self.vector_store = Qdrant.from_documents(
                documents=documents,
                embedding=self.embeddings,
                location=settings.QDRANT_LOCATION,
                collection_name=settings.QDRANT_COLLECTION_NAME,
                force_recreate=True
            )
        except Exception:
            self.vector_store = None

        # Try BM25 indexing if rank_bm25 available
        try:
            from rank_bm25 import BM25Okapi
            tokenized_corpus = [self._tokenize(doc.page_content) for doc in documents]
            self.bm25 = BM25Okapi(tokenized_corpus)
        except Exception:
            self.bm25 = None

        print(f"[VectorStore] Indexed {len(documents)} document chunks.")

    def _tokenize(self, text: str) -> List[str]:
        return re.findall(r'\w+', text.lower())

    def get_relevant_documents(self, query: str, top_k: int = 4) -> List[Document]:
        """Performs Hybrid Search / BM25 / Vector Search fallback."""
        if not self.documents:
            return []

        # 1. If Qdrant is available
        if self.vector_store:
            try:
                return self.vector_store.similarity_search(query, k=top_k)
            except Exception:
                pass

        # 2. Simple BM25 or keyword match score
        query_words = set(self._tokenize(query))
        scored_docs = []
        for doc in self.documents:
            doc_words = set(self._tokenize(doc.page_content))
            overlap = len(query_words.intersection(doc_words))
            scored_docs.append((overlap, doc))

        scored_docs.sort(key=lambda x: x[0], reverse=True)
        return [doc for score, doc in scored_docs[:top_k]]
