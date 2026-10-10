import os
import shutil
from typing import cast

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile

from app.api.schemas import ChatRequest, ChatResponse, CitationSchema, IngestResponse
from app.config import settings
from app.core.pdf_parser import FinancialPDFParser
from app.graph.state import FinGraphUpdate
from app.runtime import GraphRuntime

router = APIRouter(prefix="/api/v1", tags=["Financial RAG"])


@router.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest, http_request: Request) -> ChatResponse:
    """Processes user question through the Self-Corrective LangGraph Engine."""
    try:
        initial_state: FinGraphUpdate = {
            "question": request.question,
            "rewrite_attempts": 0,
            "retry_count": 0,
            "web_search_needed": False,
        }

        # Execute LangGraph workflow
        runtime: GraphRuntime = http_request.app.state.runtime
        result = runtime.graph.invoke(initial_state)

        citations = [CitationSchema(**c) for c in result.get("citations", [])]

        return ChatResponse(
            question=request.question,
            answer=result.get("generation", "Không thể tạo câu trả lời."),
            citations=citations,
            execution_info={
                "is_relevant": result.get("is_relevant", "yes"),
                "web_search_needed": result.get("web_search_needed", False),
                "transformed_question": result.get("transformed_question"),
                "retry_count": result.get("retry_count", 0),
            },
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error executing FinRAG engine: {str(e)}")


@router.post("/ingest", response_model=IngestResponse)
async def ingest_document(
    http_request: Request,
    file: UploadFile = File(...),
    company_name: str = Form("FPT Corporation"),
    year: str = Form("2023"),
) -> IngestResponse:
    """Uploads a Financial Report PDF, parses tables & indexes into Qdrant & BM25."""
    try:
        temp_dir = os.path.join(os.getcwd(), "data", "uploads")
        os.makedirs(temp_dir, exist_ok=True)
        # Type-only annotation of the existing upload assumption; validation is deferred.
        filename = cast(str, file.filename)
        file_path = os.path.join(temp_dir, filename)

        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        parser = FinancialPDFParser()
        docs = parser.parse_pdf(file_path, company_name=company_name, year=year)

        runtime: GraphRuntime = http_request.app.state.runtime
        runtime.retriever.index_documents(docs)

        return IngestResponse(
            status="success",
            filename=filename,
            chunks_created=len(docs),
            message=f"Đã nạp thành công {len(docs)} đoạn tài liệu tài chính vào Qdrant & BM25 Vector DB.",
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error ingesting PDF: {str(e)}")


@router.get("/health")
async def health_check(http_request: Request) -> dict[str, str]:
    """Health check endpoint."""
    runtime: GraphRuntime = http_request.app.state.runtime
    return {
        "status": "healthy",
        "app_name": settings.APP_NAME,
        "llm_provider": runtime.settings.generation.provider,
        "embedding_model": settings.EMBEDDING_MODEL,
        "vector_db": settings.QDRANT_LOCATION,
    }
