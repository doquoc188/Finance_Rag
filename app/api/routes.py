import os
import shutil
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from typing import Optional

from app.api.schemas import ChatRequest, ChatResponse, IngestResponse, CitationSchema
from app.graph.workflow import finrag_app
from app.graph.nodes import set_global_retriever
from app.core.pdf_parser import FinancialPDFParser
from app.core.vectorstore import HybridFinancialRetriever
from app.config import settings

router = APIRouter(prefix="/api/v1", tags=["Financial RAG"])

# Active global retriever instance
retriever_instance: Optional[HybridFinancialRetriever] = None

@router.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """Processes user question through the Self-Corrective LangGraph Engine."""
    try:
        initial_state = {
            "question": request.question,
            "retry_count": 0,
            "web_search_needed": False
        }
        
        # Execute LangGraph workflow
        result = finrag_app.invoke(initial_state)
        
        citations = [CitationSchema(**c) for c in result.get("citations", [])]
        
        return ChatResponse(
            question=request.question,
            answer=result.get("generation", "Không thể tạo câu trả lời."),
            citations=citations,
            execution_info={
                "is_relevant": result.get("is_relevant", "yes"),
                "web_search_needed": result.get("web_search_needed", False),
                "transformed_question": result.get("transformed_question"),
                "retry_count": result.get("retry_count", 0)
            }
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error executing FinRAG engine: {str(e)}")

@router.post("/ingest", response_model=IngestResponse)
async def ingest_document(
    file: UploadFile = File(...),
    company_name: str = Form("FPT Corporation"),
    year: str = Form("2023")
):
    """Uploads a Financial Report PDF, parses tables & indexes into Qdrant & BM25."""
    global retriever_instance
    try:
        temp_dir = os.path.join(os.getcwd(), "data", "uploads")
        os.makedirs(temp_dir, exist_ok=True)
        file_path = os.path.join(temp_dir, file.filename)
        
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        parser = FinancialPDFParser()
        docs = parser.parse_pdf(file_path, company_name=company_name, year=year)
        
        if not retriever_instance:
            retriever_instance = HybridFinancialRetriever(documents=docs)
        else:
            retriever_instance.index_documents(docs)
            
        set_global_retriever(retriever_instance)
        
        return IngestResponse(
            status="success",
            filename=file.filename,
            chunks_created=len(docs),
            message=f"Đã nạp thành công {len(docs)} đoạn tài liệu tài chính vào Qdrant & BM25 Vector DB."
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error ingesting PDF: {str(e)}")

@router.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "app_name": settings.APP_NAME,
        "llm_provider": settings.LLM_PROVIDER,
        "embedding_model": settings.EMBEDDING_MODEL,
        "vector_db": settings.QDRANT_LOCATION
    }
