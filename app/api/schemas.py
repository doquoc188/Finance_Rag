from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

class ChatRequest(BaseModel):
    question: str = Field(..., example="Doanh thu và lợi nhuận trước thuế năm 2023 của FPT là bao nhiêu?")
    company_name: Optional[str] = Field(default="FPT Corporation")
    year: Optional[str] = Field(default="2023")

class CitationSchema(BaseModel):
    source: str
    page: int
    company: str
    year: str
    snippet: str

class ChatResponse(BaseModel):
    question: str
    answer: str
    citations: List[CitationSchema]
    execution_info: Dict[str, Any]

class IngestResponse(BaseModel):
    status: str
    filename: str
    chunks_created: int
    message: str
