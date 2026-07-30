import os
import re
from typing import List, Dict, Any

try:
    from langchain_core.documents import Document
except ImportError:
    class Document:
        def __init__(self, page_content: str, metadata: dict = None):
            self.page_content = page_content
            self.metadata = metadata or {}

class FinancialPDFParser:
    """Specialized PDF & Document Parser for Financial Annual Reports and Earnings Filings."""
    
    def __init__(self, chunk_size: int = 800, chunk_overlap: int = 150):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def parse_pdf(self, file_path: str, company_name: str = "FPT Corporation", year: str = "2023") -> List[Document]:
        """Reads PDF using pdfplumber / pypdf, extracts text + tables, and returns enriched Documents."""
        docs = []
        
        try:
            import pdfplumber
            with pdfplumber.open(file_path) as pdf:
                for page_idx, page in enumerate(pdf.pages):
                    page_num = page_idx + 1
                    text = page.extract_text() or ""
                    
                    tables = page.extract_tables()
                    table_md_str = ""
                    if tables:
                        for tbl in tables:
                            table_md_str += "\n\n" + self._convert_table_to_markdown(tbl)
                    
                    combined_content = text + table_md_str
                    if combined_content.strip():
                        docs.append(Document(
                            page_content=combined_content,
                            metadata={
                                "source": os.path.basename(file_path),
                                "page": page_num,
                                "company": company_name,
                                "year": year,
                                "has_table": bool(tables)
                            }
                        ))
        except Exception as e:
            print(f"[Warning] pdfplumber fallback notice ({e}).")
            if os.path.exists(file_path):
                with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                    content = f.read()
                docs.append(Document(
                    page_content=content,
                    metadata={"source": os.path.basename(file_path), "company": company_name, "year": year, "has_table": False}
                ))
                
        return self._split_documents(docs)

    def _convert_table_to_markdown(self, table: List[List[Any]]) -> str:
        """Converts raw pdfplumber table rows into clean Markdown Table format."""
        if not table or not any(table):
            return ""
        
        md_lines = []
        header = [str(cell or "").replace("\n", " ").strip() for cell in table[0]]
        md_lines.append("| " + " | ".join(header) + " |")
        md_lines.append("| " + " | ".join(["---"] * len(header)) + " |")
        
        for row in table[1:]:
            clean_row = [str(cell or "").replace("\n", " ").strip() for cell in row]
            if any(clean_row):
                md_lines.append("| " + " | ".join(clean_row) + " |")
                
        return "\n".join(md_lines)

    def parse_text_or_markdown(self, content: str, source_name: str, company_name: str = "FPT Corporation", year: str = "2023") -> List[Document]:
        """Parses raw markdown / text financial report content directly."""
        doc = Document(
            page_content=content,
            metadata={
                "source": source_name,
                "company": company_name,
                "year": year,
                "has_table": "|" in content
            }
        )
        return self._split_documents([doc])

    def _split_documents(self, docs: List[Document]) -> List[Document]:
        """Simple fallback character splitter if langchain text_splitter is not installed."""
        chunks = []
        for doc in docs:
            content = doc.page_content
            if len(content) <= self.chunk_size:
                chunks.append(doc)
            else:
                for i in range(0, len(content), self.chunk_size - self.chunk_overlap):
                    chunk_text = content[i:i + self.chunk_size]
                    chunks.append(Document(page_content=chunk_text, metadata=dict(doc.metadata)))
        return chunks
