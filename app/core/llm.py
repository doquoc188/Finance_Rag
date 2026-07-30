import os
from typing import Any
from app.config import settings

class SimpleFakeLLM:
    """Fallback LLM for development when external APIs / libraries are not yet loaded."""
    def __init__(self, responses: list = None):
        self.responses = responses or [
            "Theo Báo cáo thường niên 2023 của Tập đoàn FPT, Doanh thu hợp nhất đạt 52.618 tỷ đồng (tăng 19,6%) và Lợi nhuận trước thuế đạt 9.203 tỷ đồng (tăng 20,1%). [Nguồn: FPT_BaoCaoThuongNien_2023.md, Trang: 14]"
        ]
        self._idx = 0

    def invoke(self, input_data: Any) -> Any:
        resp_text = self.responses[self._idx % len(self.responses)]
        self._idx += 1
        class ResponseContent:
            def __init__(self, content):
                self.content = content
        return ResponseContent(resp_text)

    def with_structured_output(self, schema_cls: Any):
        class StructuredModel:
            def invoke(self, input_data: Any):
                class Score:
                    binary_score = "yes"
                    is_useful = "yes"
                return Score()
        return StructuredModel()

def get_embeddings() -> Any:
    """Returns Embeddings model with graceful fallback."""
    try:
        if settings.OPENAI_API_KEY and not settings.OPENAI_API_KEY.startswith("mock"):
            from langchain_openai import OpenAIEmbeddings
            return OpenAIEmbeddings(model="text-embedding-3-small", openai_api_key=settings.OPENAI_API_KEY)
        else:
            from langchain_community.embeddings import HuggingFaceEmbeddings
            return HuggingFaceEmbeddings(model_name=settings.EMBEDDING_MODEL)
    except Exception:
        class DummyEmbeddings:
            def embed_documents(self, texts):
                return [[0.1] * 128 for _ in texts]
            def embed_query(self, text):
                return [0.1] * 128
        return DummyEmbeddings()

def get_llm(temperature: float = 0.0) -> Any:
    """Factory function to get LLM instance based on settings."""
    provider = settings.LLM_PROVIDER.lower()
    try:
        if provider == "openai" and settings.OPENAI_API_KEY and not settings.OPENAI_API_KEY.startswith("mock"):
            from langchain_openai import ChatOpenAI
            return ChatOpenAI(model=settings.OPENAI_MODEL, temperature=temperature, api_key=settings.OPENAI_API_KEY)
        elif provider == "groq" and settings.GROQ_API_KEY:
            from langchain_groq import ChatGroq
            return ChatGroq(model=settings.GROQ_MODEL, temperature=temperature, groq_api_key=settings.GROQ_API_KEY)
        elif provider == "ollama":
            from langchain_community.chat_models import ChatOllama
            return ChatOllama(base_url=settings.OLLAMA_BASE_URL, model=settings.OLLAMA_MODEL, temperature=temperature)
        else:
            return SimpleFakeLLM()
    except Exception:
        return SimpleFakeLLM()
