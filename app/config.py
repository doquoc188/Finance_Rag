import os

try:
    from pydantic_settings import BaseSettings
    from pydantic import Field

    class Settings(BaseSettings):
        APP_NAME: str = "FinRAG Analyst Agent"
        ENV: str = "development"
        DEBUG: bool = True

        LLM_PROVIDER: str = Field(default="openai")
        OPENAI_API_KEY: str = Field(default="")
        OPENAI_MODEL: str = Field(default="gpt-4o-mini")
        
        GROQ_API_KEY: str = Field(default="")
        GROQ_MODEL: str = Field(default="llama-3.1-70b-versatile")
        
        OLLAMA_BASE_URL: str = Field(default="http://localhost:11434")
        OLLAMA_MODEL: str = Field(default="qwen2.5:7b")

        EMBEDDING_MODEL: str = Field(default="sentence-transformers/all-MiniLM-L6-v2")

        QDRANT_LOCATION: str = Field(default=":memory:")
        QDRANT_COLLECTION_NAME: str = Field(default="financial_reports")

        WEB_SEARCH_PROVIDER: str = Field(default="duckduckgo")
        TAVILY_API_KEY: str = Field(default="")

        class Config:
            env_file = ".env"
            extra = "ignore"

    settings = Settings()

except ImportError:
    # Graceful fallback if pydantic_settings is not installed in the python environment
    class Settings:
        APP_NAME: str = os.getenv("APP_NAME", "FinRAG Analyst Agent")
        ENV: str = os.getenv("ENV", "development")
        DEBUG: bool = os.getenv("DEBUG", "True").lower() in ("true", "1")

        LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "openai")
        OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
        OPENAI_MODEL: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

        GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
        GROQ_MODEL: str = os.getenv("GROQ_MODEL", "llama-3.1-70b-versatile")

        OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", "qwen2.5:7b")

        EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

        QDRANT_LOCATION: str = os.getenv("QDRANT_LOCATION", ":memory:")
        QDRANT_COLLECTION_NAME: str = os.getenv("QDRANT_COLLECTION_NAME", "financial_reports")

        WEB_SEARCH_PROVIDER: str = os.getenv("WEB_SEARCH_PROVIDER", "duckduckgo")
        TAVILY_API_KEY: str = os.getenv("TAVILY_API_KEY", "")

    settings = Settings()
