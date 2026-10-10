from typing import Any

from app.config import settings
from app.core.errors import ModelConfigurationError, ProviderUnavailableError


def get_embeddings() -> Any:
    """Returns Embeddings model with graceful fallback."""
    try:
        if settings.OPENAI_API_KEY and not settings.OPENAI_API_KEY.startswith("mock"):
            from langchain_openai import OpenAIEmbeddings

            return OpenAIEmbeddings(
                model="text-embedding-3-small", openai_api_key=settings.OPENAI_API_KEY
            )
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
    """Initialize the configured real provider, or raise an explicit model error."""
    provider = settings.LLM_PROVIDER.lower()

    if provider == "openai":
        if not settings.OPENAI_API_KEY.strip() or settings.OPENAI_API_KEY.startswith("mock"):
            raise ModelConfigurationError(
                "OPENAI_API_KEY is required and must not be a mock key for the OpenAI provider."
            )
        model_name = settings.OPENAI_MODEL
        model_setting = "OPENAI_MODEL"
        package = "langchain-openai"
    elif provider == "groq":
        if not settings.GROQ_API_KEY.strip():
            raise ModelConfigurationError("GROQ_API_KEY is required for the Groq provider.")
        model_name = settings.GROQ_MODEL
        model_setting = "GROQ_MODEL"
        package = "langchain-groq"
    elif provider == "ollama":
        if not settings.OLLAMA_BASE_URL.strip():
            raise ModelConfigurationError("OLLAMA_BASE_URL is required for the Ollama provider.")
        model_name = settings.OLLAMA_MODEL
        model_setting = "OLLAMA_MODEL"
        package = "langchain-community"
    else:
        raise ModelConfigurationError("LLM_PROVIDER must be one of: openai, groq, ollama.")

    if not model_name.strip():
        raise ModelConfigurationError(f"{model_setting} is required for the configured provider.")

    model_class: Any
    model_options: dict[str, Any] = {"model": model_name, "temperature": temperature}
    try:
        if provider == "openai":
            from langchain_openai import ChatOpenAI

            model_class = ChatOpenAI
            model_options["api_key"] = settings.OPENAI_API_KEY
        elif provider == "groq":
            from langchain_groq import ChatGroq

            model_class = ChatGroq
            model_options["groq_api_key"] = settings.GROQ_API_KEY
        else:
            from langchain_community.chat_models import ChatOllama

            model_class = ChatOllama
            model_options["base_url"] = settings.OLLAMA_BASE_URL
    except ImportError as exc:
        raise ProviderUnavailableError(
            f"Package '{package}' and its dependencies are required for provider '{provider}'. "
            "Install it in the active environment (Groq: uv sync --extra groq)."
        ) from exc
    except Exception as exc:
        # Lazy SDK imports can fail while executing dependency initialization code.
        raise ProviderUnavailableError(
            f"Failed to load provider '{provider}'. Check the '{package}' installation."
        ) from exc

    try:
        return model_class(**model_options)
    except Exception as exc:
        # SDK constructors expose different exception types at this integration boundary.
        # Keep the cause for debugging, but never copy its potentially secret text.
        raise ProviderUnavailableError(
            f"Failed to initialize the configured {provider} model. "
            f"Check {model_setting} and the provider configuration."
        ) from exc
