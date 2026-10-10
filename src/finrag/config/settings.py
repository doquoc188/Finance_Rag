"""New configuration contracts; the legacy app does not consume them yet."""

from typing import Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from finrag.domain.retrieval import RetrievalStrategy


class _ConfigModel(BaseModel):
    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        str_strip_whitespace=True,
        validate_default=True,
        hide_input_in_errors=True,
    )


class ChatModelSettings(_ConfigModel):
    """Role-local model selection; provider capabilities are not checked here."""

    provider: str = Field(default="openai", min_length=1)
    model: str = Field(default="gpt-4o-mini", min_length=1)
    temperature: float = Field(default=0.0, ge=0, allow_inf_nan=False)


class EmbeddingSettings(_ConfigModel):
    provider: str = Field(default="huggingface", min_length=1)
    model: str = Field(default="sentence-transformers/all-MiniLM-L6-v2", min_length=1)


class RetrievalSettings(_ConfigModel):
    strategy: RetrievalStrategy = RetrievalStrategy.DENSE
    dense_top_k: int = Field(default=20, gt=0)
    sparse_top_k: int = Field(default=20, gt=0)
    fused_top_k: int = Field(default=20, gt=0)
    final_top_k: int = Field(default=4, gt=0)

    @field_validator("dense_top_k", "sparse_top_k", "fused_top_k", "final_top_k", mode="before")
    @classmethod
    def reject_boolean_counts(cls, value: object) -> object:
        if isinstance(value, bool):
            raise ValueError("Retrieval counts must be positive integers, not booleans")
        return value

    @model_validator(mode="after")
    def validate_result_count(self) -> Self:
        if self.final_top_k > self.fused_top_k:
            raise ValueError("final_top_k must not exceed fused_top_k")
        return self


class GraphRAGSettings(_ConfigModel):
    """Opt-in graph index identity only; no graph adapter is initialized."""

    enabled: bool = False
    index_name: str | None = Field(default=None, min_length=1)

    @model_validator(mode="after")
    def validate_enabled_index(self) -> Self:
        if self.enabled and self.index_name is None:
            raise ValueError("GraphRAG index_name is required when enabled")
        return self


class Settings(BaseSettings):
    """Load FINRAG_<FIELD>__<NESTED_FIELD> variables without clients or credentials."""

    model_config = SettingsConfigDict(
        env_prefix="FINRAG_",
        env_nested_delimiter="__",
        env_file=None,
        extra="forbid",
        frozen=True,
        validate_default=True,
        hide_input_in_errors=True,
    )

    generation: ChatModelSettings = Field(default_factory=ChatModelSettings)
    grading: ChatModelSettings = Field(default_factory=ChatModelSettings)
    query_rewrite: ChatModelSettings = Field(default_factory=ChatModelSettings)
    embedding: EmbeddingSettings = Field(default_factory=EmbeddingSettings)
    retrieval: RetrievalSettings = Field(default_factory=RetrievalSettings)
    graphrag: GraphRAGSettings = Field(default_factory=GraphRAGSettings)

    @model_validator(mode="after")
    def validate_graph_selection(self) -> Self:
        if self.retrieval.strategy is RetrievalStrategy.GRAPH and not self.graphrag.enabled:
            raise ValueError("Graph retrieval requires graphrag.enabled")
        return self
