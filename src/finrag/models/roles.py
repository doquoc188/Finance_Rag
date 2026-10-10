"""Explicit, typed chat-role composition outside domain and the legacy runtime."""

from dataclasses import dataclass

from langchain_core.language_models.chat_models import BaseChatModel

from finrag.config import Settings
from finrag.providers.registry import ChatProviderRegistry


@dataclass(frozen=True, slots=True)
class ModelRoles:
    """Immutable role bindings; the referenced models retain their own mutable state."""

    generation: BaseChatModel
    grading: BaseChatModel
    query_rewrite: BaseChatModel


def resolve_model_roles(settings: Settings, registry: ChatProviderRegistry) -> ModelRoles:
    """Resolve generation, grading, query_rewrite in order, returning only on success.

    Factories receive the original role configuration. Errors propagate unchanged;
    no partial result, environment lookup, fallback, caching, or cleanup is provided.
    """
    generation = registry.get(settings.generation.provider)(settings.generation)
    grading = registry.get(settings.grading.provider)(settings.grading)
    query_rewrite = registry.get(settings.query_rewrite.provider)(settings.query_rewrite)
    return ModelRoles(generation=generation, grading=grading, query_rewrite=query_rewrite)
