"""Application-owned composition; graph nodes consume only resolved chat models."""

from dataclasses import dataclass

from app.core.vectorstore import HybridFinancialRetriever
from app.graph.workflow import FinragGraph, build_finrag_graph
from finrag.config import Settings
from finrag.models import ModelRoles, resolve_model_roles
from finrag.providers import ChatProviderRegistry
from finrag.providers.builtin import register_builtin_chat_providers


@dataclass(frozen=True, slots=True)
class GraphRuntime:
    settings: Settings
    model_roles: ModelRoles
    graph: FinragGraph
    retriever: HybridFinancialRetriever


def compose_runtime(
    settings: Settings | None = None,
    *,
    registry: ChatProviderRegistry | None = None,
    retriever: HybridFinancialRetriever | None = None,
) -> GraphRuntime:
    """Resolve once; explicit caller registries own all their registrations.

    The production default uses a new empty registry, so sequential built-in
    registration cannot conflict. Provider errors propagate before graph building.
    """
    configuration = settings if settings is not None else Settings()
    if registry is None:
        registry = ChatProviderRegistry()
        register_builtin_chat_providers(registry)
    roles = resolve_model_roles(configuration, registry)
    active_retriever = (
        retriever
        if retriever is not None
        else HybridFinancialRetriever(retrieval_settings=configuration.retrieval)
    )
    graph = build_finrag_graph(
        roles,
        max_rewrite_attempts=configuration.retrieval.max_rewrite_attempts,
        retriever=active_retriever,
        retrieval_limit=configuration.retrieval.final_top_k,
    )
    return GraphRuntime(configuration, roles, graph, active_retriever)
