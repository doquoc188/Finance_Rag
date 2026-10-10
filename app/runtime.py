"""Application-owned composition; graph nodes consume only resolved chat models."""

from dataclasses import dataclass

from app.core.vectorstore import HybridFinancialRetriever
from app.graph.workflow import FinragGraph, build_finrag_graph
from finrag.config import Settings
from finrag.models import ModelRoles, resolve_model_roles
from finrag.providers import ChatProviderRegistry
from finrag.providers.builtin import register_builtin_chat_providers
from finrag.retrieval import Reranker


@dataclass(frozen=True, slots=True)
class GraphRuntime:
    settings: Settings
    model_roles: ModelRoles
    graph: FinragGraph
    retriever: HybridFinancialRetriever
    reranker: Reranker | None


def _create_reranker(model_name: str) -> Reranker:
    from finrag.retrieval.bge import BgeCrossEncoderReranker

    return BgeCrossEncoderReranker(model_name)


def compose_runtime(
    settings: Settings | None = None,
    *,
    registry: ChatProviderRegistry | None = None,
    retriever: HybridFinancialRetriever | None = None,
    reranker: Reranker | None = None,
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
    if retriever is not None:
        if reranker is not None:
            raise ValueError("reranker cannot be supplied with a fully composed retriever")
        active_retriever = retriever
        active_reranker = retriever.reranker
    else:
        if reranker is not None and not configuration.retrieval.reranker_enabled:
            raise ValueError("reranker requires retrieval.reranker_enabled")
        active_reranker = reranker
        if configuration.retrieval.reranker_enabled and active_reranker is None:
            active_reranker = _create_reranker(configuration.retrieval.reranker_model)
        active_retriever = HybridFinancialRetriever(
            retrieval_settings=configuration.retrieval,
            reranker=active_reranker,
        )
    graph = build_finrag_graph(
        roles,
        max_rewrite_attempts=configuration.retrieval.max_rewrite_attempts,
        retriever=active_retriever,
        retrieval_limit=configuration.retrieval.final_top_k,
    )
    return GraphRuntime(configuration, roles, graph, active_retriever, active_reranker)
