from typing import TYPE_CHECKING, Literal, cast

from finrag.models import ModelRoles

if TYPE_CHECKING:
    from langgraph.graph.state import CompiledStateGraph

from app.graph.nodes import (
    GraphNode,
    create_generate_node,
    create_grade_documents_node,
    create_retrieve_node,
    create_transform_query_node,
    grade_generation_node,
)
from app.graph.state import FinGraphState, FinGraphUpdate
from finrag.retrieval import Retriever


class FallbackStateGraph:
    """Fallback Workflow Engine when langgraph package is loading or not yet installed."""

    def __init__(
        self,
        retrieve: GraphNode,
        grade: GraphNode,
        generate: GraphNode,
        rewrite: GraphNode,
        max_rewrite_attempts: int,
    ) -> None:
        self._retrieve = retrieve
        self._grade = grade
        self._generate = generate
        self._rewrite = rewrite
        self._max_rewrite_attempts = max_rewrite_attempts

    def invoke(self, initial_state: FinGraphUpdate) -> FinGraphState:
        # Both engines accept partial input and populate the existing state in order.
        state = cast(FinGraphState, dict(initial_state))
        rewrite_attempts = state.get("rewrite_attempts", state.get("retry_count", 0))
        state["rewrite_attempts"] = rewrite_attempts
        state["retry_count"] = rewrite_attempts

        while True:
            state.update(self._retrieve(state))
            state.update(self._grade(state))
            if state["is_relevant"] == "yes":
                break
            if state["rewrite_attempts"] >= self._max_rewrite_attempts:
                break
            state.update(self._rewrite(state))

        state.update(self._generate(state))
        state.update(grade_generation_node(state))

        return state


type FinragGraph = (
    CompiledStateGraph[FinGraphState, None, FinGraphUpdate, FinGraphState] | FallbackStateGraph
)


def build_finrag_graph(
    model_roles: ModelRoles,
    *,
    max_rewrite_attempts: int = 2,
    retriever: Retriever | None = None,
    retrieval_limit: int = 4,
) -> FinragGraph:
    """Builds and compiles the Self-Corrective Financial RAG StateGraph."""
    if (
        isinstance(max_rewrite_attempts, bool)
        or not isinstance(max_rewrite_attempts, int)
        or max_rewrite_attempts < 0
    ):
        raise ValueError("max_rewrite_attempts must be a non-negative integer")
    if (
        isinstance(retrieval_limit, bool)
        or not isinstance(retrieval_limit, int)
        or retrieval_limit < 1
    ):
        raise ValueError("retrieval_limit must be a positive integer")
    retrieve = create_retrieve_node(retriever, retrieval_limit)
    grade = create_grade_documents_node(model_roles.grading)
    generate = create_generate_node(model_roles.generation)
    rewrite = create_transform_query_node(model_roles.query_rewrite)
    try:
        from langgraph.graph import END, START, StateGraph

        workflow = StateGraph[FinGraphState, None, FinGraphUpdate, FinGraphState](FinGraphState)

        workflow.add_node("retrieve", retrieve)
        workflow.add_node("grade_documents", grade, input_schema=FinGraphState)
        workflow.add_node("generate", generate, input_schema=FinGraphState)
        workflow.add_node("transform_query", rewrite, input_schema=FinGraphState)
        workflow.add_node("fact_check", grade_generation_node)

        workflow.add_edge(START, "retrieve")
        workflow.add_edge("retrieve", "grade_documents")

        def decide_to_generate(state: FinGraphState) -> Literal["transform_query", "generate"]:
            is_rel = state.get("is_relevant", "yes")
            rewrite_attempts = state.get("rewrite_attempts", state.get("retry_count", 0))
            if is_rel == "no" and rewrite_attempts < max_rewrite_attempts:
                return "transform_query"
            return "generate"

        workflow.add_conditional_edges(
            "grade_documents",
            decide_to_generate,
            {"transform_query": "transform_query", "generate": "generate"},
        )

        workflow.add_edge("transform_query", "retrieve")
        workflow.add_edge("generate", "fact_check")
        workflow.add_edge("fact_check", END)

        # One failed attempt traverses rewrite -> retrieve -> grade. Raise LangGraph's
        # execution ceiling with the validated policy so large finite budgets remain finite.
        recursion_limit = max(25, 3 * max_rewrite_attempts + 10)
        app = workflow.compile().with_config(recursion_limit=recursion_limit)
        print("[LangGraph] Successfully compiled FinRAG StateGraph Workflow.")
        return app
    except Exception as e:
        print(f"[Notice] Using Fallback StateGraph Engine ({type(e).__name__}).")
        return FallbackStateGraph(retrieve, grade, generate, rewrite, max_rewrite_attempts)
