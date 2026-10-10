from typing import TYPE_CHECKING, Literal, cast

from finrag.models import ModelRoles

if TYPE_CHECKING:
    from langgraph.graph.state import CompiledStateGraph

from app.graph.nodes import (
    GraphNode,
    create_generate_node,
    create_grade_documents_node,
    create_transform_query_node,
    grade_generation_node,
    retrieve_node,
    web_search_node,
)
from app.graph.state import FinGraphState, FinGraphUpdate


class FallbackStateGraph:
    """Fallback Workflow Engine when langgraph package is loading or not yet installed."""

    def __init__(self, grade: GraphNode, generate: GraphNode, rewrite: GraphNode) -> None:
        self._grade = grade
        self._generate = generate
        self._rewrite = rewrite

    def invoke(self, initial_state: FinGraphUpdate) -> FinGraphState:
        # Both engines accept partial input and populate the existing state in order.
        state = cast(FinGraphState, initial_state)
        # Step 1: Retrieve
        s1 = retrieve_node(state)
        state.update(s1)

        # Step 2: Grade Documents
        s2 = self._grade(state)
        state.update(s2)

        # Condition Check
        if state.get("web_search_needed") or state.get("is_relevant") == "no":
            s4 = self._rewrite(state)
            state.update(s4)
            s5 = web_search_node(state)
            state.update(s5)

        # Step 3: Generate
        s3 = self._generate(state)
        state.update(s3)

        # Step 6: Fact Check
        s6 = grade_generation_node(state)
        state.update(s6)

        return state


type FinragGraph = (
    CompiledStateGraph[FinGraphState, None, FinGraphUpdate, FinGraphState] | FallbackStateGraph
)


def build_finrag_graph(model_roles: ModelRoles) -> FinragGraph:
    """Builds and compiles the Self-Corrective Financial RAG StateGraph."""
    grade = create_grade_documents_node(model_roles.grading)
    generate = create_generate_node(model_roles.generation)
    rewrite = create_transform_query_node(model_roles.query_rewrite)
    try:
        from langgraph.graph import END, START, StateGraph

        workflow = StateGraph[FinGraphState, None, FinGraphUpdate, FinGraphState](FinGraphState)

        workflow.add_node("retrieve", retrieve_node)
        workflow.add_node("grade_documents", grade, input_schema=FinGraphState)
        workflow.add_node("generate", generate, input_schema=FinGraphState)
        workflow.add_node("transform_query", rewrite, input_schema=FinGraphState)
        workflow.add_node("web_search", web_search_node)
        workflow.add_node("fact_check", grade_generation_node)

        workflow.add_edge(START, "retrieve")
        workflow.add_edge("retrieve", "grade_documents")

        def decide_to_generate(state: FinGraphState) -> Literal["transform_query", "generate"]:
            web_search = state.get("web_search_needed", False)
            is_rel = state.get("is_relevant", "yes")
            retry_count = state.get("retry_count", 0)
            if (web_search or is_rel == "no") and retry_count < 2:
                return "transform_query"
            return "generate"

        workflow.add_conditional_edges(
            "grade_documents",
            decide_to_generate,
            {"transform_query": "transform_query", "generate": "generate"},
        )

        workflow.add_edge("transform_query", "web_search")
        workflow.add_edge("web_search", "generate")
        workflow.add_edge("generate", "fact_check")
        workflow.add_edge("fact_check", END)

        app = workflow.compile()
        print("[LangGraph] Successfully compiled FinRAG StateGraph Workflow.")
        return app
    except Exception as e:
        print(f"[Notice] Using Fallback StateGraph Engine ({type(e).__name__}).")
        return FallbackStateGraph(grade, generate, rewrite)
