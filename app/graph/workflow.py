from typing import Dict, Any, Literal

from app.graph.state import FinGraphState
from app.graph.nodes import (
    retrieve_node,
    grade_documents_node,
    generate_node,
    transform_query_node,
    web_search_node,
    grade_generation_node
)

class FallbackStateGraph:
    """Fallback Workflow Engine when langgraph package is loading or not yet installed."""
    def invoke(self, state: FinGraphState) -> FinGraphState:
        # Step 1: Retrieve
        s1 = retrieve_node(state)
        state.update(s1)
        
        # Step 2: Grade Documents
        s2 = grade_documents_node(state)
        state.update(s2)
        
        # Condition Check
        if state.get("web_search_needed") or state.get("is_relevant") == "no":
            s4 = transform_query_node(state)
            state.update(s4)
            s5 = web_search_node(state)
            state.update(s5)
            
        # Step 3: Generate
        s3 = generate_node(state)
        state.update(s3)
        
        # Step 6: Fact Check
        s6 = grade_generation_node(state)
        state.update(s6)
        
        return state

def build_finrag_graph():
    """Builds and compiles the Self-Corrective Financial RAG StateGraph."""
    try:
        from langgraph.graph import StateGraph, END, START
        workflow = StateGraph(FinGraphState)

        workflow.add_node("retrieve", retrieve_node)
        workflow.add_node("grade_documents", grade_documents_node)
        workflow.add_node("generate", generate_node)
        workflow.add_node("transform_query", transform_query_node)
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
            {
                "transform_query": "transform_query",
                "generate": "generate"
            }
        )
        
        workflow.add_edge("transform_query", "web_search")
        workflow.add_edge("web_search", "generate")
        workflow.add_edge("generate", "fact_check")
        workflow.add_edge("fact_check", END)

        app = workflow.compile()
        print("[LangGraph] Successfully compiled FinRAG StateGraph Workflow.")
        return app
    except Exception as e:
        print(f"[Notice] Using Fallback StateGraph Engine ({e}).")
        return FallbackStateGraph()

# Singleton app instance
finrag_app = build_finrag_graph()
