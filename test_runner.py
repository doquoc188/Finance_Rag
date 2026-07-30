import sys
import os

# Fix Windows console encoding issue
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

print("=" * 60)
print("[TEST] TESTING FINRAG ENGINE COMPONENTS")
print("=" * 60)

try:
    # 1. Test Config
    from app.config import settings
    print(f"[OK] Config loaded: APP_NAME={settings.APP_NAME}")

    # 2. Test PDF Parser
    from app.core.pdf_parser import FinancialPDFParser
    parser = FinancialPDFParser()
    sample_content = """# Báo cáo FPT 2023
| Chỉ tiêu | 2023 |
| Doanh thu hợp nhất | 52.618 tỷ đồng |
| Lợi nhuận trước thuế | 9.203 tỷ đồng |
"""
    docs = parser.parse_text_or_markdown(sample_content, source_name="FPT_2023.md", company_name="FPT", year="2023")
    print(f"[OK] FinancialPDFParser created {len(docs)} document chunks with metadata: {docs[0].metadata}")

    # 3. Test Hybrid Retriever
    from app.core.vectorstore import HybridFinancialRetriever
    retriever = HybridFinancialRetriever(documents=docs)
    retrieved = retriever.get_relevant_documents("Doanh thu FPT năm 2023", top_k=1)
    print(f"[OK] HybridFinancialRetriever found {len(retrieved)} chunk: '{retrieved[0].page_content[:60]}...'")

    # 4. Test LangGraph Workflow
    from app.graph.workflow import finrag_app
    print("[OK] LangGraph FinRAG StateGraph loaded successfully!")

    initial_state = {
        "question": "Doanh thu hợp nhất năm 2023 của FPT là bao nhiêu?",
        "retry_count": 0,
        "web_search_needed": False
    }
    result = finrag_app.invoke(initial_state)
    print("[OK] LangGraph Execution Completed!")
    print(f"   -> Response snippet: {result.get('generation')[:120]}...")
    print(f"   -> Citations count: {len(result.get('citations', []))}")

    print("=" * 60)
    print("[SUCCESS] ALL FINRAG CORE TESTS PASSED SUCCESSFULLY!")
    print("=" * 60)

except Exception as e:
    print(f"[ERROR] TEST FAILED WITH ERROR: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
