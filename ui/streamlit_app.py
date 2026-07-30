import os
import time
import requests
import streamlit as st
from langchain_core.documents import Document

# Streamlit Page Config
st.set_page_config(
    page_title="FinRAG Analyst - Trợ Lý Báo Cáo Tài Chính",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS styling for premium look
st.markdown("""
<style>
    .stApp {
        background-color: #0e1117;
        color: #e0e0e0;
    }
    .main-header {
        font-size: 2.3rem;
        font-weight: 700;
        background: linear-gradient(90deg, #1f77b4, #00d2ff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #9aa0a6;
        margin-bottom: 1.5rem;
    }
    .citation-card {
        background-color: #1a1f2c;
        border-left: 4px solid #00d2ff;
        padding: 12px;
        border-radius: 6px;
        margin-top: 8px;
        font-size: 0.9rem;
    }
    .metric-box {
        background-color: #161b22;
        border: 1px solid #30363d;
        border-radius: 8px;
        padding: 12px;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# Application Header
st.markdown('<div class="main-header">📈 FinRAG Analyst - Trợ Lý Báo Cáo Tài Chính</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Hệ thống Self-Corrective RAG Agent phân tích Báo cáo tài chính & Số liệu doanh nghiệp chuẩn Production</div>', unsafe_allow_html=True)

# API Endpoint Config
API_URL = os.getenv("API_URL", "http://localhost:8000/api/v1")

# Sidebar
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/financial-analytics.png", width=70)
    st.title(" Cấu Hình System")
    
    st.markdown("---")
    st.subheader("📄 Nạp Báo Cáo Tài Chính (Ingestion)")
    uploaded_file = st.file_uploader("Upload PDF Báo cáo tài chính", type=["pdf", "md", "txt"])
    company_name = st.text_input("Tên Doanh nghiệp", value="FPT Corporation")
    report_year = st.text_input("Năm Báo cáo", value="2023")
    
    if st.button("📥 Nạp Dữ Liệu Vào System", use_container_width=True):
        if uploaded_file:
            with st.spinner("Đang phân tích Bảng biểu & Chunking PDF..."):
                try:
                    files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "text/markdown" if uploaded_file.name.endswith(".md") else "application/pdf")}
                    data = {"company_name": company_name, "year": report_year}
                    res = requests.post(f"{API_URL}/ingest", files=files, data=data)
                    if res.status_code == 200:
                        st.success(f"✅ {res.json().get('message')}")
                    else:
                        st.error(f"Lỗi: {res.text}")
                except Exception as e:
                    st.error(f"Không thể kết nối API Server: {e}")
        else:
            st.warning("Vui lòng chọn file PDF hoặc dùng dữ liệu mẫu!")

    st.markdown("---")
    st.subheader("💡 Dữ Liệu Mẫu Có Sẵn")
    if st.button("⚡ Load Báo Cáo FPT 2023 Mẫu", use_container_width=True):
        sample_path = os.path.join(os.getcwd(), "data", "sample_reports", "FPT_BaoCaoThuongNien_2023.md")
        if os.path.exists(sample_path):
            with open(sample_path, "r", encoding="utf-8") as f:
                content = f.read()
            with st.spinner("Đang nạp Báo cáo FPT 2023 vào Vector DB..."):
                try:
                    files = {"file": ("FPT_BaoCaoThuongNien_2023.md", content.encode("utf-8"), "text/markdown")}
                    res = requests.post(f"{API_URL}/ingest", files=files, data={"company_name": "FPT Corporation", "year": "2023"})
                    if res.status_code == 200:
                        st.success("✅ Đã nạp thành công Báo cáo FPT 2023!")
                except Exception as e:
                    st.error(f"Lỗi: {e}")
        else:
            st.error("Không tìm thấy file mẫu.")

    st.markdown("---")
    st.markdown("**Core Tech Stack:**")
    st.caption("• LangChain & LangGraph StateGraph\n• Qdrant Vector DB & BM25 Hybrid\n• FastRank Reranker\n• FastAPI & Streamlit")

# Session Chat State
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Xin chào! Tôi là FinRAG Analyst. Bạn có thể hỏi tôi về số liệu doanh thu, lợi nhuận, bảng cân đối kế toán hoặc so sánh kết quả kinh doanh trong báo cáo tài chính."}
    ]

# Render Chat History
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "citations" in message and message["citations"]:
            with st.expander("📚 Trích dẫn nguồn (Citations & Source Chunks)"):
                for cit in message["citations"]:
                    st.markdown(f"""
                    <div class="citation-card">
                        <b>📄 Nguồn:</b> {cit.get('source')} | <b>Trang:</b> {cit.get('page')} | <b>Công ty:</b> {cit.get('company')} ({cit.get('year')})<br/>
                        <i>"{cit.get('snippet')}"</i>
                    </div>
                    """, unsafe_allow_html=True)

# User Query Input
if prompt := st.chat_input("Nhập câu hỏi tài chính (ví dụ: Doanh thu và Lợi nhuận trước thuế năm 2023 của FPT là bao nhiêu?)..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("FinRAG Agent đang thực thi LangGraph StateGraph..."):
            try:
                response = requests.post(
                    f"{API_URL}/chat",
                    json={"question": prompt, "company_name": "FPT Corporation", "year": "2023"},
                    timeout=60
                )
                if response.status_code == 200:
                    data = response.json()
                    answer = data.get("answer", "")
                    citations = data.get("citations", [])
                    exec_info = data.get("execution_info", {})

                    st.markdown(answer)

                    # Show execution flow info
                    if exec_info.get("web_search_needed"):
                        st.info("ℹ️ Hệ thống đã tự động kích hoạt **Query Rewriter & Web Search Fallback** để tìm thêm thông tin bổ sung!")

                    if citations:
                        with st.expander("📚 Trích dẫn nguồn (Citations & Source Chunks)"):
                            for cit in citations:
                                st.markdown(f"""
                                <div class="citation-card">
                                    <b>📄 Nguồn:</b> {cit.get('source')} | <b>Trang:</b> {cit.get('page')} | <b>Công ty:</b> {cit.get('company')} ({cit.get('year')})<br/>
                                    <i>"{cit.get('snippet')}"</i>
                                </div>
                                """, unsafe_allow_html=True)

                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "citations": citations
                    })
                else:
                    err_msg = f"❌ Lỗi từ API Server ({response.status_code}): {response.text}"
                    st.error(err_msg)
            except Exception as e:
                st.error(f"❌ Không thể kết nối tới Backend API tại `{API_URL}`: {e}")
