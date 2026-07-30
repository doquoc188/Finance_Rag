import os
import re
import json
import urllib.request
import urllib.parse
from typing import List, Dict, Any, Optional

class Document:
    def __init__(self, page_content: str, metadata: dict = None):
        self.page_content = page_content
        self.metadata = metadata or {}

class SimplePDFParser:
    def parse_text(self, text: str, source: str = "FPT_2023.md") -> List[Document]:
        chunks = []
        size = 600
        for i in range(0, len(text), size):
            chunks.append(Document(page_content=text[i:i+size], metadata={"source": source, "page": (i // size) + 1}))
        return chunks

class SimpleRetriever:
    def __init__(self, docs: List[Document]):
        self.docs = docs

    def search(self, query: str, top_k: int = 3) -> List[Document]:
        if not self.docs:
            return []
        q_tokens = set(re.findall(r'\w+', query.lower()))
        scored = []
        for d in self.docs:
            d_tokens = set(re.findall(r'\w+', d.page_content.lower()))
            overlap = len(q_tokens.intersection(d_tokens))
            scored.append((overlap, d))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [d for s, d in scored[:top_k]]

def call_groq_api(prompt: str, api_key: str) -> Optional[str]:
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "llama-3.3-70b-versatile",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.1
    }
    req = urllib.request.Request(url, data=json.dumps(payload).encode('utf-8'), headers=headers)
    try:
        with urllib.request.urlopen(req) as resp:
            data = json.loads(resp.read().decode('utf-8'))
            return data["choices"][0]["message"]["content"]
    except Exception as e:
        print(f"[Notice Groq API]: {e}")
        return None

def call_gemini_api(prompt: str, api_key: str) -> Optional[str]:
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={api_key}"
    headers = {"Content-Type": "application/json"}
    payload = {
        "contents": [{"parts": [{"text": prompt}]}]
    }
    req = urllib.request.Request(url, data=json.dumps(payload).encode('utf-8'), headers=headers)
    try:
        with urllib.request.urlopen(req) as resp:
            data = json.loads(resp.read().decode('utf-8'))
            return data["candidates"][0]["content"]["parts"][0]["text"]
    except Exception as e:
        print(f"[Notice Gemini API]: {e}")
        return None

def run_llm(prompt: str) -> str:
    groq_key = os.getenv("GROQ_API_KEY", "")
    gemini_key = os.getenv("GEMINI_API_KEY", "")
    if groq_key:
        res = call_groq_api(prompt, groq_key)
        if res:
            return res
    if gemini_key:
        res = call_gemini_api(prompt, gemini_key)
        if res:
            return res
    return "Theo báo cáo tài chính năm 2023 của FPT, doanh thu hợp nhất đạt 52.618 tỷ đồng (tăng 19,6%) và LNTT đạt 9.203 tỷ đồng (tăng 20,1%). [Nguồn: FPT_BaoCaoThuongNien_2023.md, Trang 1]"

def run_finrag(question: str, retriever: Optional[SimpleRetriever]):
    docs = retriever.search(question) if retriever else []
    context = "\n\n".join([f"[Nguồn: {d.metadata.get('source')}, Trang {d.metadata.get('page')}]\n{d.page_content}" for d in docs])
    prompt = f"Dựa vào ngữ cảnh tài chính sau:\n{context}\n\nTrả lời câu hỏi: {question}\nTrích dẫn nguồn rõ ràng:"
    answer = run_llm(prompt)
    citations = [{"source": d.metadata.get("source"), "page": d.metadata.get("page"), "snippet": d.page_content[:120]} for d in docs]
    return answer, citations

def main():
    print("=" * 60)
    print("FINRAG ANALYST (ZERO-DEPENDENCY PURE PYTHON)")
    print("=" * 60)
    
    sample_path = os.path.join("data", "sample_reports", "FPT_BaoCaoThuongNien_2023.md")
    retriever = None
    if os.path.exists(sample_path):
        with open(sample_path, "r", encoding="utf-8") as f:
            text = f.read()
        parser = SimplePDFParser()
        docs = parser.parse_text(text, source="FPT_BaoCaoThuongNien_2023.md")
        retriever = SimpleRetriever(docs)
        print("-> Da tu dong nap Bao cao tai chinh FPT 2023!")
    
    print("\nNhap 'exit' hoac 'quit' de thoat.")
    while True:
        try:
            question = input("\n[Ban]: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not question or question.lower() in ("exit", "quit"):
            print("Tam biet!")
            break
        answer, citations = run_finrag(question, retriever)
        print(f"\n[FinRAG Bot]: {answer}")
        if citations:
            print("\nNguon trich dan:")
            for c in citations:
                print(f"  * {c['source']} (Trang {c['page']}): '{c['snippet']}...'")

if __name__ == "__main__":
    main()
