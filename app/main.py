import os
import sys

# Ensure project root is in sys.path for absolute imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.routes import router
from app.config import settings

app = FastAPI(
    title=settings.APP_NAME,
    description="Production-grade Self-Corrective Financial Analyst RAG Agent powered by LangChain, LangGraph, FastAPI, & Qdrant.",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)

@app.on_event("startup")
async def startup_event():
    print("=" * 60)
    print(f"🚀 {settings.APP_NAME} SERVER STARTED SUCCESSFULLY")
    print(f"📌 LLM Provider: {settings.LLM_PROVIDER}")
    print(f"📌 Embedding Model: {settings.EMBEDDING_MODEL}")
    print(f"📌 Vector DB: Qdrant ({settings.QDRANT_LOCATION})")
    print("=" * 60)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
