from __future__ import annotations

import os
import sys
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

if TYPE_CHECKING:
    from app.runtime import GraphRuntime


def create_app(runtime_factory: Callable[[], GraphRuntime] | None = None) -> FastAPI:
    """Own one composed graph per application lifespan; imports create no models."""
    # Retain direct-script support as well as the app.main:app entry point.
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    from app.api.routes import router
    from app.config import settings
    from app.runtime import compose_runtime

    factory = runtime_factory if runtime_factory is not None else compose_runtime

    @asynccontextmanager
    async def lifespan(application: FastAPI) -> AsyncIterator[None]:
        runtime = factory()
        application.state.runtime = runtime
        for role, configuration in (
            ("generation", runtime.settings.generation),
            ("grading", runtime.settings.grading),
            ("query_rewrite", runtime.settings.query_rewrite),
        ):
            print(f"[Model role] {role}: {configuration.provider} / {configuration.model}")
        try:
            yield
        finally:
            del application.state.runtime

    application = FastAPI(
        title=settings.APP_NAME,
        description="Production-grade Self-Corrective Financial Analyst RAG Agent powered by LangChain, LangGraph, FastAPI, & Qdrant.",
        version="1.0.0",
        lifespan=lifespan,
    )
    application.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    application.include_router(router)
    return application


app = create_app()

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
