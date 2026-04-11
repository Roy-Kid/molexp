"""MolExp API Server.

This module provides the FastAPI application factory.
It strictly adheres to the standard pattern:
- /api/* -> Backend API
- /* -> Static Files (Production only, SPA support)
"""

from __future__ import annotations

import logging
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .handlers import register_exception_handlers
from .routes import create_api_router
from .schemas import HealthResponse


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None]:
    """Application lifespan: startup and shutdown events."""
    logging.info("MolExp server starting up")
    yield
    logging.info("MolExp server shutting down")


def create_app(static_dir: str | None = None) -> FastAPI:
    """Create and configure the FastAPI application.

    Args:
        static_dir: Directory containing built UI files (Production only).
                    If provided, the app will serve these files at the root.
                    If None, the app runs in API-only mode (Development).
    """
    app = FastAPI(
        title="MolExp API",
        version="0.1.0",
        description="Research workflow management API",
        docs_url="/api/docs",
        openapi_url="/api/openapi.json",
        lifespan=lifespan,
    )

    # 1. CORS Configuration (Dev Mode Support)
    origins = [
        "http://localhost:3000",
        "http://localhost:5173",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:5173",
    ]

    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # 2. Exception Handlers
    register_exception_handlers(app)

    # 3. API Routes (all under /api prefix)
    api_router = create_api_router()
    app.include_router(api_router, prefix="/api")

    # 4. System Routes (Health Check)
    @app.get("/api/health", response_model=HealthResponse, tags=["system"])
    def health_check() -> HealthResponse:
        from molexp.plugins import Capability, registry

        return HealthResponse(
            status="healthy",
            workspace_available=True,
            capabilities={
                cap.value: registry.is_available(cap) for cap in Capability
            },
        )

    # 5. Root fallback (when no static UI is served)
    if static_dir is None:

        @app.get("/", tags=["system"])
        def root():
            return {
                "service": "molexp",
                "docs": "/api/docs",
                "health": "/api/health",
            }

    return app


# Default app instance for Development Mode (uvicorn reload)
app = create_app()
