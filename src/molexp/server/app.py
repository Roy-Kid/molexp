"""MolExp API Server.

This module provides the FastAPI application factory.
It strictly adheres to the standard pattern:
- /api/* -> Backend API
- /* -> Static Files (Production only, SPA support)
"""

from __future__ import annotations

import logging
import os
from typing import Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from molexp.workflow.plugin import get_task_registry, load_plugins

from .handlers import register_exception_handlers
from .routes import create_api_router
from .schemas import HealthResponse



def create_app(static_dir: Optional[str] = None) -> FastAPI:
    """Create and configure the FastAPI application.

    Args:
        static_dir: Directory containing built UI files (Production only).
                    If provided, the app will serve these files at the root.
                    If None, the app runs in API-only mode (Development).
    """
    app = FastAPI(
        title="MolExp API",
        version="0.3.0",
        description="Research workflow management API",
        docs_url="/api/docs",  # Move docs to distinct path
        openapi_url="/api/openapi.json",
    )

    # 1. CORS Configuration (Dev Mode Support)
    # Allows requests from standard frontend dev server ports.
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

    # 3. API Routes
    # STRICT REQUIREMENT: All API routes must be under /api prefix
    api_router = create_api_router()
    app.include_router(api_router, prefix="/api")

    # 4. Startup Events
    @app.on_event("startup")
    async def startup_event():
        """Load node plugins at startup."""
        try:
            load_plugins()
            registry = get_task_registry()
            node_count = len(registry.list_all())
            print(f"✓ Loaded {node_count} node types from plugins")
        except Exception as e:
            logging.warning(f"Failed to load plugins: {e}")

    # 5. System Routes (Health Check)
    @app.get("/api/health", response_model=HealthResponse, tags=["system"])
    def health_check() -> HealthResponse:
        return HealthResponse(
            status="healthy",
            workspace_available=True,
            ir_available=True,
        )
    return app


# Default app instance for Development Mode (uvicorn reload)
# This instance runs in API-only mode (static_dir=None)
app = create_app()
