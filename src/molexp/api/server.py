"""MolExp API Server - Refactored with modular routes.

This module provides the FastAPI application for the MolExp API.
All endpoints are organized into modular route files under the routes/ directory.
"""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from molexp.workflow.plugin import get_node_registry, load_plugins

from .handlers import register_exception_handlers
from .routes import create_api_router
from .schemas import HealthResponse

# ============================================================================
# Feature Flags
# ============================================================================

# Try to import workspace module
try:
    from molexp.workspace import Workspace

    WORKSPACE_AVAILABLE = True
except ImportError:
    WORKSPACE_AVAILABLE = False

# Try to import IR module
try:
    from molexp.ir.loader import load_workflow_from_json

    IR_AVAILABLE = True
except ImportError:
    IR_AVAILABLE = False


# ============================================================================
# Application Setup
# ============================================================================


def create_app() -> FastAPI:
    """Create and configure the FastAPI application.

    Returns:
        Configured FastAPI application
    """
    app = FastAPI(
        title="MolExp API",
        version="0.3.0",
        description="Research workflow management API with modular architecture",
    )

    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:3000", "http://localhost:5173"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Register exception handlers
    register_exception_handlers(app)

    # Include API routes
    api_router = create_api_router()
    app.include_router(api_router)

    # Register startup event
    @app.on_event("startup")
    async def startup_event():
        """Load node plugins at startup."""
        load_plugins()
        registry = get_node_registry()
        node_count = len(registry.list_all())
        print(f"✓ Loaded {node_count} node types from plugins")

    # Health check endpoint
    @app.get("/health", response_model=HealthResponse, tags=["system"])
    def health_check() -> HealthResponse:
        """Health check endpoint."""
        return HealthResponse(
            status="healthy",
            workspace_available=WORKSPACE_AVAILABLE,
            ir_available=IR_AVAILABLE,
        )

    return app


# Create the app instance
app = create_app()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("molexp.api.server:app", host="0.0.0.0", port=8000, reload=True)
