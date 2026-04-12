"""MolExp API Server.

This module provides the FastAPI application factory.
It strictly adheres to the standard pattern:
- /api/* -> Backend API
- /* -> Static Files (Production only, SPA support)
"""

from __future__ import annotations

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from mollog import get_logger

from .handlers import register_exception_handlers
from .routes import create_api_router
from .schemas import HealthResponse

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None]:
    """Application lifespan: startup and shutdown events."""
    logger.info("MolExp server starting up")
    yield
    logger.info("MolExp server shutting down")


# ---------------------------------------------------------------------------
# Bundled frontend discovery
# ---------------------------------------------------------------------------


def _find_bundled_webapp() -> Path | None:
    """Locate the ``_webapp`` directory shipped inside the installed package.

    Uses ``importlib.resources`` so this works regardless of whether the
    package was installed from a wheel, an editable install, or a checkout.
    Returns *None* when the frontend has not been compiled (normal in dev).
    """
    from importlib import resources

    try:
        pkg_path = Path(str(resources.files("molexp")))
        webapp = pkg_path / "_webapp"
        if webapp.is_dir() and (webapp / "index.html").exists():
            return webapp
    except Exception:  # noqa: BLE001
        pass
    return None


# ---------------------------------------------------------------------------
# SPA mount helper
# ---------------------------------------------------------------------------


def _mount_webapp(app: FastAPI, webapp_dir: Path) -> None:
    """Mount the React SPA and its static assets onto *app*.

    Must be called **after** all ``/api`` routes have been registered so that
    API routes take priority over the catch-all SPA fallback.
    """
    from fastapi.responses import FileResponse
    from starlette.staticfiles import StaticFiles

    index_html = str(webapp_dir / "index.html")

    # Serve hashed JS / CSS / images produced by rsbuild
    static_subdir = webapp_dir / "static"
    if static_subdir.is_dir():
        app.mount(
            "/static",
            StaticFiles(directory=str(static_subdir)),
            name="webapp_static",
        )

    # SPA fallback — serves index.html for every non-API, non-static path
    @app.get("/{full_path:path}", include_in_schema=False)
    async def _spa_fallback(full_path: str) -> FileResponse:
        # If an actual file exists at the root (favicon.ico, robots.txt, …)
        candidate = webapp_dir / full_path
        if full_path and candidate.is_file():
            return FileResponse(str(candidate))
        return FileResponse(index_html)


# ---------------------------------------------------------------------------
# Application factory
# ---------------------------------------------------------------------------


def create_app(
    static_dir: str | Path | None = None,
    *,
    serve_static: bool = True,
) -> FastAPI:
    """Create and configure the FastAPI application.

    Args:
        static_dir: Explicit path to built UI files.  When *None* and
            *serve_static* is ``True`` the bundled ``_webapp`` directory
            is auto-detected via ``importlib.resources``.
        serve_static: Set to ``False`` to run in API-only mode.
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

    # 5. Static file serving (production) or root fallback (dev / no build)
    webapp_path: Path | None = None
    if serve_static:
        if static_dir is not None:
            webapp_path = Path(static_dir)
        else:
            webapp_path = _find_bundled_webapp()

    if (
        webapp_path is not None
        and webapp_path.is_dir()
        and (webapp_path / "index.html").exists()
    ):
        _mount_webapp(app, webapp_path)
    else:
        @app.get("/", tags=["system"])
        def root():
            return {
                "service": "molexp",
                "docs": "/api/docs",
                "health": "/api/health",
            }

    return app


# Default app instance for Development Mode (uvicorn --reload).
# API-only — the frontend dev server (localhost:5173) runs separately.
app = create_app(serve_static=False)
