"""FastAPI exception handlers for MolExp API.

This module registers exception handlers that convert MolExpError exceptions
to consistent JSON error responses.
"""

from __future__ import annotations

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from molexp.workflow.types import WorkflowError
from molexp.workspace.errors import (
    ExperimentExistsError as WorkspaceExperimentExistsError,
)
from molexp.workspace.errors import (
    ExperimentNotFoundError as WorkspaceExperimentNotFoundError,
)
from molexp.workspace.errors import (
    ProjectExistsError as WorkspaceProjectExistsError,
)
from molexp.workspace.errors import (
    ProjectNotFoundError as WorkspaceProjectNotFoundError,
)
from molexp.workspace.errors import (
    RunExistsError as WorkspaceRunExistsError,
)
from molexp.workspace.errors import (
    RunNotFoundError as WorkspaceRunNotFoundError,
)

from .exceptions import (
    DuplicateResourceError,
    ExperimentNotFoundError,
    MolExpError,
    ProjectNotFoundError,
    RunNotFoundError,
)


def register_exception_handlers(app: FastAPI) -> None:
    """Register exception handlers on the FastAPI app.

    Args:
        app: FastAPI application instance
    """

    @app.exception_handler(MolExpError)
    async def molexp_error_handler(request: Request, exc: MolExpError) -> JSONResponse:  # noqa: ARG001
        """Handle MolExpError exceptions with consistent JSON responses."""
        return JSONResponse(
            status_code=exc.status_code,
            content=exc.to_dict(),
        )

    @app.exception_handler(ValueError)
    async def value_error_handler(request: Request, exc: ValueError) -> JSONResponse:  # noqa: ARG001
        """Handle ValueError exceptions as validation errors."""
        return JSONResponse(
            status_code=400,
            content={
                "error": {
                    "code": "VALIDATION_ERROR",
                    "message": str(exc),
                }
            },
        )

    @app.exception_handler(WorkflowError)
    async def workflow_error_handler(request: Request, exc: WorkflowError) -> JSONResponse:  # noqa: ARG001
        """Map workflow-layer compile errors (cycles, unknown tasks, unreachable
        nodes, …) raised by ``ir_to_spec`` onto a structured 4xx — never a 500."""
        return JSONResponse(
            status_code=400,
            content={
                "error": {
                    "code": "WORKFLOW_INVALID",
                    "message": str(exc),
                }
            },
        )

    @app.exception_handler(FileNotFoundError)
    async def file_not_found_handler(request: Request, exc: FileNotFoundError) -> JSONResponse:  # noqa: ARG001
        """Handle Python FileNotFoundError."""
        return JSONResponse(
            status_code=404,
            content={
                "error": {
                    "code": "NOT_FOUND",
                    "message": str(exc),
                }
            },
        )

    # ── Workspace-layer error → HTTP 404 / 409 ────────────────────────────
    #
    # The workspace layer raises typed entity errors at the storage
    # boundary; the server layer maps them onto its existing
    # ``NotFoundError`` / ``DuplicateResourceError`` HTTP envelopes so
    # routes can simply re-raise without per-route try/except.

    @app.exception_handler(WorkspaceProjectNotFoundError)
    async def workspace_project_not_found_handler(
        request: Request,  # noqa: ARG001
        exc: WorkspaceProjectNotFoundError,
    ) -> JSONResponse:
        return JSONResponse(
            status_code=404,
            content=ProjectNotFoundError(exc.entity_id).to_dict(),
        )

    @app.exception_handler(WorkspaceExperimentNotFoundError)
    async def workspace_experiment_not_found_handler(
        request: Request,  # noqa: ARG001
        exc: WorkspaceExperimentNotFoundError,
    ) -> JSONResponse:
        return JSONResponse(
            status_code=404,
            content=ExperimentNotFoundError(exc.entity_id).to_dict(),
        )

    @app.exception_handler(WorkspaceRunNotFoundError)
    async def workspace_run_not_found_handler(
        request: Request,  # noqa: ARG001
        exc: WorkspaceRunNotFoundError,
    ) -> JSONResponse:
        return JSONResponse(
            status_code=404,
            content=RunNotFoundError(exc.entity_id).to_dict(),
        )

    @app.exception_handler(WorkspaceProjectExistsError)
    async def workspace_project_exists_handler(
        request: Request,  # noqa: ARG001
        exc: WorkspaceProjectExistsError,
    ) -> JSONResponse:
        return JSONResponse(
            status_code=409,
            content=DuplicateResourceError("Project", exc.entity_id).to_dict(),
        )

    @app.exception_handler(WorkspaceExperimentExistsError)
    async def workspace_experiment_exists_handler(
        request: Request,  # noqa: ARG001
        exc: WorkspaceExperimentExistsError,
    ) -> JSONResponse:
        return JSONResponse(
            status_code=409,
            content=DuplicateResourceError("Experiment", exc.entity_id).to_dict(),
        )

    @app.exception_handler(WorkspaceRunExistsError)
    async def workspace_run_exists_handler(
        request: Request,  # noqa: ARG001
        exc: WorkspaceRunExistsError,
    ) -> JSONResponse:
        return JSONResponse(
            status_code=409,
            content=DuplicateResourceError("Run", exc.entity_id).to_dict(),
        )
