"""FastAPI exception handlers for MolExp API.

This module registers exception handlers that convert MolExpError exceptions
to consistent JSON error responses.
"""

from __future__ import annotations

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from .exceptions import MolExpError


def register_exception_handlers(app: FastAPI) -> None:
    """Register exception handlers on the FastAPI app.

    Args:
        app: FastAPI application instance
    """

    @app.exception_handler(MolExpError)
    async def molexp_error_handler(request: Request, exc: MolExpError) -> JSONResponse:
        """Handle MolExpError exceptions with consistent JSON responses."""
        return JSONResponse(
            status_code=exc.status_code,
            content=exc.to_dict(),
        )

    @app.exception_handler(ValueError)
    async def value_error_handler(request: Request, exc: ValueError) -> JSONResponse:
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

    @app.exception_handler(FileNotFoundError)
    async def file_not_found_handler(
        request: Request, exc: FileNotFoundError
    ) -> JSONResponse:
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
