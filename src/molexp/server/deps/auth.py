"""Auth DI for the FastAPI surface.

Cookie name and public-path policy live here so routes and the gate share
one spelling. The service itself is :mod:`molexp.services.auth`.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import Cookie, Depends, HTTPException, Request, status

from molexp.services.auth import (
    AuthError,
    AuthUser,
    get_auth_service,
    is_auth_enabled,
)

#: HttpOnly session cookie (browser + EventSource).
SESSION_COOKIE = "molexp_session"


def get_session_id(
    request: Request,
    molexp_session: Annotated[str | None, Cookie(alias=SESSION_COOKIE)] = None,
) -> str | None:
    """Resolve the opaque session id from cookie or ``Authorization: Bearer``."""
    if molexp_session:
        return molexp_session
    auth_header = request.headers.get("authorization") or request.headers.get("Authorization")
    if auth_header and auth_header.lower().startswith("bearer "):
        token = auth_header[7:].strip()
        return token or None
    return None


def get_optional_user(
    session_id: Annotated[str | None, Depends(get_session_id)] = None,
) -> AuthUser | None:
    return get_auth_service().resolve_session(session_id)


def require_user(
    user: Annotated[AuthUser | None, Depends(get_optional_user)] = None,
) -> AuthUser:
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required",
        )
    return user


def require_user_if_enabled(
    request: Request,
    user: Annotated[AuthUser | None, Depends(get_optional_user)] = None,
) -> AuthUser | None:
    """Gate for protected routers: when auth is off, pass; when on, require user."""
    if not is_auth_enabled():
        return None
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required",
        )
    try:
        get_auth_service().assert_method_allowed(user, request.method)
    except AuthError as exc:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=exc.message) from exc
    return user


def require_admin(
    user: Annotated[AuthUser, Depends(require_user)],
) -> AuthUser:
    try:
        get_auth_service().assert_can_manage_users(user)
    except AuthError as exc:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=exc.message) from exc
    return user


def auth_error_http(exc: AuthError) -> HTTPException:
    if exc.code in {"invalid_credentials"}:
        return HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail=exc.message)
    if exc.code in {"rate_limited"}:
        return HTTPException(status_code=status.HTTP_429_TOO_MANY_REQUESTS, detail=exc.message)
    if exc.code in {"forbidden"}:
        return HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=exc.message)
    if exc.code in {"not_empty", "user_error"}:
        return HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=exc.message)
    return HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=exc.message)
