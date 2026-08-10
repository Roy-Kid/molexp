"""HTTP auth surface — verbs aligned with ``gh auth`` / ``molexp auth``.

Public (no session when auth on): ``status``, ``login``.
Authenticated: ``logout``, ``me``, ``token``, ``refresh``, ``switch``.
Admin: ``/users`` CRUD.
"""

from __future__ import annotations

from typing import Annotated, Literal

from fastapi import APIRouter, Depends, Request, Response
from pydantic import BaseModel, Field

from molexp.server.deps.auth import (
    SESSION_COOKIE,
    auth_error_http,
    get_session_id,
    require_admin,
    require_user,
)
from molexp.services.auth import (
    AuthError,
    AuthUser,
    AuthUserPublic,
    get_auth_service,
    is_auth_enabled,
)
from molexp.services.auth.models import AuthRole

router = APIRouter(prefix="/auth", tags=["auth"])


class LoginRequest(BaseModel):
    username: str = Field(..., min_length=1)
    password: str = Field(..., min_length=1)


class SwitchRequest(BaseModel):
    username: str = Field(..., min_length=1)
    password: str = Field(..., min_length=1)


class CreateUserRequest(BaseModel):
    username: str = Field(..., min_length=1)
    password: str = Field(..., min_length=1)
    role: AuthRole = "operator"
    workspaces: list[str] = Field(default_factory=lambda: ["*"])


class PatchUserRequest(BaseModel):
    role: AuthRole | None = None
    workspaces: list[str] | None = None
    disabled: bool | None = None


class PasswordRequest(BaseModel):
    password: str = Field(..., min_length=1)


class AuthStatusResponse(BaseModel):
    enabled: bool
    authenticated: bool
    user: AuthUserPublic | None = None


class AuthTokenResponse(BaseModel):
    token: str
    token_type: Literal["bearer"] = "bearer"


class AuthUserListResponse(BaseModel):
    users: list[AuthUserPublic]


def _set_session_cookie(response: Response, session_id: str) -> None:
    response.set_cookie(
        key=SESSION_COOKIE,
        value=session_id,
        httponly=True,
        samesite="lax",
        path="/",
        secure=False,
        max_age=60 * 60 * 24 * 7,
    )


def _clear_session_cookie(response: Response) -> None:
    response.delete_cookie(key=SESSION_COOKIE, path="/")


@router.get("/status", response_model=AuthStatusResponse)
def auth_status(
    session_id: Annotated[str | None, Depends(get_session_id)] = None,
) -> AuthStatusResponse:
    service = get_auth_service()
    state = service.status(session_id, enabled=is_auth_enabled())
    return AuthStatusResponse(
        enabled=state.enabled,
        authenticated=state.authenticated,
        user=state.user,
    )


@router.post("/login", response_model=AuthUserPublic)
def auth_login(body: LoginRequest, response: Response, request: Request) -> AuthUserPublic:
    service = get_auth_service()
    client = request.client.host if request.client else None
    try:
        user, session = service.login(body.username, body.password, client_key=client)
    except AuthError as exc:
        raise auth_error_http(exc) from exc
    _set_session_cookie(response, session.session_id)
    return user.public()


@router.post("/logout", status_code=204)
def auth_logout(
    response: Response,
    session_id: Annotated[str | None, Depends(get_session_id)] = None,
) -> None:
    get_auth_service().logout(session_id)
    _clear_session_cookie(response)


@router.get("/me", response_model=AuthUserPublic)
def auth_me(user: Annotated[AuthUser, Depends(require_user)]) -> AuthUserPublic:
    return user.public()


@router.get("/token", response_model=AuthTokenResponse)
def auth_token(
    user: Annotated[AuthUser, Depends(require_user)],
    session_id: Annotated[str | None, Depends(get_session_id)] = None,
) -> AuthTokenResponse:
    del user  # auth gate only
    token = get_auth_service().token_for(session_id)
    if not token:
        raise auth_error_http(AuthError("invalid_credentials", "No active session"))
    return AuthTokenResponse(token=token)


@router.post("/refresh", response_model=AuthUserPublic)
def auth_refresh(
    response: Response,
    user: Annotated[AuthUser, Depends(require_user)],
    session_id: Annotated[str | None, Depends(get_session_id)] = None,
) -> AuthUserPublic:
    record = get_auth_service().refresh(session_id)
    if record is None:
        raise auth_error_http(AuthError("invalid_credentials", "No active session"))
    _set_session_cookie(response, record.session_id)
    return user.public()


@router.post("/switch", response_model=AuthUserPublic)
def auth_switch(
    body: SwitchRequest,
    response: Response,
    session_id: Annotated[str | None, Depends(get_session_id)] = None,
) -> AuthUserPublic:
    # switch is allowed when auth is on without requiring the *current* user
    # (matches gh auth switch). If auth is off this is a no-op surface.
    if not is_auth_enabled():
        # Still allow switching/login semantics when auth is off so CLI tests
        # can mint sessions; the gate does not require auth for this path when
        # listed as authenticated-only — mount it public-ish via gate exception.
        pass
    try:
        user, session = get_auth_service().switch(session_id, body.username, body.password)
    except AuthError as exc:
        raise auth_error_http(exc) from exc
    _set_session_cookie(response, session.session_id)
    return user.public()


@router.get("/users", response_model=AuthUserListResponse)
def list_users(
    _admin: Annotated[AuthUser, Depends(require_admin)],
) -> AuthUserListResponse:
    return AuthUserListResponse(users=get_auth_service().list_users())


@router.post("/users", response_model=AuthUserPublic, status_code=201)
def create_user(
    body: CreateUserRequest,
    _admin: Annotated[AuthUser, Depends(require_admin)],
) -> AuthUserPublic:
    try:
        user = get_auth_service().create_user(
            body.username,
            body.password,
            role=body.role,
            workspaces=body.workspaces,
        )
    except AuthError as exc:
        raise auth_error_http(exc) from exc
    return user.public()


@router.patch("/users/{username}", response_model=AuthUserPublic)
def patch_user(
    username: str,
    body: PatchUserRequest,
    _admin: Annotated[AuthUser, Depends(require_admin)],
) -> AuthUserPublic:
    service = get_auth_service()
    try:
        user = service.get_user(username)
        if user is None:
            raise AuthError("user_error", f"user not found: {username}")
        if body.role is not None:
            user = service.set_role(username, body.role)
        if body.workspaces is not None:
            user = service.set_workspaces(username, body.workspaces)
        if body.disabled is not None:
            user = service.set_disabled(username, body.disabled)
    except AuthError as exc:
        raise auth_error_http(exc) from exc
    return user.public()


@router.post("/users/{username}/password", response_model=AuthUserPublic)
def set_user_password(
    username: str,
    body: PasswordRequest,
    _admin: Annotated[AuthUser, Depends(require_admin)],
) -> AuthUserPublic:
    try:
        user = get_auth_service().set_password(username, body.password)
    except AuthError as exc:
        raise auth_error_http(exc) from exc
    return user.public()


@router.delete("/users/{username}", status_code=204)
def delete_user(
    username: str,
    _admin: Annotated[AuthUser, Depends(require_admin)],
) -> None:
    try:
        get_auth_service().delete_user(username)
    except AuthError as exc:
        raise auth_error_http(exc) from exc
