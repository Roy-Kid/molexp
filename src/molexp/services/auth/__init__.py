"""Filesystem-backed auth for the ``molexp serve`` HTTP surface.

Operator-scoped (not workspace-scoped): users and sessions live under
``~/.molexp/auth/``. CLI (``molexp auth``) and the server share this package;
neither shell may own a second user store.

Public surface is intentionally small — import submodules for stores/policy
internals. The process-level enable flag is set by the server (or tests),
never by domain layers.
"""

from __future__ import annotations

from molexp.services.auth.models import (
    DEFAULT_ADMIN_USERNAME,
    AuthRole,
    AuthUser,
    AuthUserPublic,
    SessionRecord,
)
from molexp.services.auth.paths import AUTH_DIR, SESSIONS_DIR, USERS_PATH, default_auth_root
from molexp.services.auth.service import (
    AuthError,
    AuthService,
    AuthState,
    get_auth_service,
    is_auth_enabled,
    reset_auth_service,
    set_auth_enabled,
    set_auth_root,
)

__all__ = [
    "AUTH_DIR",
    "DEFAULT_ADMIN_USERNAME",
    "SESSIONS_DIR",
    "USERS_PATH",
    "AuthError",
    "AuthRole",
    "AuthService",
    "AuthState",
    "AuthUser",
    "AuthUserPublic",
    "SessionRecord",
    "default_auth_root",
    "get_auth_service",
    "is_auth_enabled",
    "reset_auth_service",
    "set_auth_enabled",
    "set_auth_root",
]
