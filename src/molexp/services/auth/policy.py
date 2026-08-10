"""Role + workspace allowlist checks."""

from __future__ import annotations

from molexp.services.auth.models import WRITE_ROLES, AuthUser


def can_write(user: AuthUser) -> bool:
    """Return True when *user* may issue mutating HTTP methods."""
    return user.role in WRITE_ROLES and not user.disabled


def can_manage_users(user: AuthUser) -> bool:
    return user.role == "admin" and not user.disabled


def can_write_operator_config(user: AuthUser) -> bool:
    """Agent keys / operator config writes — admin only in v1."""
    return user.role == "admin" and not user.disabled


def workspace_allowed(user: AuthUser, workspace_key: str) -> bool:
    if user.disabled:
        return False
    if "*" in user.workspaces:
        return True
    return workspace_key in user.workspaces


def filter_workspace_keys(user: AuthUser, keys: list[str]) -> list[str]:
    if "*" in user.workspaces:
        return list(keys)
    allowed = set(user.workspaces)
    return [k for k in keys if k in allowed]


def is_safe_method(method: str) -> bool:
    return method.upper() in {"GET", "HEAD", "OPTIONS"}
