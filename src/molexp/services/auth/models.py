"""Auth domain models — pure data, no I/O."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Literal

from pydantic import BaseModel, Field

#: Bootstrap / default username (CLI ``auth login`` default ``-u``, UI placeholder).
DEFAULT_ADMIN_USERNAME = "admin"

AuthRole = Literal["admin", "operator", "viewer"]

VALID_ROLES: frozenset[str] = frozenset({"admin", "operator", "viewer"})

#: Roles allowed to mutate workspace state (POST/PUT/PATCH/DELETE).
WRITE_ROLES: frozenset[str] = frozenset({"admin", "operator"})


class AuthUser(BaseModel):
    """One filesystem user record (password hash included — never serialize to UI)."""

    username: str
    password_hash: str
    role: AuthRole = "operator"
    workspaces: list[str] = Field(default_factory=lambda: ["*"])
    disabled: bool = False
    created_at: str
    updated_at: str

    def public(self) -> AuthUserPublic:
        return AuthUserPublic(
            username=self.username,
            role=self.role,
            workspaces=list(self.workspaces),
            disabled=self.disabled,
            created_at=self.created_at,
            updated_at=self.updated_at,
        )


class AuthUserPublic(BaseModel):
    """Wire-safe user summary (no password hash)."""

    username: str
    role: AuthRole
    workspaces: list[str] = Field(default_factory=list)
    disabled: bool = False
    created_at: str = ""
    updated_at: str = ""


class SessionRecord(BaseModel):
    """Opaque session mapped to a username."""

    session_id: str
    username: str
    created_at: str
    expires_at: str

    def is_expired(self, *, now: datetime | None = None) -> bool:

        current = now or datetime.now(UTC)
        try:
            exp = datetime.fromisoformat(self.expires_at)
        except ValueError:
            return True
        if exp.tzinfo is None:
            exp = exp.replace(tzinfo=UTC)
        return current >= exp
