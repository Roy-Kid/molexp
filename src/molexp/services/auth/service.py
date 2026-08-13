"""AuthService façade + process-level enable flag.

CLI and server share one service instance (per auth root). The enable flag
is process-global: ``molexp serve --auth`` (or config) sets it before
``create_app`` serves traffic.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from molexp.services.auth.models import (
    DEFAULT_ADMIN_USERNAME,
    AuthRole,
    AuthUser,
    AuthUserPublic,
    SessionRecord,
)
from molexp.services.auth.passwords import verify_password
from molexp.services.auth.paths import default_auth_root
from molexp.services.auth.policy import (
    can_manage_users,
    can_write,
    can_write_operator_config,
    filter_workspace_keys,
    is_safe_method,
    workspace_allowed,
)
from molexp.services.auth.rate_limit import LoginRateLimiter, default_login_limiter
from molexp.services.auth.session_store import DEFAULT_SESSION_TTL_HOURS, SessionStore
from molexp.services.auth.user_store import UserStore

# Process-level state (server + tests). Not persisted.
_auth_enabled: bool = False
_auth_root: Path | None = None
_auth_service: AuthService | None = None


def _session_ttl_hours_from_config() -> int:
    """Read ``auth.session_ttl_hours`` from operator config (default 7 days)."""
    try:
        from molexp.services.operator_config import load_operator_config

        cfg = load_operator_config()
        auth = cfg.get("auth")
        if isinstance(auth, dict):
            raw = auth.get("session_ttl_hours")
            if isinstance(raw, int | float) and int(raw) > 0:
                return int(raw)
    except Exception:
        pass
    return DEFAULT_SESSION_TTL_HOURS


class AuthError(Exception):
    """Auth-domain failure with a stable ``code`` for HTTP mapping."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


@dataclass(frozen=True)
class AuthState:
    """Snapshot for ``GET /api/auth/status`` / ``molexp auth status``."""

    enabled: bool
    authenticated: bool
    user: AuthUserPublic | None


class AuthService:
    """User + session operations against a filesystem auth root."""

    def __init__(
        self,
        root: Path | None = None,
        *,
        ttl_hours: int | None = None,
        rate_limiter: LoginRateLimiter | None = None,
    ) -> None:
        if root is not None:
            self.root = root
        elif _auth_root is not None:
            self.root = _auth_root
        else:
            self.root = default_auth_root()
        self.users = UserStore(self.root)
        hours = ttl_hours if ttl_hours is not None else _session_ttl_hours_from_config()
        self.sessions = SessionStore(self.root, ttl_hours=hours)
        self._limiter = rate_limiter if rate_limiter is not None else default_login_limiter

    def ensure_layout(self) -> None:
        self.users.ensure_layout()
        self.sessions.ensure_layout()

    # ── login / session (gh-shaped) ──────────────────────────────────────

    def login(
        self,
        username: str,
        password: str,
        *,
        client_key: str | None = None,
    ) -> tuple[AuthUser, SessionRecord]:
        """Authenticate *username*/*password* and mint a session.

        *client_key* (typically client host) is rate-limited together with
        the username so both distributed and single-account brute force are
        slowed.
        """
        rate_keys = [f"user:{username}"]
        if client_key:
            rate_keys.append(f"client:{client_key}")
        for key in rate_keys:
            try:
                self._limiter.check(key)
            except ValueError as exc:
                raise AuthError("rate_limited", str(exc)) from exc

        user = self.users.get(username)
        ok = (
            user is not None and not user.disabled and verify_password(user.password_hash, password)
        )
        if not ok:
            for key in rate_keys:
                self._limiter.record_failure(key)
            raise AuthError("invalid_credentials", "Invalid username or password")

        for key in rate_keys:
            self._limiter.record_success(key)
        assert user is not None  # narrowed by ok
        session = self.sessions.create(user.username)
        return user, session

    def logout(self, session_id: str | None) -> None:
        if session_id:
            self.sessions.revoke(session_id)

    def resolve_session(self, session_id: str | None) -> AuthUser | None:
        if not session_id:
            return None
        record = self.sessions.get(session_id)
        if record is None:
            return None
        user = self.users.get(record.username)
        if user is None or user.disabled:
            self.sessions.revoke(session_id)
            return None
        return user

    def refresh(self, session_id: str | None) -> SessionRecord | None:
        if not session_id:
            return None
        return self.sessions.refresh(session_id)

    def switch(
        self,
        session_id: str | None,
        username: str,
        password: str,
    ) -> tuple[AuthUser, SessionRecord]:
        """Log in as *username* and drop the previous session (gh auth switch)."""
        user, new_session = self.login(username, password)
        if session_id:
            self.sessions.revoke(session_id)
        return user, new_session

    def status(self, session_id: str | None, *, enabled: bool | None = None) -> AuthState:
        on = is_auth_enabled() if enabled is None else enabled
        user = self.resolve_session(session_id) if on or session_id else None
        # When auth is off, report unauthenticated unless a session happens to exist.
        if not on:
            return AuthState(enabled=False, authenticated=False, user=None)
        if user is None:
            return AuthState(enabled=True, authenticated=False, user=None)
        return AuthState(enabled=True, authenticated=True, user=user.public())

    def token_for(self, session_id: str | None) -> str | None:
        """Return the opaque session id when valid (``auth token``)."""
        if not session_id:
            return None
        record = self.sessions.get(session_id)
        return record.session_id if record is not None else None

    # ── bootstrap / users ────────────────────────────────────────────────

    def bootstrap_admin(
        self,
        password: str,
        *,
        username: str = DEFAULT_ADMIN_USERNAME,
    ) -> AuthUser:
        """Create the first admin when the store is empty."""
        if self.users.count() > 0:
            raise AuthError("not_empty", "User store is not empty; use users create instead")
        return self.users.create(username, password, role="admin", workspaces=["*"])

    def create_user(
        self,
        username: str,
        password: str,
        *,
        role: AuthRole = "operator",
        workspaces: list[str] | None = None,
    ) -> AuthUser:
        try:
            return self.users.create(username, password, role=role, workspaces=workspaces)
        except ValueError as exc:
            raise AuthError("user_error", str(exc)) from exc

    def list_users(self) -> list[AuthUserPublic]:
        return [u.public() for u in self.users.list_users()]

    def get_user(self, username: str) -> AuthUser | None:
        return self.users.get(username)

    def set_password(self, username: str, password: str) -> AuthUser:
        try:
            user = self.users.set_password(username, password)
        except ValueError as exc:
            raise AuthError("user_error", str(exc)) from exc
        self.sessions.revoke_user(username)
        return user

    def set_role(self, username: str, role: AuthRole) -> AuthUser:
        try:
            return self.users.set_role(username, role)
        except ValueError as exc:
            raise AuthError("user_error", str(exc)) from exc

    def set_workspaces(self, username: str, workspaces: list[str]) -> AuthUser:
        try:
            return self.users.set_workspaces(username, workspaces)
        except ValueError as exc:
            raise AuthError("user_error", str(exc)) from exc

    def set_disabled(self, username: str, disabled: bool) -> AuthUser:
        try:
            user = self.users.set_disabled(username, disabled)
        except ValueError as exc:
            raise AuthError("user_error", str(exc)) from exc
        if disabled:
            self.sessions.revoke_user(username)
        return user

    def delete_user(self, username: str) -> None:
        try:
            self.users.delete(username)
        except ValueError as exc:
            raise AuthError("user_error", str(exc)) from exc
        self.sessions.revoke_user(username)

    # ── policy helpers ───────────────────────────────────────────────────

    def assert_can_write(self, user: AuthUser) -> None:
        if not can_write(user):
            raise AuthError("forbidden", "Write access denied for this role")

    def assert_can_manage_users(self, user: AuthUser) -> None:
        if not can_manage_users(user):
            raise AuthError("forbidden", "Admin role required")

    def assert_can_write_operator_config(self, user: AuthUser) -> None:
        if not can_write_operator_config(user):
            raise AuthError("forbidden", "Admin role required for operator config")

    def assert_workspace_access(self, user: AuthUser, workspace_key: str) -> None:
        if not workspace_allowed(user, workspace_key):
            raise AuthError("forbidden", f"No access to workspace {workspace_key!r}")

    def filter_workspaces(self, user: AuthUser, keys: list[str]) -> list[str]:
        return filter_workspace_keys(user, keys)

    def assert_method_allowed(self, user: AuthUser, method: str) -> None:
        if is_safe_method(method):
            return
        self.assert_can_write(user)

    def has_users(self) -> bool:
        return self.users.count() > 0


def is_auth_enabled() -> bool:
    return _auth_enabled


def set_auth_enabled(enabled: bool) -> None:
    global _auth_enabled
    _auth_enabled = enabled


def set_auth_root(root: Path | None) -> None:
    """Override auth root (tests). Resets the cached service."""
    global _auth_root, _auth_service
    _auth_root = root
    _auth_service = None


def get_auth_service() -> AuthService:
    global _auth_service
    if _auth_service is None:
        root = _auth_root if _auth_root is not None else default_auth_root()
        _auth_service = AuthService(root)
    return _auth_service


def reset_auth_service() -> None:
    """Clear process auth state (tests / shutdown)."""
    global _auth_service, _auth_enabled, _auth_root
    _auth_service = None
    _auth_enabled = False
    _auth_root = None
