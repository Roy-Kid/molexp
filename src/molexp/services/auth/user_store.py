"""Atomic filesystem user table (``users.json``)."""

from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from molexp.services.auth.models import VALID_ROLES, AuthRole, AuthUser
from molexp.services.auth.passwords import hash_password


def _utc_now_iso() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()


def _chmod(path: Path, mode: int) -> None:
    import contextlib

    with contextlib.suppress(OSError):
        path.chmod(mode)


class UserStore:
    """Load / mutate the ``users.json`` table under an auth root."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.path = root / "users.json"

    def ensure_layout(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        _chmod(self.root, 0o700)

    def load(self) -> dict[str, AuthUser]:
        if not self.path.exists():
            return {}
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}
        if not isinstance(raw, dict):
            return {}
        users_raw = raw.get("users", raw)
        if not isinstance(users_raw, dict):
            return {}
        out: dict[str, AuthUser] = {}
        for _key, value in users_raw.items():
            if not isinstance(value, dict):
                continue
            try:
                user = AuthUser.model_validate(value)
            except Exception:
                continue
            out[user.username] = user
        return out

    def save(self, users: dict[str, AuthUser]) -> None:
        self.ensure_layout()
        payload: dict[str, Any] = {
            "version": 1,
            "users": {name: user.model_dump() for name, user in sorted(users.items())},
        }
        text = json.dumps(payload, indent=2)
        tmp = self.path.with_suffix(".tmp")
        fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
        tmp.replace(self.path)
        _chmod(self.path, 0o600)

    def get(self, username: str) -> AuthUser | None:
        return self.load().get(username)

    def list_users(self) -> list[AuthUser]:
        return sorted(self.load().values(), key=lambda u: u.username)

    def has_any_admin(self) -> bool:
        return any(u.role == "admin" and not u.disabled for u in self.load().values())

    def count(self) -> int:
        return len(self.load())

    def create(
        self,
        username: str,
        password: str,
        *,
        role: AuthRole = "operator",
        workspaces: list[str] | None = None,
    ) -> AuthUser:
        username = username.strip()
        if not username:
            raise ValueError("username must be non-empty")
        if role not in VALID_ROLES:
            raise ValueError(f"invalid role: {role!r}")
        users = self.load()
        if username in users:
            raise ValueError(f"user already exists: {username}")
        # Empty store → first user is always admin (bootstrap).
        if not users:
            role = "admin"
        now = _utc_now_iso()
        user = AuthUser(
            username=username,
            password_hash=hash_password(password),
            role=role,
            workspaces=list(workspaces) if workspaces is not None else ["*"],
            disabled=False,
            created_at=now,
            updated_at=now,
        )
        users[username] = user
        self.save(users)
        return user

    def set_password(self, username: str, password: str) -> AuthUser:
        users = self.load()
        user = users.get(username)
        if user is None:
            raise ValueError(f"user not found: {username}")
        updated = user.model_copy(
            update={
                "password_hash": hash_password(password),
                "updated_at": _utc_now_iso(),
            }
        )
        users[username] = updated
        self.save(users)
        return updated

    def set_role(self, username: str, role: AuthRole) -> AuthUser:
        if role not in VALID_ROLES:
            raise ValueError(f"invalid role: {role!r}")
        users = self.load()
        user = users.get(username)
        if user is None:
            raise ValueError(f"user not found: {username}")
        if user.role == "admin" and role != "admin" and self._admin_count(users) <= 1:
            raise ValueError("cannot demote the last admin")
        updated = user.model_copy(update={"role": role, "updated_at": _utc_now_iso()})
        users[username] = updated
        self.save(users)
        return updated

    def set_workspaces(self, username: str, workspaces: list[str]) -> AuthUser:
        if not workspaces:
            raise ValueError("workspaces must be non-empty (use ['*'] for all)")
        users = self.load()
        user = users.get(username)
        if user is None:
            raise ValueError(f"user not found: {username}")
        updated = user.model_copy(
            update={"workspaces": list(workspaces), "updated_at": _utc_now_iso()}
        )
        users[username] = updated
        self.save(users)
        return updated

    def set_disabled(self, username: str, disabled: bool) -> AuthUser:
        users = self.load()
        user = users.get(username)
        if user is None:
            raise ValueError(f"user not found: {username}")
        if disabled and user.role == "admin" and self._admin_count(users) <= 1:
            raise ValueError("cannot disable the last admin")
        updated = user.model_copy(update={"disabled": disabled, "updated_at": _utc_now_iso()})
        users[username] = updated
        self.save(users)
        return updated

    def delete(self, username: str) -> None:
        users = self.load()
        user = users.get(username)
        if user is None:
            raise ValueError(f"user not found: {username}")
        if user.role == "admin" and self._admin_count(users) <= 1:
            raise ValueError("cannot delete the last admin")
        del users[username]
        self.save(users)

    @staticmethod
    def _admin_count(users: dict[str, AuthUser]) -> int:
        return sum(1 for u in users.values() if u.role == "admin" and not u.disabled)
