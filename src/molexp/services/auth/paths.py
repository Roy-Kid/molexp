"""Canonical on-disk locations for the auth store."""

from __future__ import annotations

from pathlib import Path

#: Default operator auth root (sibling of ``~/.molexp/config.json``).
AUTH_DIR = Path.home() / ".molexp" / "auth"
USERS_PATH = AUTH_DIR / "users.json"
SESSIONS_DIR = AUTH_DIR / "sessions"
SECRET_PATH = AUTH_DIR / "secret"


def default_auth_root() -> Path:
    """Return the default ``~/.molexp/auth`` directory."""
    return AUTH_DIR
