"""Opaque file-backed sessions under ``auth/sessions/``."""

from __future__ import annotations

import json
import os
import secrets
from datetime import UTC, datetime, timedelta
from pathlib import Path

from molexp.services.auth.models import SessionRecord

DEFAULT_SESSION_TTL_HOURS = 24 * 7  # 7 days


def _utc_now() -> datetime:
    return datetime.now(UTC).replace(microsecond=0)


def _chmod(path: Path, mode: int) -> None:
    import contextlib

    with contextlib.suppress(OSError):
        path.chmod(mode)


class SessionStore:
    """Create / resolve / revoke opaque session tokens as one file each."""

    def __init__(self, root: Path, *, ttl_hours: int = DEFAULT_SESSION_TTL_HOURS) -> None:
        self.root = root
        self.dir = root / "sessions"
        self.ttl_hours = ttl_hours

    def ensure_layout(self) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        _chmod(self.dir, 0o700)

    def _path(self, session_id: str) -> Path:
        # Session ids are url-safe; still refuse path separators.
        if "/" in session_id or "\\" in session_id or ".." in session_id:
            raise ValueError("invalid session id")
        return self.dir / f"{session_id}.json"

    def create(self, username: str) -> SessionRecord:
        self.ensure_layout()
        now = _utc_now()
        session_id = secrets.token_urlsafe(32)
        record = SessionRecord(
            session_id=session_id,
            username=username,
            created_at=now.isoformat(),
            expires_at=(now + timedelta(hours=self.ttl_hours)).isoformat(),
        )
        path = self._path(session_id)
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(record.model_dump_json(indent=2))
        _chmod(path, 0o600)
        return record

    def get(self, session_id: str) -> SessionRecord | None:
        if not session_id:
            return None
        try:
            path = self._path(session_id)
        except ValueError:
            return None
        if not path.exists():
            return None
        try:
            record = SessionRecord.model_validate_json(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError, ValueError):
            return None
        if record.is_expired():
            self.revoke(session_id)
            return None
        return record

    def refresh(self, session_id: str) -> SessionRecord | None:
        record = self.get(session_id)
        if record is None:
            return None
        now = _utc_now()
        updated = record.model_copy(
            update={"expires_at": (now + timedelta(hours=self.ttl_hours)).isoformat()}
        )
        path = self._path(session_id)
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(updated.model_dump_json(indent=2))
        return updated

    def revoke(self, session_id: str) -> None:
        import contextlib

        try:
            path = self._path(session_id)
        except ValueError:
            return
        with contextlib.suppress(OSError):
            path.unlink(missing_ok=True)

    def revoke_user(self, username: str) -> int:
        """Revoke every session for *username*. Returns count deleted."""
        self.ensure_layout()
        n = 0
        for path in self.dir.glob("*.json"):
            try:
                record = SessionRecord.model_validate_json(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError, ValueError):
                continue
            if record.username == username:
                try:
                    path.unlink(missing_ok=True)
                    n += 1
                except OSError:
                    pass
        return n
