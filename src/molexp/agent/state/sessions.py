"""Session metadata + JSONL log stores.

The harness writes three independent JSONL streams per session:

- ``messages.jsonl`` — semantic ``Message`` records (harness-owned).
- ``events.jsonl`` — append-only event log.
- ``model_io.jsonl`` — raw model request/response (model plugin owned).

The harness only reads ``messages.jsonl``; the model plugin owns
``model_io.jsonl`` exclusively.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field, is_dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Iterable

from molexp.agent.types import Goal, Message, SessionStatus, utc_now


@dataclass(frozen=True)
class SessionMetadata:
    """Latest summary persisted to ``session.json``."""

    session_id: str
    goal: Goal
    status: SessionStatus
    created_at: datetime = field(default_factory=utc_now)
    updated_at: datetime = field(default_factory=utc_now)
    summary: str = ""


class SessionStore:
    """Per-workspace persistence for sessions.

    The store is layout-only — it does not run the session. The
    orchestration layer constructs a ``SessionMetadata`` and asks the
    store to flush it after every turn.
    """

    METADATA_FILENAME = "session.json"
    MESSAGES_FILENAME = "messages.jsonl"
    EVENTS_FILENAME = "events.jsonl"
    MODEL_IO_FILENAME = "model_io.jsonl"
    PROVIDER_BLOBS_DIRNAME = "provider_blobs"
    CHECKPOINTS_DIRNAME = "checkpoints"
    ARTIFACTS_DIRNAME = "artifacts"

    def __init__(self, root: Path) -> None:
        self._root = root
        self._root.mkdir(parents=True, exist_ok=True)

    def session_dir(self, session_id: str) -> Path:
        path = self._root / session_id
        path.mkdir(parents=True, exist_ok=True)
        return path

    def write_metadata(self, meta: SessionMetadata) -> None:
        path = self.session_dir(meta.session_id) / self.METADATA_FILENAME
        payload = _to_json_safe(meta)
        _atomic_write_json(path, payload)

    def read_metadata(self, session_id: str) -> SessionMetadata | None:
        path = self.session_dir(session_id) / self.METADATA_FILENAME
        if not path.exists():
            return None
        with path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        return _metadata_from_json(payload)

    def list_sessions(self) -> tuple[SessionMetadata, ...]:
        if not self._root.exists():
            return ()
        out: list[SessionMetadata] = []
        for child in sorted(self._root.iterdir()):
            if not child.is_dir():
                continue
            meta = self.read_metadata(child.name)
            if meta is not None:
                out.append(meta)
        return tuple(out)

    # JSONL streams ----------------------------------------------------

    def append_messages(self, session_id: str, messages: Iterable[Message]) -> None:
        path = self.session_dir(session_id) / self.MESSAGES_FILENAME
        _append_jsonl(path, (_to_json_safe(m) for m in messages))

    def read_messages(self, session_id: str) -> tuple[Message, ...]:
        path = self.session_dir(session_id) / self.MESSAGES_FILENAME
        if not path.exists():
            return ()
        out: list[Message] = []
        with path.open("r", encoding="utf-8") as f:
            for raw in f:
                raw = raw.strip()
                if not raw:
                    continue
                payload = json.loads(raw)
                out.append(
                    Message(
                        role=payload["role"],
                        content=payload["content"],
                        name=payload.get("name"),
                        metadata=payload.get("metadata", {}),
                    )
                )
        return tuple(out)

    def append_event(self, session_id: str, event: Any) -> None:
        path = self.session_dir(session_id) / self.EVENTS_FILENAME
        _append_jsonl(path, [_to_json_safe(event)])

    def append_model_io(self, session_id: str, payload: dict[str, Any]) -> None:
        """Plugin-only entry point for the raw model_io layer."""

        path = self.session_dir(session_id) / self.MODEL_IO_FILENAME
        _append_jsonl(path, [payload])

    def write_checkpoint(
        self, session_id: str, name: str, payload: dict[str, Any]
    ) -> Path:
        """Atomically write a per-session checkpoint JSON file."""

        ckpt_dir = self.session_dir(session_id) / self.CHECKPOINTS_DIRNAME
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        path = ckpt_dir / f"{name}.json"
        _atomic_write_json(path, _to_json_safe(payload))
        return path


# Serialization helpers -------------------------------------------------

def _to_json_safe(obj: Any) -> Any:
    if is_dataclass(obj) and not isinstance(obj, type):
        payload = asdict(obj)
        payload["__type__"] = type(obj).__name__
        return _to_json_safe(payload)
    if isinstance(obj, dict):
        return {k: _to_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_json_safe(v) for v in obj]
    if isinstance(obj, datetime):
        return obj.isoformat()
    if isinstance(obj, Enum):
        return obj.value
    return obj


def _atomic_write_json(path: Path, payload: Any) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def _append_jsonl(path: Path, lines: Iterable[Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for line in lines:
            f.write(json.dumps(line, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


def _metadata_from_json(payload: dict[str, Any]) -> SessionMetadata:
    goal_payload = payload["goal"]
    goal = Goal(
        description=goal_payload["description"],
        constraints=goal_payload.get("constraints", {}),
        success_criteria=goal_payload.get("success_criteria", []),
        instructions_override=goal_payload.get("instructions_override"),
        skill_id=goal_payload.get("skill_id"),
    )
    return SessionMetadata(
        session_id=payload["session_id"],
        goal=goal,
        status=SessionStatus(payload["status"]),
        created_at=datetime.fromisoformat(payload["created_at"]),
        updated_at=datetime.fromisoformat(payload["updated_at"]),
        summary=payload.get("summary", ""),
    )
