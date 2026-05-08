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
from collections.abc import Iterable
from pathlib import Path

from molexp._typing import HashablePayload, JSONValue
from molexp.agent._legacy_types import to_jsonable
from molexp.agent.sessions.types import SessionMetadata
from molexp.agent.types import Message


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
        _atomic_write_json(path, to_jsonable(meta))

    def read_metadata(self, session_id: str) -> SessionMetadata | None:
        path = self.session_dir(session_id) / self.METADATA_FILENAME
        if not path.exists():
            return None
        with path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        return SessionMetadata.model_validate(payload)

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
        _append_jsonl(path, (to_jsonable(m) for m in messages))

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
                out.append(Message.model_validate_json(raw))
        return tuple(out)

    def append_event(self, session_id: str, event: HashablePayload) -> None:
        path = self.session_dir(session_id) / self.EVENTS_FILENAME
        _append_jsonl(path, [to_jsonable(event)])

    def append_model_io(self, session_id: str, payload: dict[str, JSONValue]) -> None:
        """Plugin-only entry point for the raw model_io layer."""

        path = self.session_dir(session_id) / self.MODEL_IO_FILENAME
        _append_jsonl(path, [payload])

    def write_checkpoint(self, session_id: str, name: str, payload: dict[str, JSONValue]) -> Path:
        """Atomically write a per-session checkpoint JSON file."""

        ckpt_dir = self.session_dir(session_id) / self.CHECKPOINTS_DIRNAME
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        path = ckpt_dir / f"{name}.json"
        _atomic_write_json(path, to_jsonable(payload))
        return path


def _atomic_write_json(path: Path, payload: JSONValue) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def _append_jsonl(path: Path, lines: Iterable[JSONValue]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for line in lines:
            f.write(json.dumps(line, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())
