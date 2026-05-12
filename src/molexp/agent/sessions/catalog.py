"""Session catalog — agent-layer combiner of session metadata + workspace storage.

The agent owns sessions; workspace owns generic storage. ``SessionCatalog``
mediates between them: it writes ``session.json`` under
``<workspace_root>/.subsystems/agent.sessions/<session_id>/`` (using
workspace's :class:`SubsystemStore`) and upserts a flat row into the agent's
own session-row index (kept here as a JSON file under the same subsystem
directory, since workspace no longer hosts a sessions section).

Per-session pydantic-ai ``ModelMessage`` history lives alongside as
``model_messages.json`` — the LLM-native conversation context that
:class:`~molexp.agent.session.AgentSession` carries between turns. The
codec lives under :mod:`molexp.agent._pydanticai.messages_codec` so
this file stays free of pydantic-ai imports.

History: this logic used to live in ``molexp.workspace.sessions`` as the
``SessionLibrary`` class. The rectification spec (2026-05-09) moved it up
to the agent layer because it was inherently agent-shaped (knew about
``goal_summary`` projection, agent ``status`` strings, etc.) — workspace's
job is to vend storage primitives, not to interpret session schemas.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, cast

from molexp._typing import JSONValue
from molexp.workspace import atomic_write_json

_SessionRow = dict[str, JSONValue]
_SessionRowList = list[_SessionRow]

if TYPE_CHECKING:
    from molexp.workspace import Workspace

# Legacy on-disk root for flat session storage: ``<root>/sessions/<sid>/``.
# New code should use :class:`molexp.agent.folders.Agent` /
# :class:`molexp.agent.folders.AgentSession`, which mount sessions under
# their owning Agent (``<root>/agents/<aid>/agent_sessions/<sid>/``).
SESSIONS_DIRNAME = "sessions"
SESSION_METADATA_FILENAME = "session.json"
SESSION_INDEX_FILENAME = "_index.json"
MODEL_MESSAGES_FILENAME = "model_messages.json"


class _SessionMetadataLike(Protocol):
    """Duck-typed shape accepted by :meth:`SessionCatalog.create`.

    The agent layer's ``SessionMetadata`` (a pydantic model) satisfies
    this; a plain ``dict`` falls through the ``isinstance(metadata,
    dict)`` branch inside :func:`_to_dict`.
    """

    def model_dump(self, *, mode: str = ...) -> dict[str, JSONValue]: ...


def _to_dict(metadata: _SessionMetadataLike | dict[str, JSONValue]) -> dict[str, JSONValue]:
    """Project a duck-typed metadata object into a plain JSON-shaped dict.

    Accepts either a pydantic-like object (via ``model_dump(mode='json')``)
    or a ``dict``. Raises ``TypeError`` for anything else so callers find
    out at the boundary instead of silently writing a malformed
    ``session.json``.
    """
    if isinstance(metadata, dict):
        out: dict[str, JSONValue] = {}
        for k, v in metadata.items():
            out[str(k)] = cast(JSONValue, v)
        return out
    model_dump = getattr(metadata, "model_dump", None)
    if callable(model_dump):
        result = model_dump(mode="json")
        if isinstance(result, dict):
            out2: dict[str, JSONValue] = {}
            for k, v in result.items():
                out2[str(k)] = cast(JSONValue, v)
            return out2
    raise TypeError(
        f"session metadata must be a pydantic BaseModel or a dict; got {type(metadata).__name__}"
    )


def _project_to_index_row(
    payload: dict[str, JSONValue], *, workspace_id: str
) -> dict[str, JSONValue]:
    """Flatten a session-metadata payload into the index-row schema.

    The session index row is a flat dict with stable column names:
    ``session_id`` / ``workspace_id`` / ``status`` / ``goal_summary`` /
    ``created_at`` / ``updated_at`` / ``run_id``. ``goal_summary`` is
    best-effort: a structured ``goal.description`` if present, else "".
    """
    goal = payload.get("goal")
    if isinstance(goal, dict):
        goal_summary = str(goal.get("description") or "")
    else:
        goal_summary = ""
    return {
        "session_id": str(payload["session_id"]),
        "workspace_id": workspace_id,
        "status": str(payload.get("status") or ""),
        "goal_summary": goal_summary,
        "created_at": payload.get("created_at"),
        "updated_at": payload.get("updated_at"),
        "run_id": payload.get("run_id"),
    }


class SessionCatalog:
    """Workspace-scoped session catalog: metadata + flat row index.

    Each method touches both the on-disk ``session.json`` file and the
    agent's session-row index (``_index.json`` under the same subsystem
    directory) so the two never diverge from agent consumers' point of
    view.

    Attributes:
        workspace: The :class:`molexp.workspace.Workspace` whose
            ``.subsystems/agent.sessions/`` directory backs this catalog.

    Example::

        from molexp.workspace import Workspace
        from molexp.agent.sessions import SessionCatalog, SessionMetadata

        ws = Workspace("./lab")
        catalog = SessionCatalog(ws)
        meta = SessionMetadata(session_id="s1", goal=..., status="running")
        catalog.create(meta, run_id="run-42")
    """

    def __init__(self, workspace: Workspace) -> None:
        self._workspace = workspace

    # ── Path vending ──────────────────────────────────────────────────────

    @property
    def _store_dir(self) -> Path:
        path = self._workspace.root / SESSIONS_DIRNAME
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def _index_path(self) -> Path:
        return self._store_dir / SESSION_INDEX_FILENAME

    def _session_dir(self, session_id: str) -> Path:
        if not session_id:
            raise ValueError("session_id must be a non-empty string")
        if "/" in session_id or "\\" in session_id or session_id in {".", ".."}:
            raise ValueError(f"invalid session_id {session_id!r}")
        return self._store_dir / session_id

    def _load_index(self) -> dict[str, dict[str, JSONValue]]:
        path = self._index_path
        if not path.exists():
            return {}
        try:
            with path.open() as fh:
                data = json.load(fh)
        except (OSError, json.JSONDecodeError):
            return {}
        if not isinstance(data, dict):
            return {}
        out: dict[str, dict[str, JSONValue]] = {}
        for k, v in data.items():
            if isinstance(v, dict):
                out[str(k)] = cast(dict[str, JSONValue], v)
        return out

    def _save_index(self, index: dict[str, dict[str, JSONValue]]) -> None:
        atomic_write_json(self._index_path, index)

    # ── Public API ────────────────────────────────────────────────────────

    def create(
        self,
        metadata: _SessionMetadataLike | dict[str, JSONValue],
        *,
        run_id: str | None = None,
    ) -> dict[str, JSONValue]:
        """Persist metadata + register an index row.

        Returns the index row that was upserted. ``run_id`` overrides
        whatever ``metadata`` carries (handy when the agent layer wants
        to bind a session to a run after metadata construction).
        """
        payload = _to_dict(metadata)
        if run_id is not None:
            payload = {**payload, "run_id": run_id}
        session_id = str(payload["session_id"])

        session_dir = self._session_dir(session_id)
        session_dir.mkdir(parents=True, exist_ok=True)
        atomic_write_json(session_dir / SESSION_METADATA_FILENAME, payload)

        row = _project_to_index_row(payload, workspace_id=self._workspace.id)
        index = self._load_index()
        index[session_id] = row
        self._save_index(index)
        return row

    def list(self) -> _SessionRowList:
        """Return every session row from the index."""
        return list(self._load_index().values())

    def get(self, session_id: str) -> dict[str, JSONValue] | None:
        """Return one session row by id, or ``None`` if absent."""
        return self._load_index().get(session_id)

    def delete(self, session_id: str) -> None:
        """Remove the index row and the on-disk session directory.

        Silent on missing entries — the operation is idempotent.
        """
        session_dir = self._session_dir(session_id)
        if session_dir.exists():
            shutil.rmtree(session_dir)
        index = self._load_index()
        index.pop(session_id, None)
        self._save_index(index)

    # ── pydantic-ai ModelMessage history ──────────────────────────────────

    def read_model_messages(self, session_id: str) -> tuple[Any, ...]:
        """Load the persisted pydantic-ai ``ModelMessage`` tuple for one session.

        Returns the empty tuple when no history file exists (a fresh
        session never persisted, or one that has not yet had a turn).
        Decoding errors propagate; callers that want graceful fallback
        on schema drift can wrap this in ``try/except``.
        """
        path = self._session_dir(session_id) / MODEL_MESSAGES_FILENAME
        if not path.exists():
            return ()
        # Codec lives under the ``_pydanticai/`` firewall — imported
        # lazily so ``import molexp.agent.sessions`` does not eagerly
        # pull in ``pydantic_ai``.
        from molexp.agent._pydanticai.messages_codec import load_model_messages

        return load_model_messages(path.read_bytes())

    def write_model_messages(
        self,
        session_id: str,
        messages: tuple[Any, ...],
    ) -> None:
        """Persist the pydantic-ai ``ModelMessage`` tuple atomically.

        Writes nothing and removes any prior file when ``messages`` is
        empty — keeps the on-disk state tidy and means a never-used
        session leaves no stale ``model_messages.json``.
        """
        session_dir = self._session_dir(session_id)
        session_dir.mkdir(parents=True, exist_ok=True)
        path = session_dir / MODEL_MESSAGES_FILENAME
        if not messages:
            if path.exists():
                path.unlink()
            return
        # Codec lives under the ``_pydanticai/`` firewall — see
        # :func:`read_model_messages` for the reasoning.
        from molexp.agent._pydanticai.messages_codec import dump_model_messages

        payload = dump_model_messages(messages)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_bytes(payload)
        tmp.replace(path)


__all__ = [
    "MODEL_MESSAGES_FILENAME",
    "SESSIONS_DIRNAME",
    "SESSION_METADATA_FILENAME",
    "SessionCatalog",
]
