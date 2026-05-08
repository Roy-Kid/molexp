"""Workspace-side facade for the agent session catalog + on-disk metadata.

The ``SessionLibrary`` mediates two concerns that previously sat in two
different layers:

- the ``sessions`` section of :class:`AssetCatalog` — a flat, plain-dict
  index of every session this workspace knows about;
- the per-session ``session.json`` file under
  ``<workspace_root>/.subsystems/agent.sessions/<session_id>/``.

Both writes happen atomically from the agent layer's perspective:
``workspace.sessions.create(metadata)`` writes the file and updates the
catalog row in one call. The library accepts duck-typed metadata —
either a pydantic ``BaseModel`` (uses ``model_dump``) or a plain
``dict``. **No** ``molexp.agent`` import: the workspace layer must
remain free of agent-side type dependencies.
"""

from __future__ import annotations

import shutil
from builtins import list as _list
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, cast

from molexp._typing import JSONValue

from .base import _atomic_write_json
from .subsystem import SubsystemStore


class _SessionMetadataLike(Protocol):
    """Duck-typed shape accepted by ``SessionLibrary.create``.

    The agent layer's ``SessionMetadata`` (a pydantic model) satisfies this;
    a plain ``dict`` falls through the ``isinstance(metadata, dict)`` branch
    inside :func:`_to_dict` instead.
    """

    def model_dump(self, *, mode: str = ...) -> dict[str, JSONValue]: ...


if TYPE_CHECKING:
    from .workspace import Workspace

SESSIONS_SUBSYSTEM_KIND = "agent.sessions"
SESSION_METADATA_FILENAME = "session.json"


def _to_dict(metadata: _SessionMetadataLike | dict[str, JSONValue]) -> dict[str, JSONValue]:
    """Project a duck-typed metadata object into a plain dict.

    Accepts either a pydantic-like object (via ``model_dump(mode='json')``)
    or a ``dict``. Raises ``TypeError`` for anything else so callers find
    out at the boundary instead of silently writing a malformed
    ``session.json``.
    """
    if isinstance(metadata, dict):
        # ``metadata.items()`` is typed as ``Iterable[(object, object)]``
        # under ``dict`` invariance plus the union with the Protocol arm;
        # narrow each value back to ``JSONValue`` via ``cast`` since the
        # callers contract is "JSON-shaped or BaseModel-with-JSON-dump".
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


def _project_to_catalog_row(
    payload: dict[str, JSONValue], *, workspace_id: str
) -> dict[str, JSONValue]:
    """Flatten a session-metadata payload into the catalog row schema.

    Keeps the catalog row free of nested objects — the catalog stores
    flat columns only. ``goal_summary`` is best-effort: a structured
    ``goal.description`` if present, else ``""``.
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


class SessionLibrary:
    """Workspace-scoped session view: ``session.json`` + catalog row.

    Each method touches both the on-disk metadata file and the
    ``sessions`` catalog section so the two never diverge from
    workspace consumers' point of view.
    """

    def __init__(self, workspace: Workspace) -> None:
        self._workspace = workspace

    # ── Path vending ──────────────────────────────────────────────────────

    def _store(self) -> SubsystemStore:
        return self._workspace.subsystem_store(SESSIONS_SUBSYSTEM_KIND)

    def _session_dir(self, session_id: str) -> Path:
        if not session_id:
            raise ValueError("session_id must be a non-empty string")
        if "/" in session_id or "\\" in session_id or session_id in {".", ".."}:
            raise ValueError(f"invalid session_id {session_id!r}")
        return self._store().dir() / session_id

    # ── Public API ────────────────────────────────────────────────────────

    def create(
        self,
        metadata: _SessionMetadataLike | dict[str, JSONValue],
        *,
        run_id: str | None = None,
    ) -> dict[str, JSONValue]:
        """Persist metadata + register a catalog row.

        Returns the catalog row that was upserted. ``run_id`` overrides
        whatever ``metadata`` carries (handy when the agent layer wants
        to bind a session to a run after metadata construction).
        """
        payload = _to_dict(metadata)
        if run_id is not None:
            payload = {**payload, "run_id": run_id}
        session_id = str(payload["session_id"])

        session_dir = self._session_dir(session_id)
        session_dir.mkdir(parents=True, exist_ok=True)
        _atomic_write_json(session_dir / SESSION_METADATA_FILENAME, payload)

        row = _project_to_catalog_row(payload, workspace_id=self._workspace.id)
        self._workspace.catalog.upsert_session(row)
        return row

    def list(self) -> _list[dict[str, JSONValue]]:
        """Return every session row from the catalog."""
        return self._workspace.catalog.query_sessions(workspace_id=self._workspace.id)

    def get(self, session_id: str) -> dict[str, JSONValue] | None:
        """Return one session row by id, or ``None`` if absent."""
        rows = self._workspace.catalog.query_sessions(workspace_id=self._workspace.id)
        for row in rows:
            if row.get("session_id") == session_id:
                return row
        return None

    def delete(self, session_id: str) -> None:
        """Remove the catalog row and the on-disk session directory.

        Silent on missing entries — the operation is idempotent.
        """
        session_dir = self._session_dir(session_id)
        if session_dir.exists():
            shutil.rmtree(session_dir)
        self._workspace.catalog.remove_session(session_id)


__all__ = [
    "SESSIONS_SUBSYSTEM_KIND",
    "SESSION_METADATA_FILENAME",
    "SessionLibrary",
]
