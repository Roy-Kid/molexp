"""``SessionCatalog`` round-trips session metadata through workspace storage.

Phase 1 of the rectification spec moved the workspace-side session
library (which forced an agent-shaped surface into the storage layer)
up to the agent layer as ``SessionCatalog``. This test pins the new
contract:

- ``create()`` writes a ``session.json`` under
  ``<workspace>/.subsystems/agent.sessions/<session_id>/`` AND
  registers a row in the agent's own ``_index.json``.
- ``list()`` / ``get()`` read the index.
- ``delete()`` removes both the row and the on-disk dir.
- The catalog accepts duck-typed metadata: pydantic-like (uses
  ``model_dump``) or plain ``dict``.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from molexp.agent.sessions import (
    SESSIONS_DIRNAME,
    SessionCatalog,
)
from molexp.workspace import Workspace


@pytest.fixture
def workspace(tmp_path: Path) -> Workspace:
    ws = Workspace(tmp_path / "lab")
    ws.materialize()
    return ws


def _session_dict(session_id: str = "s-1", **extras: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "session_id": session_id,
        "status": "running",
        "goal": {"description": "probe"},
        "created_at": "2026-05-09T00:00:00Z",
        "updated_at": "2026-05-09T00:00:00Z",
    }
    payload.update(extras)
    return payload


def test_create_writes_session_json_under_subsystem_dir(workspace: Workspace) -> None:
    catalog = SessionCatalog(workspace)
    catalog.create(_session_dict("s-1"))

    expected = workspace.root / SESSIONS_DIRNAME / "s-1" / "session.json"
    assert expected.exists(), f"session.json should live at {expected}"
    payload = json.loads(expected.read_text())
    assert payload["session_id"] == "s-1"
    assert payload["status"] == "running"


def test_create_registers_index_row(workspace: Workspace) -> None:
    catalog = SessionCatalog(workspace)
    row = catalog.create(_session_dict("s-2"))

    assert row["session_id"] == "s-2"
    assert row["workspace_id"] == workspace.id
    assert row["status"] == "running"
    assert row["goal_summary"] == "probe"


def test_list_returns_every_registered_session(workspace: Workspace) -> None:
    catalog = SessionCatalog(workspace)
    catalog.create(_session_dict("s-a"))
    catalog.create(_session_dict("s-b"))

    rows = catalog.list()
    assert sorted(r["session_id"] for r in rows) == ["s-a", "s-b"]


def test_get_returns_existing_row_or_none(workspace: Workspace) -> None:
    catalog = SessionCatalog(workspace)
    catalog.create(_session_dict("s-c"))

    hit = catalog.get("s-c")
    assert hit is not None
    assert hit["session_id"] == "s-c"

    miss = catalog.get("does-not-exist")
    assert miss is None


def test_delete_drops_row_and_directory(workspace: Workspace) -> None:
    catalog = SessionCatalog(workspace)
    catalog.create(_session_dict("s-d"))
    session_dir = workspace.root / SESSIONS_DIRNAME / "s-d"
    assert session_dir.exists()

    catalog.delete("s-d")
    assert catalog.get("s-d") is None
    assert not session_dir.exists()


def test_delete_is_idempotent(workspace: Workspace) -> None:
    catalog = SessionCatalog(workspace)
    catalog.delete("never-created")  # should not raise


def test_run_id_override_at_create_time(workspace: Workspace) -> None:
    catalog = SessionCatalog(workspace)
    row = catalog.create(_session_dict("s-e", run_id="will-be-overridden"), run_id="actual-run")
    assert row["run_id"] == "actual-run"


def test_pydantic_like_metadata_accepted(workspace: Workspace) -> None:
    """Anything implementing ``model_dump(mode='json')`` is accepted."""

    class _PydanticLike:
        def model_dump(self, *, mode: str = "python") -> dict[str, object]:
            return _session_dict("s-pyd")

    catalog = SessionCatalog(workspace)
    row = catalog.create(_PydanticLike())  # type: ignore[arg-type]
    assert row["session_id"] == "s-pyd"


def test_invalid_metadata_type_raises(workspace: Workspace) -> None:
    catalog = SessionCatalog(workspace)
    with pytest.raises(TypeError, match="pydantic BaseModel or a dict"):
        catalog.create(42)  # type: ignore[arg-type]


def test_invalid_session_id_rejected(workspace: Workspace) -> None:
    catalog = SessionCatalog(workspace)
    for bad in ("../escape", "a/b", "."):
        with pytest.raises(ValueError):
            catalog.create(_session_dict(bad))


# ── pydantic-ai ModelMessage round trip ───────────────────────────────────


def test_read_model_messages_missing_returns_empty(workspace: Workspace) -> None:
    """Sessions that never persisted history read back as the empty tuple."""
    catalog = SessionCatalog(workspace)
    assert catalog.read_model_messages("never-written") == ()


def test_write_then_read_round_trips_pydantic_ai_messages(workspace: Workspace) -> None:
    """Persisted pydantic-ai messages survive a write/read cycle on disk."""
    pytest.importorskip("pydantic_ai")
    from pydantic_ai.messages import ModelRequest, ModelResponse, TextPart, UserPromptPart

    catalog = SessionCatalog(workspace)
    original = (
        ModelRequest(parts=[UserPromptPart(content="ping")]),
        ModelResponse(parts=[TextPart(content="pong")]),
    )
    catalog.write_model_messages("s-mm", original)

    expected_path = (
        workspace.root / SESSIONS_DIRNAME / "s-mm" / "model_messages.json"
    )
    assert expected_path.exists(), "write should leave a model_messages.json on disk"

    restored = catalog.read_model_messages("s-mm")
    assert list(restored) == list(original)


def test_write_empty_messages_deletes_existing_file(workspace: Workspace) -> None:
    """Persisting ``()`` is idempotent cleanup, not a stale empty file."""
    pytest.importorskip("pydantic_ai")
    from pydantic_ai.messages import ModelRequest, UserPromptPart

    catalog = SessionCatalog(workspace)
    catalog.write_model_messages("s-clean", (ModelRequest(parts=[UserPromptPart(content="x")]),))
    path = (
        workspace.root / SESSIONS_DIRNAME / "s-clean" / "model_messages.json"
    )
    assert path.exists()

    catalog.write_model_messages("s-clean", ())
    assert not path.exists()
    assert catalog.read_model_messages("s-clean") == ()
