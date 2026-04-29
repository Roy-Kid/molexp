"""Unit tests for the on-disk session metadata store."""

from __future__ import annotations

import json

import pytest

from molexp.plugins.agent_pydanticai.sessions_store import (
    METADATA_FILE,
    SESSIONS_DIR_NAME,
    get_persisted_session,
    list_persisted_sessions,
    write_session_metadata,
)


@pytest.mark.unit
def test_list_returns_empty_when_no_sessions_dir(tmp_path):
    assert list_persisted_sessions(tmp_path) == []
    # And the call should have created the dir for future writes.
    assert (tmp_path / SESSIONS_DIR_NAME).is_dir()


@pytest.mark.unit
def test_write_then_list_round_trip(tmp_path):
    write_session_metadata(
        tmp_path,
        "sess-1",
        status="completed",
        goal_description="goal one",
        constraints={"scope": "project"},
        success_criteria=["criterion"],
        created_at="2026-04-28T10:00:00Z",
        completed_at="2026-04-28T10:05:00Z",
    )
    rows = list_persisted_sessions(tmp_path)
    assert len(rows) == 1
    row = rows[0]
    assert row.session_id == "sess-1"
    assert row.status == "completed"
    assert row.goal_description == "goal one"
    assert row.created_at == "2026-04-28T10:00:00Z"
    assert row.completed_at == "2026-04-28T10:05:00Z"


@pytest.mark.unit
def test_list_sorted_descending_by_created_at(tmp_path):
    write_session_metadata(
        tmp_path,
        "older",
        status="completed",
        goal_description="x",
        created_at="2026-01-01T00:00:00Z",
    )
    write_session_metadata(
        tmp_path,
        "newer",
        status="completed",
        goal_description="x",
        created_at="2026-04-01T00:00:00Z",
    )
    rows = list_persisted_sessions(tmp_path)
    assert [r.session_id for r in rows] == ["newer", "older"]


@pytest.mark.unit
def test_write_overwrites_atomically(tmp_path):
    """Re-writing reflects the new state and never leaves a half-written file."""
    write_session_metadata(
        tmp_path,
        "sess-1",
        status="running",
        goal_description="g",
    )
    write_session_metadata(
        tmp_path,
        "sess-1",
        status="completed",
        goal_description="g",
        completed_at="2026-04-28T11:00:00Z",
    )
    rows = list_persisted_sessions(tmp_path)
    assert len(rows) == 1
    assert rows[0].status == "completed"
    assert rows[0].completed_at == "2026-04-28T11:00:00Z"
    # No leftover .tmp file from the second write.
    session_dir = tmp_path / SESSIONS_DIR_NAME / "sess-1"
    assert (session_dir / METADATA_FILE).exists()
    assert not (session_dir / "metadata.tmp").exists()


@pytest.mark.unit
def test_list_skips_corrupt_metadata(tmp_path):
    """One bad metadata.json must not blank out the entire listing."""
    write_session_metadata(tmp_path, "sess-good", status="completed", goal_description="g")
    bad_dir = tmp_path / SESSIONS_DIR_NAME / "sess-bad"
    bad_dir.mkdir(parents=True)
    (bad_dir / METADATA_FILE).write_text("{not json")
    rows = list_persisted_sessions(tmp_path)
    assert len(rows) == 1
    assert rows[0].session_id == "sess-good"


@pytest.mark.unit
def test_get_persisted_session_returns_none_when_missing(tmp_path):
    assert get_persisted_session(tmp_path, "nope") is None


@pytest.mark.unit
def test_get_persisted_session_round_trip(tmp_path):
    write_session_metadata(
        tmp_path,
        "sess-x",
        status="failed",
        goal_description="boom",
        created_at="2026-04-28T12:00:00Z",
    )
    summary = get_persisted_session(tmp_path, "sess-x")
    assert summary is not None
    assert summary.status == "failed"
    assert summary.goal_description == "boom"


@pytest.mark.unit
def test_metadata_payload_format(tmp_path):
    """Pin the on-disk format so a runtime change forces an explicit update."""
    write_session_metadata(
        tmp_path,
        "sess-1",
        status="running",
        goal_description="g",
        constraints={"scope": "project"},
        success_criteria=["c1"],
        created_at="t0",
    )
    raw = json.loads((tmp_path / SESSIONS_DIR_NAME / "sess-1" / METADATA_FILE).read_text())
    assert raw["session_id"] == "sess-1"
    assert raw["status"] == "running"
    assert raw["goal"]["description"] == "g"
    assert raw["goal"]["constraints"] == {"scope": "project"}
    assert raw["goal"]["success_criteria"] == ["c1"]
    assert raw["created_at"] == "t0"
    # No completed_at when not provided.
    assert "completed_at" not in raw
