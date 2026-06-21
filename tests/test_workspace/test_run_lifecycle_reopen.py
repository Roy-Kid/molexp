"""RED tests for resume-reopen in ``RunLifecycle.enter`` (verb B).

``run.start(execution_id=X)`` where X matches an EXISTING execution record must
REOPEN that record (flip it back to running, clear ``finished_at``, reuse the
``executions/<X>/`` dir) instead of appending a fresh record. A ``start()`` with
no id, or with an id matching nothing, keeps the current append behaviour.

Production code does not exist yet — these tests are expected to fail on the
``len(execution_history)`` / ``.status`` / ``.finished_at`` assertions until the
reopen branch ships.
"""

from __future__ import annotations

import json
from pathlib import Path


def _last_exec_id(run) -> str:
    return run.execution_history[-1].execution_id


def test_reopen_does_not_append_record(run) -> None:
    """Re-entering with an existing exec id leaves history length unchanged."""
    with run.start():
        pass
    exec1 = _last_exec_id(run)
    assert len(run.execution_history) == 1

    with run.start(execution_id=exec1):
        assert len(run.execution_history) == 1  # not appended


def test_reopen_flips_record_to_running(run) -> None:
    """The reopened record reads ``status == "running"`` inside the block."""
    with run.start():
        pass
    exec1 = _last_exec_id(run)

    with run.start(execution_id=exec1):
        hist = run.execution_history
        rec = next(r for r in hist if r.execution_id == exec1)
        assert rec.status == "running"


def test_reopen_clears_finished_at(run) -> None:
    """Reopening clears the previously-stamped ``finished_at``."""
    with run.start():
        pass
    exec1 = _last_exec_id(run)
    closed = next(r for r in run.execution_history if r.execution_id == exec1)
    assert closed.finished_at is not None  # was closed by the first exit

    with run.start(execution_id=exec1):
        rec = next(r for r in run.execution_history if r.execution_id == exec1)
        assert rec.finished_at is None


def test_reopen_rewrites_execution_json_status_running(run) -> None:
    """The on-disk ``executions/<X>/execution.json`` flips back to running."""
    with run.start():
        pass
    exec1 = _last_exec_id(run)
    exec_json = Path(run.run_dir) / "executions" / exec1 / "execution.json"
    assert exec_json.exists()

    with run.start(execution_id=exec1):
        # The reused exec dir's metadata flips back to running …
        payload = json.loads(exec_json.read_text())
        assert payload["status"] == "running"
        # … and this is a genuine reopen, not a fresh append onto the same dir.
        assert len(run.execution_history) == 1


def test_start_without_execution_id_appends(run) -> None:
    """A plain ``start()`` keeps appending a fresh record (current behaviour)."""
    with run.start():
        pass
    assert len(run.execution_history) == 1

    with run.start():
        assert len(run.execution_history) == 2


def test_unknown_execution_id_appends(run) -> None:
    """An exec id matching no record appends rather than reopening."""
    with run.start():
        pass
    assert len(run.execution_history) == 1

    with run.start(execution_id="exec-does-not-exist"):
        assert len(run.execution_history) == 2
