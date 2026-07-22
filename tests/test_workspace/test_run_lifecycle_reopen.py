"""Resume-reopen branch of ``RunLifecycle.enter`` (resume verb B).

``run.start(execution_id=X)`` where X matches an EXISTING execution record
REOPENS that record — flips it back to running, clears ``finished_at``, and
reuses the ``executions/<X>/`` dir — instead of appending a fresh record. A
``start()`` with no id, or with an id matching nothing, keeps the append
behaviour (rerun / first attempt).
"""

from __future__ import annotations

import json
from pathlib import Path


def _last_exec_id(run) -> str:
    return run.execution_history[-1].execution_id


class TestReopenExecution:
    def test_reopen_clears_finished_at(self, run) -> None:
        with run.start():
            pass
        exec1 = _last_exec_id(run)
        closed = next(r for r in run.execution_history if r.execution_id == exec1)
        assert closed.finished_at is not None  # closed by the first exit

        with run.start(execution_id=exec1):
            rec = next(r for r in run.execution_history if r.execution_id == exec1)
            assert rec.finished_at is None

    def test_reopen_rewrites_execution_json_status_running(self, run) -> None:
        with run.start():
            pass
        exec1 = _last_exec_id(run)
        exec_json = Path(run.run_dir) / "executions" / exec1 / "execution.json"
        assert exec_json.exists()

        with run.start(execution_id=exec1):
            payload = json.loads(exec_json.read_text())
            assert payload["status"] == "running"
            # A genuine reopen, not a fresh append onto the same dir.
            assert len(run.execution_history) == 1

    def test_start_without_execution_id_appends(self, run) -> None:
        with run.start():
            pass
        assert len(run.execution_history) == 1

        with run.start():
            assert len(run.execution_history) == 2

    def test_unknown_execution_id_appends(self, run) -> None:
        with run.start():
            pass
        assert len(run.execution_history) == 1

        with run.start(execution_id="exec-does-not-exist"):
            assert len(run.execution_history) == 2
