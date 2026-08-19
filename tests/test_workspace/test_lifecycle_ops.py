"""``molexp.workspace.lifecycle_ops.cancel_run`` — the canonical intervene verb.

vision-loop-07 §2: the workspace-layer core the ``molexp.lifecycle.run_cancel``
capability (and any future CLI/server rewiring) shares. Its own contract, tested
once here:

* **reap first** — every verb entry reaps (`workspace.run_reaper`): a stale
  ``running`` run with a dead same-host owner flips to ``failed`` *before* the
  verb decides, so a zombie is never "cancelled";
* **running only** — any non-``running`` status after reaping refuses loudly
  (``ValueError``);
* a genuine cancel flips status to ``cancelled`` and clears the ownership stamp
  in the ``ops`` sidecar (the low-level ``Run.cancel`` flip reached through the guard).

The reaper's own rules (dead-pid clears ownership, cross-host stale-heartbeat) are
owned by ``tests/test_cli/test_reap_zombie.py``; ``is_retryable`` by
``test_run_ops_consumers.py`` — not re-tested here.
"""

from __future__ import annotations

import os
import platform
from datetime import UTC, datetime
from pathlib import Path

import pytest

from molexp.workspace import Workspace
from molexp.workspace.lifecycle_ops import cancel_run
from molexp.workspace.models import RunStatus
from molexp.workspace.run import Run
from molexp.workspace.run_ops import RunOpsState

# ── fixtures ─────────────────────────────────────────────────────────────────


@pytest.fixture
def run(tmp_path: Path) -> Run:
    ws = Workspace(root=tmp_path, name="cancel-lab")
    exp = ws.add_project("p").add_experiment("e", workflow_source="s.py", params={})
    new_run = exp.add_run(params={"seed": 1})
    new_run.materialize()
    return new_run


def _mark_running(run: Run, *, pid: int, host: str, heartbeat_at: datetime | None) -> None:
    run.write_ops(
        RunOpsState(
            status="running",
            owner_pid=pid,
            owner_host=host,
            heartbeat_at=heartbeat_at,
        )
    )


def _dead_pid() -> int:
    """A pid that is exceedingly unlikely to exist on this host."""
    pid = 2**22 - 7
    with pytest.raises(ProcessLookupError):
        os.kill(pid, 0)
    return pid


class TestCancelRunning:
    def test_live_running_run_is_cancelled(self, run: Run) -> None:
        _mark_running(run, pid=os.getpid(), host=platform.node(), heartbeat_at=datetime.now(UTC))
        cancel_run(run)
        assert run.status == "cancelled"

    def test_cancel_clears_the_ownership_stamp(self, run: Run) -> None:
        _mark_running(run, pid=os.getpid(), host=platform.node(), heartbeat_at=datetime.now(UTC))
        cancel_run(run)
        state = run.read_ops()
        assert state.owner_pid is None
        assert state.owner_host is None
        assert state.heartbeat_at is None


class TestRefusesNonRunning:
    @pytest.mark.parametrize(
        "status",
        [RunStatus.PENDING, RunStatus.SUCCEEDED, RunStatus.FAILED, RunStatus.CANCELLED],
        ids=["pending", "succeeded", "failed", "cancelled"],
    )
    def test_non_running_status_refuses_and_preserves_status(
        self, run: Run, status: RunStatus
    ) -> None:
        run.update_ops(lambda s: s.model_copy(update={"status": status}))
        with pytest.raises(ValueError, match="running"):
            cancel_run(run)
        assert run.status == status.value


class TestZombieReapedFirst:
    def test_dead_owner_is_reaped_then_refused(self, run: Run) -> None:
        """Same-host dead pid → reap flips to failed → cancel refuses (not running)."""
        _mark_running(run, pid=_dead_pid(), host=platform.node(), heartbeat_at=datetime.now(UTC))
        with pytest.raises(ValueError, match="failed"):
            cancel_run(run)
        assert run.status == "failed"
