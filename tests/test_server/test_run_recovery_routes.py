"""Run-recovery route behavior (bugs 4 + 5 + event read path).

* Bug 5 — the server's start/resume/rerun verbs reap a zombie ``running`` run
  (same-host dead owner pid) before deciding, exactly like the CLI, so UI
  users are not stuck behind a permanent 409.
* Bug 4 — ``POST .../rerun?fresh=true`` persists the bypass-cache request
  marker into the new execution slot (the cross-process channel the worker
  honors).
* Event spine — ``GET .../events`` surfaces the run's recent workspace
  timeline through the shared ``read_workspace_events`` code path.
* Status vocabulary — a legacy ``"completed"`` top-level ``workflow.json``
  status is normalized to ``"succeeded"`` on read.
"""

from __future__ import annotations

import json
import platform
from datetime import UTC, datetime, timedelta
from pathlib import Path

from molexp.workspace.models import RunStatus

_DEAD_PID = 2**22 + 12345  # far beyond default pid_max on macOS/Linux runners


def _prefix(project, experiment) -> str:
    return f"/api/projects/{project.id}/experiments/{experiment.id}/runs"


def _make_zombie(run, *, host: str | None = None, heartbeat_age_s: float | None = None) -> None:
    """Stamp *run* as ``running`` with a dead/stale owner."""
    heartbeat = (
        datetime.now(UTC) - timedelta(seconds=heartbeat_age_s)
        if heartbeat_age_s is not None
        else None
    )
    run.update_ops(
        lambda s: s.model_copy(
            update={
                "status": RunStatus.RUNNING,
                "owner_pid": _DEAD_PID,
                "owner_host": host or platform.node(),
                "heartbeat_at": heartbeat,
            }
        )
    )


class TestServerReapsZombies:
    def test_resume_reaps_dead_same_host_owner_instead_of_409(
        self, client, project, experiment, run
    ) -> None:
        _make_zombie(run)
        resp = client.post(f"{_prefix(project, experiment)}/{run.id}/resume")
        # Reaped to failed → inside the retryable domain → resume proceeds.
        assert resp.status_code == 201, resp.text
        assert run.read_ops().status is RunStatus.FAILED or resp.json()["status"] == "failed"

    def test_rerun_reaps_dead_same_host_owner_instead_of_409(
        self, client, project, experiment, run
    ) -> None:
        _make_zombie(run)
        resp = client.post(f"{_prefix(project, experiment)}/{run.id}/rerun")
        assert resp.status_code == 201, resp.text

    def test_start_reaps_then_still_409_outside_pending(
        self, client, project, experiment, run
    ) -> None:
        # Start owns only pending; a reaped run lands in 'failed', so Start
        # still 409s — but with an honest status instead of a phantom
        # 'running'.
        _make_zombie(run)
        resp = client.post(f"{_prefix(project, experiment)}/{run.id}/run", json={})
        assert resp.status_code == 409
        assert "'failed'" in resp.json()["detail"]


class TestRerunFreshMarker:
    def test_rerun_fresh_persists_bypass_marker(self, client, project, experiment, run) -> None:
        run.update_ops(lambda s: s.model_copy(update={"status": RunStatus.FAILED}))
        resp = client.post(f"{_prefix(project, experiment)}/{run.id}/rerun?fresh=true")
        assert resp.status_code == 201, resp.text
        execution_id = resp.json()["executionId"]
        marker = Path(str(run.run_dir)) / "executions" / execution_id / "fresh.json"
        assert marker.exists()
        assert json.loads(marker.read_text())["bypass_cache"] is True

    def test_rerun_without_fresh_writes_no_marker(self, client, project, experiment, run) -> None:
        run.update_ops(lambda s: s.model_copy(update={"status": RunStatus.FAILED}))
        resp = client.post(f"{_prefix(project, experiment)}/{run.id}/rerun")
        assert resp.status_code == 201, resp.text
        execution_id = resp.json()["executionId"]
        marker = Path(str(run.run_dir)) / "executions" / execution_id / "fresh.json"
        assert not marker.exists()


class TestRunEventsEndpoint:
    def test_events_default_on_shows_creation_milestone(
        self, client, project, experiment, run
    ) -> None:
        """Run-lifecycle events are default-on: a freshly added run already
        carries its run.created milestone — no opt-in step exists."""
        resp = client.get(f"{_prefix(project, experiment)}/{run.id}/events")
        assert resp.status_code == 200
        events = resp.json()
        assert [e["type"] for e in events] == ["run.created"]
        assert run.id in events[0]["refs"]

    def test_events_404_for_unknown_run(self, client, project, experiment) -> None:
        resp = client.get(f"{_prefix(project, experiment)}/nope/events")
        assert resp.status_code == 404


class TestExecutionStatusVocabulary:
    def test_legacy_completed_normalized_to_succeeded(
        self, client, project, experiment, run
    ) -> None:
        from molexp.workspace.models import ExecutionRecord

        exec_id = f"exec-{run.id}"
        record = ExecutionRecord(
            execution_id=exec_id,
            started_at=datetime(2026, 1, 1, tzinfo=UTC),
            status="succeeded",
        )
        run.update_ops(lambda s: s.model_copy(update={"executions": (record,)}))
        wf_dir = Path(str(run.run_dir)) / "executions" / exec_id
        wf_dir.mkdir(parents=True, exist_ok=True)
        (wf_dir / "workflow.json").write_text(
            json.dumps({"execution_id": exec_id, "status": "completed", "task_configs": []})
        )
        resp = client.get(f"{_prefix(project, experiment)}/{run.id}/execution")
        assert resp.status_code == 200
        assert resp.json()["status"] == "succeeded"
