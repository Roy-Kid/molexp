"""Server run routes reflect the OKF ``_ops`` hot-state sidecar (wsokf-07).

``get_run`` / ``list_runs`` status, the ``update_run_status`` write, the
``cancel`` verb, and the resume/rerun 409 gate all resolve from
``Run.read_ops()`` / ``RunOpsState`` instead of ``RunMetadata.status``.
``RunResponse.from_model`` maps ``run.read_ops().executions``.
"""

from __future__ import annotations

from datetime import UTC, datetime

from molexp.workspace.models import RunStatus
from molexp.workspace.run_ops import RunOpsState


def _prefix(project, experiment) -> str:
    return f"/api/projects/{project.id}/experiments/{experiment.id}/runs"


class TestStatusReflectsOps:
    def test_get_run_status_from_ops(self, client, project, experiment, run) -> None:
        run.update_ops(lambda s: s.model_copy(update={"status": RunStatus.FAILED}))
        resp = client.get(f"{_prefix(project, experiment)}/{run.id}")
        assert resp.status_code == 200
        assert resp.json()["status"] == "failed"

    def test_execution_history_from_ops(self, client, project, experiment, run) -> None:
        run.write_ops(
            RunOpsState.model_validate(
                {
                    "status": "failed",
                    "executions": [
                        {
                            "execution_id": "exec-a",
                            "started_at": datetime(2026, 1, 1, tzinfo=UTC).isoformat(),
                            "status": "failed",
                        }
                    ],
                }
            )
        )
        resp = client.get(f"{_prefix(project, experiment)}/{run.id}")
        assert resp.status_code == 200
        history = resp.json()["executionHistory"]
        assert [h["executionId"] for h in history] == ["exec-a"]

    def test_update_status_writes_ops(self, client, project, experiment, run) -> None:
        resp = client.patch(
            f"{_prefix(project, experiment)}/{run.id}/status",
            json={"status": "succeeded"},
        )
        assert resp.status_code == 200
        assert resp.json()["status"] == "succeeded"
        assert run.read_ops().status.value == "succeeded"

    def test_cancel_reflects_ops(self, client, project, experiment, run) -> None:
        resp = client.post(f"{_prefix(project, experiment)}/{run.id}/cancel")
        assert resp.status_code == 200
        assert resp.json()["status"] == "cancelled"
        assert run.read_ops().status.value == "cancelled"


class TestRetryableGate:
    def test_resume_409_when_not_retryable(self, client, project, experiment, run) -> None:
        # pending run → not in RETRYABLE_STATUSES → 409.
        run.update_ops(lambda s: s.model_copy(update={"status": RunStatus.PENDING}))
        resp = client.post(f"{_prefix(project, experiment)}/{run.id}/resume")
        assert resp.status_code == 409

    def test_rerun_409_when_succeeded(self, client, project, experiment, run) -> None:
        run.update_ops(lambda s: s.model_copy(update={"status": RunStatus.SUCCEEDED}))
        resp = client.post(f"{_prefix(project, experiment)}/{run.id}/rerun")
        assert resp.status_code == 409
