"""Run hot-state read accessors resolve from the OKF ``_ops/run.json`` sidecar (wsokf-07).

After wsokf-07/10 a Run's hot machine state — status, retryable domain, and
execution history — is sourced from :class:`molexp.workspace.run_ops.RunOpsState`
in ``_ops/run.json`` through :meth:`Run.read_ops`, not from ``RunMetadata`` in
``run.json``. This file owns the **read** side: ``Run.status`` /
``Run.is_retryable`` / ``Run.execution_history`` consume the sidecar. (The
lifecycle *write* side + the run.json/_ops split live in
``test_runmetadata_single_source``; ownership + heartbeat in ``test_run_heartbeat``.)
"""

from __future__ import annotations

from datetime import UTC, datetime

from molexp.workspace.models import RunStatus
from molexp.workspace.run_ops import RunOpsState


class TestStatusReadsFromOps:
    def test_status_reads_from_ops_sidecar(self, run) -> None:
        run.materialize()
        run.update_ops(lambda s: s.model_copy(update={"status": RunStatus.FAILED}))
        assert run.status == "failed"
        assert run.read_ops().status is RunStatus.FAILED

    def test_is_retryable_reads_from_ops_sidecar(self, run) -> None:
        run.materialize()
        assert run.is_retryable is False
        run.update_ops(lambda s: s.model_copy(update={"status": RunStatus.CANCELLED}))
        assert run.is_retryable is True
        run.update_ops(lambda s: s.model_copy(update={"status": RunStatus.SUCCEEDED}))
        assert run.is_retryable is False

    def test_execution_history_reads_from_ops_sidecar(self, run) -> None:
        run.materialize()
        state = RunOpsState.model_validate(
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
        run.write_ops(state)
        assert [r.execution_id for r in run.execution_history] == ["exec-a"]
