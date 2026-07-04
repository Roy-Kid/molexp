"""Run hot-state resolves from the OKF ``_ops/run.json`` sidecar (wsokf-07).

After wsokf-07 the Run-level hot machine state — status, retryable domain,
execution history, ownership (pid/host), heartbeat, current execution id — is
sourced from :class:`molexp.workspace.run_ops.RunOpsState` in ``_ops/run.json``
through :meth:`Run.read_ops`, not from ``RunMetadata`` in ``run.json``. The
identity ``run.json`` stays byte-compatible (additive change), but ``Run.status``
and friends read the ops sidecar.
"""

from __future__ import annotations

from datetime import UTC, datetime

from molexp.workspace.models import RunStatus
from molexp.workspace.run_ops import RunOpsState


class TestStatusReadsFromOps:
    def test_status_reflects_ops_not_run_json(self, run) -> None:
        run.materialize()
        run.update_ops(lambda s: s.model_copy(update={"status": RunStatus.FAILED}))
        # Ops says failed; that is what Run.status returns.
        assert run.status == "failed"
        assert run.read_ops().status is RunStatus.FAILED

    def test_is_retryable_reads_ops(self, run) -> None:
        run.materialize()
        assert run.is_retryable is False
        run.update_ops(lambda s: s.model_copy(update={"status": RunStatus.CANCELLED}))
        assert run.is_retryable is True
        run.update_ops(lambda s: s.model_copy(update={"status": RunStatus.SUCCEEDED}))
        assert run.is_retryable is False

    def test_execution_history_accessor_reads_ops(self, run) -> None:
        run.materialize()
        run.update_ops(
            lambda s: s.model_copy(
                update={
                    "executions": (
                        *s.executions,
                        # ExecutionRecord constructed via RunOpsState round-trip below
                    )
                }
            )
        )
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


class TestLifecyclePersistsOps:
    def test_enter_exit_writes_status_and_ownership_to_ops(self, run) -> None:
        ctx = run.start()
        with ctx:
            state = run.read_ops()
            assert state.status is RunStatus.RUNNING
            assert state.owner_pid is not None
            assert state.owner_host is not None
            assert state.heartbeat_at is not None
            # aware-UTC heartbeat.
            assert state.heartbeat_at.tzinfo is not None
            # An execution record was opened.
            assert state.current_execution_id is not None
            assert any(r.status == "running" for r in state.executions)
        after = run.read_ops()
        assert after.status is RunStatus.SUCCEEDED
        assert after.owner_pid is None
        assert after.finished_at is not None
        assert all(r.status != "running" for r in after.executions)
