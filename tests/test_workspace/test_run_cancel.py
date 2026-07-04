"""Tests for Run.cancel() public method."""

from __future__ import annotations

from molexp.workspace.run import RunStatus


class TestRunCancel:
    def test_cancel_pending_run(self, run):
        assert run.status == "pending"
        run.cancel()
        assert run.status == RunStatus.CANCELLED
        assert run.status == "cancelled"

    def test_cancel_is_idempotent(self, run):
        run.cancel()
        run.cancel()
        assert run.status == RunStatus.CANCELLED

    def test_cancel_does_not_modify_job_ids(self, run):
        run._update_metadata(executor_info={"job_id": "uuid-123", "scheduler_job_id": "456"})
        run.cancel()
        assert run.metadata.executor_info["job_id"] == "uuid-123"
        assert run.metadata.executor_info["scheduler_job_id"] == "456"
        assert run.status == "cancelled"
