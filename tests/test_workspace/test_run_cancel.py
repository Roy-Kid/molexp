"""Tests for ``Run.cancel()`` — the canonical stop verb (workspace-owned)."""

from __future__ import annotations

from molexp.workspace.run import RunStatus


class TestRunCancel:
    def test_pending_run_becomes_cancelled(self, run):
        assert run.status == "pending"
        run.cancel()
        assert run.status == RunStatus.CANCELLED

    def test_cancel_is_idempotent(self, run):
        run.cancel()
        run.cancel()
        assert run.status == RunStatus.CANCELLED

    def test_cancel_leaves_run_json_provenance_untouched(self, run):
        run._update_metadata(executor_info={"job_id": "uuid-123", "scheduler_job_id": "456"})
        run.cancel()
        assert run.metadata.executor_info["job_id"] == "uuid-123"
        assert run.metadata.executor_info["scheduler_job_id"] == "456"
        assert run.status == "cancelled"
