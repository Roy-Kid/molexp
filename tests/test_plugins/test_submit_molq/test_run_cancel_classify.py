"""Unit tests for ``molexp.plugins.submit_molq.cancel.classify``.

``classify`` is pure inspection: it maps a run's ops/executor state to a
:class:`CancelPlan` without executing anything. One test per branch of the
classification contract (molq / local-same-host / uncancellable reasons).
"""

from __future__ import annotations

import platform

import pytest

from molexp.plugins.submit_molq.cancel import classify
from molexp.workspace import Workspace
from molexp.workspace.models import RunStatus


@pytest.fixture
def running_run(tmp_path):
    ws = Workspace(root=tmp_path, name="lab")
    ws.materialize()
    e = ws.add_project("p").add_experiment("e", workflow_source="s.py", params={})
    r = e.add_run(params={"seed": 1})
    # Hot state (status / ownership) lives in the OKF _ops sidecar (wsokf-10).
    r.update_ops(lambda s: s.model_copy(update={"status": RunStatus.RUNNING}))
    return r


class TestClassify:
    def test_molq_backend_with_job_id_classifies_molq(self, running_run):
        running_run._update_metadata(
            executor_info={
                "backend": "molq",
                "scheduler": "slurm",
                "cluster_name": "hpc",
                "job_id": "abc-uuid-123",
                "scheduler_job_id": "88001",
            }
        )
        plan = classify(running_run)
        assert plan.kind == "molq"
        assert plan.detail == "hpc"
        assert plan.job_id == "abc-uuid-123"

    def test_local_pid_same_host_classifies_local(self, running_run):
        running_run.update_ops(
            lambda s: s.model_copy(update={"owner_pid": 12345, "owner_host": platform.node()})
        )
        plan = classify(running_run)
        assert plan.kind == "local"
        assert plan.detail == "12345"

    def test_pid_on_different_host_is_uncancellable(self, running_run):
        running_run.update_ops(
            lambda s: s.model_copy(
                update={"owner_pid": 12345, "owner_host": "some-other-host.example"}
            )
        )
        plan = classify(running_run)
        assert plan.kind == "none"
        assert "different host" in plan.detail

    def test_terminal_status_is_uncancellable(self, running_run):
        running_run.update_ops(lambda s: s.model_copy(update={"status": RunStatus.SUCCEEDED}))
        plan = classify(running_run)
        assert plan.kind == "none"
        assert plan.detail == "already terminal"

    def test_no_pid_or_scheduler_info_is_uncancellable(self, running_run):
        plan = classify(running_run)
        assert plan.kind == "none"
