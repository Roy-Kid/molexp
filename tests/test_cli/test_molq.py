"""Plugin unit tests for molq-backed execution.

Scope kept here (not duplicated in ``test_plugins/test_submit_molq`` or
``test_run_cancel``):
  * ``make_submit_handler`` persists the molq ``executor_info`` block onto the
    run (the plugin's provenance-write behaviour).

Cancel-via-molq wiring is owned by ``test_run_cancel.py``; transport/staging
wiring by ``test_plugins/test_submit_molq/test_submit_remote.py``.
"""

from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace

from molexp.plugins.submit_molq.submit import make_submit_handler
from molexp.workspace import Workspace


def _make_workspace(tmp_path):
    workspace = Workspace(root=tmp_path / "workspace", name="Test")
    project = workspace.add_project("demo")
    experiment = project.add_experiment("train")
    run = experiment.add_run(params={"seed": 0})
    return workspace, experiment, run


def test_submit_handler_persists_executor_info(monkeypatch, tmp_path):
    _workspace, _experiment, run = _make_workspace(tmp_path)
    script = tmp_path / "train.py"
    script.write_text("x = 1\n")

    class DummyDuration:
        @staticmethod
        def parse(value: str) -> str:
            return f"duration:{value}"

    class DummyMemory:
        @staticmethod
        def parse(value: str) -> str:
            return f"memory:{value}"

    class DummyJobResources:
        def __init__(self, **kwargs):  # noqa: ANN003
            self.kwargs = kwargs

    class DummyJobScheduling:
        def __init__(self, **kwargs):  # noqa: ANN003
            self.kwargs = kwargs

    class DummyJobExecution:
        def __init__(self, **kwargs):  # noqa: ANN003
            self.kwargs = kwargs

    class DummyCluster:
        def __init__(self, *, name: str, scheduler: str, **_kwargs) -> None:
            self.name = name
            self.scheduler = scheduler

    class DummySubmitor:
        def __init__(self, target, **_kwargs) -> None:
            self.cluster_name = target.name
            self.scheduler = target.scheduler

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def submit_job(self, **_kwargs):
            return SimpleNamespace(job_id="molq-job-123", scheduler_job_id="sched-456")

    class DummyScript:
        @staticmethod
        def inline(text: str) -> str:
            return text

    fake_molq = ModuleType("molq")
    fake_molq.Cluster = DummyCluster
    fake_molq.Duration = DummyDuration
    fake_molq.JobExecution = DummyJobExecution
    fake_molq.JobResources = DummyJobResources
    fake_molq.JobScheduling = DummyJobScheduling
    fake_molq.Memory = DummyMemory
    fake_molq.Script = DummyScript
    fake_molq.Submitor = DummySubmitor
    monkeypatch.setitem(sys.modules, "molq", fake_molq)

    handler = make_submit_handler(
        scheduler="slurm",
        cluster="cluster-a",
        resources={"cpus": 4, "mem": "8G", "time": "1h"},
        scheduling={"queue": "cpu"},
    )
    handler(
        script,
        run,
        SimpleNamespace(),
        SimpleNamespace(name="demo"),
    )

    assert run.metadata.executor_info == {
        "backend": "molq",
        "scheduler": "slurm",
        "cluster_name": "cluster-a",
        "job_id": "molq-job-123",
        "scheduler_job_id": "sched-456",
    }
