"""CLI and plugin tests for molq-backed execution."""

from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace

from typer.testing import CliRunner

from molexp.cli import app
from molexp.plugins.submit_molq.submit import make_submit_handler
from molexp.workspace import Workspace

runner = CliRunner()


def _make_workspace(tmp_path):
    workspace = Workspace(root=tmp_path / "workspace", name="Test")
    project = workspace.add_project("demo")
    experiment = project.add_experiment("train")
    run = experiment.add_run(params={"seed": 0})
    return workspace, experiment, run


def test_run_scheduler_uses_generic_molq_backend(monkeypatch, tmp_path):
    script = tmp_path / "train.py"
    workspace_root = tmp_path / "workspace"
    script.write_text(
        "\n".join(
            [
                "import molexp as me",
                "from molexp.workflow import default_binding_registry, promote_callable",
                "",
                f"ws = me.Workspace({str(workspace_root)!r})",
                "project = ws.add_project('demo')",
                "exp = project.add_experiment('train')",
                "",
                "def train(ctx: me.RunContext) -> None:",
                "    ctx.set_result('ok', True)",
                "",
                "default_binding_registry.bind(exp, promote_callable(train, name='train'))",
                "me.entry(ws)",
                "",
            ]
        )
    )

    captured: dict[str, str] = {}

    def fake_make_submit_handler(*, scheduler, cluster, resources, scheduling, target=None):
        captured["scheduler"] = scheduler
        captured["target"] = target

        class DummyHandler:
            def __init__(self) -> None:
                self.submitted_runs = []

            def __call__(self, _script, mol_run, _exp_spec, _project_spec):
                mol_run._update_metadata(executor_info={"backend": "molq", "scheduler": scheduler})
                self.submitted_runs.append(mol_run)

        return DummyHandler()

    import molexp.plugins.submit_molq.metadata as metadata_mod
    import molexp.plugins.submit_molq.submit as submit_mod

    monkeypatch.setattr(metadata_mod, "supported_schedulers", lambda: ("local", "slurm"))
    monkeypatch.setattr(submit_mod, "make_submit_handler", fake_make_submit_handler)

    result = runner.invoke(
        app, ["run", str(script), "--scheduler", "local", "-t", str(workspace_root)]
    )

    assert result.exit_code == 0, result.output
    assert captured["scheduler"] == "local"


def test_run_cancel_uses_molq_handle(monkeypatch, tmp_path):
    workspace, experiment, run = _make_workspace(tmp_path)
    run._update_metadata(
        status="pending",
        executor_info={
            "backend": "molq",
            "scheduler": "slurm",
            "cluster_name": "default",
            "job_id": "molq-job-123",
            "scheduler_job_id": "sched-456",
        },
    )

    calls: list[tuple[str, str, str]] = []

    class FakeCluster:
        def __init__(self, *, name: str, scheduler: str, **_kwargs) -> None:
            self.name = name
            self.scheduler = scheduler

    class FakeSubmitor:
        def __init__(self, target, **_kwargs) -> None:
            self.cluster_name = target.name
            self.scheduler = target.scheduler

        def cancel_job(self, job_id: str) -> None:
            calls.append((self.cluster_name, self.scheduler, job_id))

        def close(self) -> None:
            return None

    fake_molq = ModuleType("molq")
    fake_molq.Cluster = FakeCluster
    fake_molq.Submitor = FakeSubmitor
    monkeypatch.setitem(sys.modules, "molq", fake_molq)

    result = runner.invoke(
        app,
        ["runs", "cancel", run.id, "--yes", "-t", str(workspace.root)],
    )

    assert result.exit_code == 0, result.output
    assert calls == [("default", "slurm", "molq-job-123")]

    reloaded = Workspace.load(workspace.root)
    reloaded_run = reloaded.get_project("demo").get_experiment(experiment.id).get_run(run.id)
    assert reloaded_run is not None
    assert reloaded_run.status == "cancelled"


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
