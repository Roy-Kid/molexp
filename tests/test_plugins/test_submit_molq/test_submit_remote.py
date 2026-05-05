"""SubmitHandler must route through the target's transport when one is set.

Uses a fake transport that records every shell call + file op, plus monkey-
patches molq's :class:`Submitor` so we don't need a working SSH endpoint or
to actually launch a worker subprocess.  The assertions cover the wiring —
not the worker behaviour, which is exercised by the molq suite.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from molq.transport import CommandResult

from molexp.workspace import ComputeTarget, Workspace


@dataclass
class RecordingTransport:
    """A transport that no-ops every method but records the calls."""

    calls: list[tuple[str, tuple, dict]] = field(default_factory=list)

    def _record(self, method: str, args: tuple, kwargs: dict) -> None:
        self.calls.append((method, args, kwargs))

    def run(self, argv, *, cwd=None, env=None, input=None, timeout=None):
        self._record("run", (tuple(argv),), {"cwd": cwd, "env": env, "input": input})
        return CommandResult(argv=tuple(argv), returncode=0, stdout="", stderr="")

    def read_text(self, path: str) -> str:
        return ""

    def read_bytes(self, path: str) -> bytes:
        return b""

    def write_text(self, path: str, data: str, *, mode: int = 0o600) -> None:
        self._record("write_text", (path,), {"mode": mode})

    def write_bytes(self, path: str, data: bytes, *, mode: int = 0o600) -> None:
        self._record("write_bytes", (path,), {"mode": mode})

    def exists(self, path: str) -> bool:
        return True

    def mkdir(self, path: str, *, parents: bool = True, exist_ok: bool = True) -> None:
        self._record("mkdir", (path,), {})

    def chmod(self, path: str, mode: int) -> None:
        return None

    def remove(self, path: str, *, recursive: bool = False) -> None:
        return None

    def upload(self, local: str, remote: str, *, recursive: bool = False, exclude=()) -> None:
        self._record("upload", (local, remote), {"recursive": recursive})

    def download(self, remote: str, local: str, *, recursive: bool = False, exclude=()) -> None:
        self._record("download", (remote, local), {"recursive": recursive})


def _make_run(tmp_path: Path):
    ws = Workspace(tmp_path)
    ws.materialize()
    project = ws.project("p")
    experiment = project.experiment("e", params={})
    run = experiment.run(parameters={"seed": 1})
    return ws, project, experiment, run


def test_submit_handler_with_target_stages_in_and_uses_transport(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    ws, project, experiment, run = _make_run(tmp_path)
    target = ComputeTarget(
        name="hpc",
        host="me@cluster",
        scheduler="slurm",
        scratch_root="/scratch/me/molexp",
    )

    transport = RecordingTransport()
    monkeypatch.setattr(
        "molexp.plugins.submit_molq.submit.to_transport"
        if False  # the import is local inside __call__; patch the module instead
        else "molexp.workspace.targets.to_transport",
        lambda _t: transport,
    )

    # Patch Submitor so we don't actually try to talk to a real cluster.
    captured_submitor: dict[str, Any] = {}

    class FakeJob:
        job_id = "fake-job-id"
        scheduler_job_id = "fake-sched-id"

    class FakeSubmitor:
        def __init__(self, target, *, jobs_dir):
            captured_submitor["cluster_name"] = target.name
            captured_submitor["scheduler"] = target.scheduler
            captured_submitor["jobs_dir"] = jobs_dir
            captured_submitor["transport"] = target.transport
            self._event_bus = type("EB", (), {"on": lambda *_a, **_kw: None})()

        def __enter__(self): return self
        def __exit__(self, *a): return False

        def submit_job(self, *, argv, resources, scheduling, execution, metadata):
            captured_submitor["submit_argv"] = argv
            captured_submitor["submit_cwd"] = execution.cwd
            captured_submitor["submit_metadata"] = metadata
            return FakeJob()

    import molexp.plugins.submit_molq.submit as submit_mod
    monkeypatch.setattr(submit_mod, "Submitor", FakeSubmitor, raising=False)
    # The Submitor name is imported lazily inside __call__; patch the import
    # site to ensure our fake gets used.
    monkeypatch.setattr("molq.Submitor", FakeSubmitor)

    from molexp.plugins.submit_molq.submit import make_submit_handler

    handler = make_submit_handler(
        scheduler="ignored-when-target-set",
        cluster=None,
        resources={},
        scheduling={},
        target=target,
    )
    handler(None, run, experiment, project)

    # Stage-in happened (one upload of the run dir to the remote scratch).
    uploads = [c for c in transport.calls if c[0] == "upload"]
    assert len(uploads) == 1
    src, dst = uploads[0][1]
    assert src == str(Path(run.run_dir).resolve())
    assert dst.startswith("/scratch/me/molexp/")

    # Submitor was constructed with the target's scheduler + transport.
    assert captured_submitor["scheduler"] == "slurm"
    assert captured_submitor["transport"] is transport
    # The worker is told to chdir into the remote exec dir.
    assert captured_submitor["submit_cwd"].startswith("/scratch/me/molexp/")
    assert "executions/" in captured_submitor["submit_cwd"]
    # Argv points at the remote run dir, not the local one.
    argv = captured_submitor["submit_argv"]
    assert "molexp.cli" in argv and "execute" in argv
    assert any(a.startswith("/scratch/me/molexp/") for a in argv)


def test_submit_handler_without_target_uses_local_transport(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """No-target path: transport=None on the handler → Cluster falls back
    to ``LocalTransport`` and no staging happens."""
    _, project, experiment, run = _make_run(tmp_path)

    captured: dict[str, Any] = {}

    class FakeJob:
        job_id = "x"
        scheduler_job_id = "y"

    class FakeSubmitor:
        def __init__(self, target, *, jobs_dir):
            captured["transport"] = target.transport
            captured["jobs_dir"] = jobs_dir
            self._event_bus = type("EB", (), {"on": lambda *_a, **_kw: None})()
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def submit_job(self, *, argv, resources, scheduling, execution, metadata):
            captured["cwd"] = execution.cwd
            return FakeJob()

    monkeypatch.setattr("molq.Submitor", FakeSubmitor)

    from molexp.plugins.submit_molq.submit import make_submit_handler

    handler = make_submit_handler(
        scheduler="local",
        cluster=None,
        resources={},
        scheduling={},
        target=None,
    )
    handler(None, run, experiment, project)

    # No target → SubmitHandler does not synthesise an SshTransport; the
    # Cluster gets transport=None and falls back to the default LocalTransport.
    from molq.transport import LocalTransport

    assert isinstance(captured["transport"], LocalTransport)
    # jobs_dir lives under the LOCAL run dir, not on a remote scratch path.
    assert captured["jobs_dir"].startswith(str(run.run_dir))
