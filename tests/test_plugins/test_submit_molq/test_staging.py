"""Tests for ``stage_in`` / ``stage_out`` using a fake transport.

The fake records every upload/download/mkdir so the tests assert on the
local<->remote transfer contract without a real SSH endpoint.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest
from molq.transport import CommandResult, TransportError

from molexp.plugins.submit_molq import staging as staging_mod
from molexp.plugins.submit_molq.staging import stage_in, stage_out
from molexp.workspace import ComputeTarget, Workspace


@dataclass
class FakeTransport:
    """Records every call so tests can assert on uploads / downloads."""

    uploads: list[tuple[str, str, bool]] = field(default_factory=list)
    downloads: list[tuple[str, str, bool]] = field(default_factory=list)
    mkdirs: list[str] = field(default_factory=list)
    existing: set[str] = field(default_factory=set)
    raise_on_download: set[str] = field(default_factory=set)

    def run(self, *_a: Any, **_kw: Any) -> CommandResult:
        return CommandResult(argv=(), returncode=0, stdout="", stderr="")

    def read_text(self, path: str) -> str:
        return ""

    def read_bytes(self, path: str) -> bytes:
        return b""

    def write_text(self, path: str, data: str, *, mode: int = 0o600) -> None:
        return None

    def write_bytes(self, path: str, data: bytes, *, mode: int = 0o600) -> None:
        return None

    def exists(self, path: str) -> bool:
        return path in self.existing

    def mkdir(self, path: str, *, parents: bool = True, exist_ok: bool = True) -> None:
        self.mkdirs.append(path)

    def chmod(self, path: str, mode: int) -> None:
        return None

    def remove(self, path: str, *, recursive: bool = False) -> None:
        return None

    def upload(self, local: str, remote: str, *, recursive: bool = False, exclude=()) -> None:
        self.uploads.append((local, remote, recursive))

    def download(self, remote: str, local: str, *, recursive: bool = False, exclude=()) -> None:
        if remote in self.raise_on_download:
            raise TransportError("simulated", remote=remote)
        self.downloads.append((remote, local, recursive))


def _make_run(tmp_path: Path):
    """Create a workspace + project + experiment + run hierarchy on disk."""
    ws = Workspace(tmp_path)
    ws.materialize()
    project = ws.add_project("p")
    experiment = project.add_experiment("e", params={})
    run = experiment.add_run(params={"seed": 1})
    return ws, run


def _remote_target() -> ComputeTarget:
    return ComputeTarget(name="hpc", host="me@h", scheduler="slurm", scratch_root="/scratch")


class TestStageIn:
    def test_noop_when_target_dir_equals_run_dir(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Local target whose computed run dir == run.run_dir → no rsync."""
        _ws, run = _make_run(tmp_path)
        target = ComputeTarget(name="loop", scratch_root=str(tmp_path))
        monkeypatch.setattr(staging_mod, "target_run_dir", lambda *_a, **_kw: str(run.run_dir))

        transport = FakeTransport()
        stage_in(transport, run, target)

        assert transport.uploads == []
        assert transport.mkdirs == []

    def test_remote_uploads_run_dir_after_mkdir(self, tmp_path: Path) -> None:
        _ws, run = _make_run(tmp_path)
        transport = FakeTransport()

        stage_in(transport, run, _remote_target())

        assert len(transport.uploads) == 1
        src, dst, recursive = transport.uploads[0]
        assert src == str(Path(run.run_dir).resolve())
        assert dst.startswith("/scratch/")
        assert dst.endswith(f"/{run.id}")
        assert recursive is True
        # Ensures the target dir was created before upload.
        assert dst in transport.mkdirs


class TestStageOut:
    def test_noop_when_remote_dir_equals_run_dir(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        _ws, run = _make_run(tmp_path)
        target = ComputeTarget(name="loop", scratch_root="/tmp")
        monkeypatch.setattr(staging_mod, "target_run_dir", lambda *_a, **_kw: str(run.run_dir))

        transport = FakeTransport()
        stage_out(transport, run, target, "exec-1")

        assert transport.downloads == []

    def test_remote_pulls_exec_and_run_json_and_skips_absent_optionals(
        self, tmp_path: Path
    ) -> None:
        _ws, run = _make_run(tmp_path)
        # `existing` is empty → transport.exists() is False for the optional dirs.
        transport = FakeTransport()

        stage_out(transport, run, _remote_target(), "exec-abc")

        remote_paths = [d[0] for d in transport.downloads]
        # Always pulls executions/<id> and run.json.
        assert any(p.endswith("/executions/exec-abc") for p in remote_paths)
        assert any(p.endswith("/run.json") for p in remote_paths)
        # Absent optional dirs are skipped, not pulled.
        assert not any(p.endswith("/artifacts") for p in remote_paths)
        assert not any(p.endswith("/.ckpt") for p in remote_paths)
        assert not any(p.endswith("/assets.json") for p in remote_paths)

    def test_swallows_transport_error_on_download(self, tmp_path: Path) -> None:
        """A TransportError on the exec-dir pull is swallowed, never raised."""
        _ws, run = _make_run(tmp_path)
        ws = run.experiment.project.workspace
        transport = FakeTransport(
            raise_on_download={
                f"/scratch/{ws.metadata.id}/{run.experiment.project.id}/"
                f"{run.experiment.id}/{run.id}/executions/exec-x"
            }
        )
        stage_out(transport, run, _remote_target(), "exec-x")  # must not raise
