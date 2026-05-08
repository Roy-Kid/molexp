"""Tests for stage_in / stage_out using a fake transport that records calls."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from molq.transport import CommandResult, TransportError

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
    project = ws.project("p")
    experiment = project.experiment("e", params={})
    run = experiment.run(parameters={"seed": 1})
    return ws, run


# ---------------------------------------------------------------------------
# stage_in
# ---------------------------------------------------------------------------


def test_stage_in_local_target_is_noop(tmp_path: Path) -> None:
    """Local target with scratch_root that resolves to run_dir → no rsync."""
    _ws, run = _make_run(tmp_path)
    # scratch_root chosen so target_run_dir == run.run_dir
    target = ComputeTarget(
        name="loop",
        scratch_root=str(tmp_path),
    )
    # The default target_run_dir is scratch_root/<wsid>/<pid>/<eid>/<rid>;
    # the local optimisation only short-circuits when that path equals run.run_dir.
    # Make a target whose computed dir matches by carefully setting scratch_root.
    # Compose the local Workspace dir where it actually lives:
    # ws.root/projects/<pid>/experiments/<eid>/runs/<rid>
    # Since we can't easily make those paths match, just assert the no-op
    # path is taken when transport's run_dir == src.
    transport = FakeTransport()
    # Force the no-op by using a fake target whose target_run_dir equals run_dir.
    from molexp.plugins.submit_molq import staging as staging_mod

    monkey_target_run_dir = lambda *_a, **_kw: str(run.run_dir)  # noqa: E731
    original = staging_mod.target_run_dir
    staging_mod.target_run_dir = monkey_target_run_dir  # type: ignore[assignment]
    try:
        stage_in(transport, run, target)
    finally:
        staging_mod.target_run_dir = original  # type: ignore[assignment]
    assert transport.uploads == []
    assert transport.mkdirs == []


def test_stage_in_remote_uploads_run_dir(tmp_path: Path) -> None:
    _ws, run = _make_run(tmp_path)
    target = ComputeTarget(
        name="hpc",
        host="me@h",
        scheduler="slurm",
        scratch_root="/scratch",
    )
    transport = FakeTransport()
    stage_in(transport, run, target)

    assert len(transport.uploads) == 1
    src, dst, recursive = transport.uploads[0]
    assert src == str(Path(run.run_dir).resolve())
    assert dst.startswith("/scratch/")
    assert dst.endswith(f"/{run.id}")
    assert recursive is True
    # Ensures the target dir was created before upload.
    assert dst in transport.mkdirs


# ---------------------------------------------------------------------------
# stage_out
# ---------------------------------------------------------------------------


def test_stage_out_local_is_noop(tmp_path: Path) -> None:
    _ws, run = _make_run(tmp_path)
    target = ComputeTarget(name="loop", scratch_root="/tmp")

    from molexp.plugins.submit_molq import staging as staging_mod

    transport = FakeTransport()
    original = staging_mod.target_run_dir
    staging_mod.target_run_dir = lambda *_a, **_kw: str(run.run_dir)  # type: ignore[assignment]
    try:
        stage_out(transport, run, target, "exec-1")
    finally:
        staging_mod.target_run_dir = original  # type: ignore[assignment]
    assert transport.downloads == []


def test_stage_out_remote_pulls_exec_dir_and_run_json(tmp_path: Path) -> None:
    _ws, run = _make_run(tmp_path)
    target = ComputeTarget(
        name="hpc",
        host="me@h",
        scheduler="slurm",
        scratch_root="/scratch",
    )
    transport = FakeTransport()
    stage_out(transport, run, target, "exec-abc")

    # Always pulls executions/<id> and run.json
    remote_paths = [d[0] for d in transport.downloads]
    assert any(p.endswith("/executions/exec-abc") for p in remote_paths)
    assert any(p.endswith("/run.json") for p in remote_paths)


def test_stage_out_skips_missing_optional_dirs(tmp_path: Path) -> None:
    """When artifacts/ etc. don't exist remotely, stage_out shouldn't error."""
    _ws, run = _make_run(tmp_path)
    target = ComputeTarget(
        name="hpc",
        host="me@h",
        scheduler="slurm",
        scratch_root="/scratch",
    )
    transport = FakeTransport()
    # `existing` is empty so transport.exists() returns False for everything.
    stage_out(transport, run, target, "exec-x")
    # executions/<id> is always pulled; artifacts/.ckpt/assets.json are skipped.
    remote_paths = [d[0] for d in transport.downloads]
    assert not any(p.endswith("/artifacts") for p in remote_paths)
    assert not any(p.endswith("/.ckpt") for p in remote_paths)
    assert not any(p.endswith("/assets.json") for p in remote_paths)


def test_stage_out_is_idempotent(tmp_path: Path) -> None:
    """Calling stage_out twice should not double-register anything observable."""
    _ws, run = _make_run(tmp_path)
    target = ComputeTarget(
        name="hpc",
        host="me@h",
        scheduler="slurm",
        scratch_root="/scratch",
    )
    transport = FakeTransport()
    stage_out(transport, run, target, "exec-x")
    n_first = len(transport.downloads)
    stage_out(transport, run, target, "exec-x")
    n_second = len(transport.downloads)
    # rsync handles dedup at the file level — what matters is that we don't
    # raise and that the call count grows in a bounded way (one round per call).
    assert n_second == n_first * 2


def test_stage_out_swallows_transport_errors_for_optional_paths(tmp_path: Path) -> None:
    """If the remote exec dir isn't accessible (early failure), no exception."""
    _ws, run = _make_run(tmp_path)
    target = ComputeTarget(
        name="hpc",
        host="me@h",
        scheduler="slurm",
        scratch_root="/scratch",
    )
    transport = FakeTransport(
        raise_on_download={
            f"/scratch/{run.experiment.project.workspace.metadata.id}/"
            f"{run.experiment.project.id}/{run.experiment.id}/{run.id}/"
            f"executions/exec-x"
        }
    )
    stage_out(transport, run, target, "exec-x")  # must not raise
