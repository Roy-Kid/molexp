"""Behavior locks for ``molexp.workspace.target`` — the Target address family.

One ``ComputeTarget``-rooted family: ``parse_target`` / ``resolve_target``
produce ``LocalTarget`` / ``RemoteTarget`` address views that ARE
``ComputeTarget`` subclasses (the targets-merge invariant), plus the
transport + filesystem bridges and the SSH session cache.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from molq.transport import LocalTransport, SshTransport

from molexp.workspace import (
    ComputeTarget,
    SSHSession,
    Workspace,
    add_target,
    resolve_compute_target,
)
from molexp.workspace.fs_local import LocalFileSystem
from molexp.workspace.target import (
    LocalTarget,
    RemoteTarget,
    SessionManager,
    TargetNeedsResolution,
    TargetNotFound,
    parse_target,
    resolve_target,
    target_to_filesystem,
)


@pytest.fixture
def ws(tmp_path: Path) -> Workspace:
    w = Workspace(tmp_path / "lab")
    w.materialize()
    return w


class TestParseTarget:
    def test_falsy_resolves_to_cwd_local(self) -> None:
        for raw in (None, ""):
            target = parse_target(raw)
            assert isinstance(target, LocalTarget)
            assert target.path == Path.cwd()

    def test_local_path_is_a_compute_target_address_view(self, tmp_path: Path) -> None:
        target = parse_target(str(tmp_path))
        assert isinstance(target, ComputeTarget)  # unified family (targets-merge)
        assert isinstance(target, LocalTarget)
        assert target.host is None
        assert target.is_remote is False
        assert target.scratch_root == str(tmp_path.resolve())
        assert target.scheduler == "local"

    def test_scp_remote_is_a_compute_target_with_all_fields(self) -> None:
        target = parse_target("me@host.example:/data/ws")
        assert isinstance(target, ComputeTarget)  # unified family (targets-merge)
        assert isinstance(target, RemoteTarget)
        assert target.is_remote is True
        assert target.user == "me"
        assert target.host == "host.example"
        assert target.path == "/data/ws"
        assert target.scratch_root == "/data/ws"
        assert target.scp_notation == "me@host.example:/data/ws"
        assert str(target) == "me@host.example:/data/ws"

    def test_shell_expanded_home_restored_to_tilde(self) -> None:
        local_home = os.path.expanduser("~")  # noqa: PTH111
        target = parse_target(f"host.example:{local_home}/runs")
        assert isinstance(target, RemoteTarget)
        assert target.path == "~/runs"

    def test_at_name_raises_needs_resolution(self) -> None:
        with pytest.raises(TargetNeedsResolution):
            parse_target("@cluster")


class TestResolveTarget:
    def test_local_spec_pairs_local_transport(self, tmp_path: Path) -> None:
        target, transport = resolve_target(str(tmp_path))
        assert isinstance(target, LocalTarget)
        assert isinstance(transport, LocalTransport)

    def test_remote_spec_pairs_ssh_transport(self) -> None:
        target, transport = resolve_target("me@host.example:/data/ws")
        assert isinstance(target, RemoteTarget)
        assert isinstance(transport, SshTransport)

    def test_at_name_without_workspace_raises_needs_resolution(self) -> None:
        with pytest.raises(TargetNeedsResolution):
            resolve_target("@cluster", None)

    def test_at_name_unknown_raises_not_found(self, ws: Workspace) -> None:
        with pytest.raises(TargetNotFound):
            resolve_target("@does-not-exist", ws)

    def test_at_name_resolves_local_registry_target(self, ws: Workspace, tmp_path: Path) -> None:
        scratch = tmp_path / "scratch"
        add_target(ws, ComputeTarget(name="box", scratch_root=str(scratch)))
        target, transport = resolve_target("@box", ws)
        assert isinstance(target, LocalTarget)
        assert target.path == Path(str(scratch))
        assert isinstance(transport, LocalTransport)

    def test_at_name_resolves_remote_registry_target_with_all_fields(self, ws: Workspace) -> None:
        add_target(
            ws,
            ComputeTarget(
                name="hpc",
                host="me@cluster.example",
                scheduler="slurm",
                scratch_root="/scratch/me",
                default_resources={"gpus": 1},
            ),
        )
        target, transport = resolve_target("@hpc", ws)
        assert isinstance(target, RemoteTarget)
        assert target.user == "me"
        assert target.host == "cluster.example"
        assert target.path == "/scratch/me"
        # The address view carries the registry record's fields (unified family).
        assert target.name == "hpc"
        assert target.scheduler == "slurm"
        assert target.default_resources == {"gpus": 1}
        assert isinstance(transport, SshTransport)

    def test_at_local_falls_back_to_builtin_local(self, ws: Workspace) -> None:
        """CLI ``@local`` resolves like the server: built-in local target."""
        target, transport = resolve_target("@local", ws)
        assert isinstance(target, LocalTarget)
        assert target.name == "local"
        assert target.path == Path(str(ws.root))
        assert isinstance(transport, LocalTransport)


class TestResolveComputeTarget:
    def test_named_lookup_local_fallback_and_missing_raises(self, ws: Workspace) -> None:
        """The single named-target resolution path: named lookup, the built-in
        ``local`` fallback, and a raw ``KeyError`` on an unknown name."""
        add_target(ws, ComputeTarget(name="laptop", scratch_root="/tmp/molexp"))
        assert resolve_compute_target(ws, "laptop").scratch_root == "/tmp/molexp"
        assert resolve_compute_target(ws, "local").scratch_root == str(ws.root)
        with pytest.raises(KeyError):
            resolve_compute_target(ws, "ghost")


class TestTargetToFilesystem:
    def test_local_target_yields_local_filesystem(self, tmp_path: Path) -> None:
        assert isinstance(target_to_filesystem(parse_target(str(tmp_path))), LocalFileSystem)

    def test_remote_target_yields_remote_filesystem(self) -> None:
        from molexp.workspace.fs_remote import RemoteFileSystem

        fs = target_to_filesystem(parse_target("me@host.example:/data"))
        assert isinstance(fs, RemoteFileSystem)


@pytest.fixture(autouse=True)
def _clean_sessions():
    SessionManager.close_all()
    yield
    SessionManager.close_all()


class TestSessionManager:
    def _remote(self) -> RemoteTarget:
        return RemoteTarget(
            user="me",
            host="host.example",
            port=2222,
            path="/data/ws",
            identity_file=None,
        )

    def test_get_or_create_caches_by_scp_notation(self) -> None:
        target = self._remote()
        first = SessionManager.get_or_create(target)
        second = SessionManager.get_or_create(target)
        assert first is second
        assert first.name == "me@host.example:/data/ws"
        assert SessionManager.get(target) is first
        assert SessionManager.get_by_name("me@host.example:/data/ws") is first

    def test_close_removes_session(self) -> None:
        target = self._remote()
        SessionManager.get_or_create(target)
        assert SessionManager.close("me@host.example:/data/ws") is True
        assert SessionManager.get(target) is None
        assert SessionManager.close("me@host.example:/data/ws") is False

    def test_session_type_is_sshsession_not_bare_session(self) -> None:
        """workspace ``Session`` was renamed ``SSHSession`` — the bare name now
        belongs exclusively to the agent layer's LLM conversation session."""
        session = SessionManager.get_or_create(self._remote())
        assert isinstance(session, SSHSession)
        assert not hasattr(__import__("molexp.workspace", fromlist=["x"]), "Session")
