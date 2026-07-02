"""Behavior locks for ``molexp.workspace.target`` — the Target address family.

Written BEFORE the targets-merge refactor (two parallel target systems →
one ``ComputeTarget``-rooted family) to pin the observable behavior of the
address-parsing side: ``parse_target`` / ``resolve_target`` / the transport +
filesystem bridges / the SSH session cache.  Every assertion here must hold
both before and after the merge.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from molq.transport import LocalTransport, SshTransport

from molexp.workspace import ComputeTarget, Workspace, add_target
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
    target_to_transport,
)


@pytest.fixture
def ws(tmp_path: Path) -> Workspace:
    w = Workspace(tmp_path / "lab")
    w.materialize()
    return w


# ---------------------------------------------------------------------------
# parse_target
# ---------------------------------------------------------------------------


class TestParseTarget:
    def test_none_and_empty_resolve_to_cwd(self) -> None:
        for raw in (None, ""):
            target = parse_target(raw)
            assert isinstance(target, LocalTarget)
            assert target.path == Path.cwd()

    def test_local_path(self, tmp_path: Path) -> None:
        target = parse_target(str(tmp_path))
        assert isinstance(target, LocalTarget)
        assert target.path == tmp_path.resolve()
        assert str(target) == str(tmp_path.resolve())

    def test_scp_notation_with_user(self) -> None:
        target = parse_target("me@host.example:/data/ws")
        assert isinstance(target, RemoteTarget)
        assert target.user == "me"
        assert target.host == "host.example"
        assert target.path == "/data/ws"
        assert target.scp_notation == "me@host.example:/data/ws"
        assert str(target) == "me@host.example:/data/ws"

    def test_scp_notation_without_user(self) -> None:
        target = parse_target("host.example:/data/ws")
        assert isinstance(target, RemoteTarget)
        assert target.user is None
        assert target.host == "host.example"
        assert target.path == "/data/ws"

    def test_shell_expanded_home_restored_to_tilde(self) -> None:
        local_home = os.path.expanduser("~")  # noqa: PTH111
        target = parse_target(f"host.example:{local_home}/runs")
        assert isinstance(target, RemoteTarget)
        assert target.path == "~/runs"

    def test_at_name_raises_needs_resolution(self) -> None:
        with pytest.raises(TargetNeedsResolution):
            parse_target("@cluster")


# ---------------------------------------------------------------------------
# resolve_target
# ---------------------------------------------------------------------------


class TestResolveTarget:
    def test_local_spec(self, tmp_path: Path) -> None:
        target, transport = resolve_target(str(tmp_path))
        assert isinstance(target, LocalTarget)
        assert isinstance(transport, LocalTransport)

    def test_remote_spec(self) -> None:
        target, transport = resolve_target("me@host.example:/data/ws")
        assert isinstance(target, RemoteTarget)
        assert isinstance(transport, SshTransport)

    def test_at_name_without_workspace_raises(self) -> None:
        with pytest.raises(TargetNeedsResolution):
            resolve_target("@cluster", None)

    def test_at_name_unknown_raises_not_found(self, ws: Workspace) -> None:
        with pytest.raises(TargetNotFound):
            resolve_target("@does-not-exist", ws)

    def test_at_name_local_registry_target(self, ws: Workspace, tmp_path: Path) -> None:
        scratch = tmp_path / "scratch"
        add_target(ws, ComputeTarget(name="box", scratch_root=str(scratch)))
        target, transport = resolve_target("@box", ws)
        assert isinstance(target, LocalTarget)
        assert target.path == Path(str(scratch))
        assert isinstance(transport, LocalTransport)

    def test_at_name_remote_registry_target(self, ws: Workspace) -> None:
        add_target(
            ws,
            ComputeTarget(
                name="hpc",
                host="me@cluster.example",
                scheduler="slurm",
                scratch_root="/scratch/me",
            ),
        )
        target, transport = resolve_target("@hpc", ws)
        assert isinstance(target, RemoteTarget)
        assert target.user == "me"
        assert target.host == "cluster.example"
        assert target.path == "/scratch/me"
        assert isinstance(transport, SshTransport)


# ---------------------------------------------------------------------------
# Transport / filesystem bridges
# ---------------------------------------------------------------------------


class TestBridges:
    def test_local_target_to_transport(self, tmp_path: Path) -> None:
        assert isinstance(target_to_transport(parse_target(str(tmp_path))), LocalTransport)

    def test_remote_target_to_transport(self) -> None:
        transport = target_to_transport(parse_target("me@host.example:/data"))
        assert isinstance(transport, SshTransport)

    def test_local_target_to_filesystem(self, tmp_path: Path) -> None:
        assert isinstance(target_to_filesystem(parse_target(str(tmp_path))), LocalFileSystem)

    def test_remote_target_to_filesystem(self) -> None:
        from molexp.workspace.fs_remote import RemoteFileSystem

        fs = target_to_filesystem(parse_target("me@host.example:/data"))
        assert isinstance(fs, RemoteFileSystem)


# ---------------------------------------------------------------------------
# Unified family (targets-merge): address views ARE ComputeTargets
# ---------------------------------------------------------------------------


class TestUnifiedFamily:
    def test_target_alias_is_compute_target(self) -> None:
        from molexp.workspace.target import Target

        assert Target is ComputeTarget

    def test_local_view_is_a_compute_target(self, tmp_path: Path) -> None:
        target = parse_target(str(tmp_path))
        assert isinstance(target, ComputeTarget)
        assert target.host is None
        assert target.is_remote is False
        assert target.scratch_root == str(tmp_path.resolve())
        assert target.scheduler == "local"

    def test_remote_view_is_a_compute_target(self) -> None:
        target = parse_target("me@host.example:/data/ws")
        assert isinstance(target, ComputeTarget)
        assert target.is_remote is True
        assert target.scratch_root == "/data/ws"

    def test_at_name_view_carries_registry_fields(self, ws: Workspace) -> None:
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
        target, _transport = resolve_target("@hpc", ws)
        assert target.name == "hpc"
        assert target.scheduler == "slurm"
        assert target.default_resources == {"gpus": 1}

    def test_at_local_falls_back_to_builtin(self, ws: Workspace) -> None:
        """CLI ``@local`` resolves like the server: built-in local target."""
        target, transport = resolve_target("@local", ws)
        assert isinstance(target, LocalTarget)
        assert target.name == "local"
        assert target.path == Path(str(ws.root))
        assert isinstance(transport, LocalTransport)

    def test_remote_view_model_dump_round_trip(self) -> None:
        target = parse_target("me@host.example:/data/ws")
        assert isinstance(target, RemoteTarget)
        again = RemoteTarget(**target.model_dump())
        assert again.model_dump() == target.model_dump()

    def test_resolve_compute_target_is_the_workspace_named_resolver(self, ws: Workspace) -> None:
        from molexp.workspace import resolve_compute_target

        add_target(ws, ComputeTarget(name="laptop", scratch_root="/tmp/molexp"))
        assert resolve_compute_target(ws, "laptop").scratch_root == "/tmp/molexp"
        assert resolve_compute_target(ws, "local").scratch_root == str(ws.root)
        with pytest.raises(KeyError):
            resolve_compute_target(ws, "ghost")


# ---------------------------------------------------------------------------
# Session cache (SSH connection reuse)
# ---------------------------------------------------------------------------


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

    def test_close_all_and_list(self) -> None:
        SessionManager.get_or_create(self._remote())
        SessionManager.get_or_create(
            RemoteTarget(user=None, host="other.example", port=None, path="/x")
        )
        assert len(SessionManager.list_sessions()) == 2
        assert SessionManager.close_all() == 2
        assert SessionManager.list_sessions() == []

    def test_sessions_are_ssh_sessions(self) -> None:
        """workspace ``Session`` was renamed ``SSHSession`` — the bare name now
        belongs exclusively to the agent layer's LLM conversation session."""
        from molexp.workspace import SSHSession

        session = SessionManager.get_or_create(self._remote())
        assert isinstance(session, SSHSession)
        assert not hasattr(__import__("molexp.workspace", fromlist=["x"]), "Session")
