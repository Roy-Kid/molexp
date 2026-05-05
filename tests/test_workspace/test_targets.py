"""Tests for the compute-target registry and the molq Transport bridge."""

from __future__ import annotations

from pathlib import Path

import pytest
from molq.transport import LocalTransport, SshTransport

from molexp.workspace import (
    ComputeTarget,
    Workspace,
    add_target,
    get_target,
    has_target,
    list_targets,
    remove_target,
    to_transport,
)

# ---------------------------------------------------------------------------
# ComputeTarget validation
# ---------------------------------------------------------------------------


class TestComputeTargetValidation:
    def test_local_target_minimal(self) -> None:
        t = ComputeTarget(name="laptop", scratch_root="/tmp/molexp")
        assert t.name == "laptop"
        assert t.host is None
        assert t.scheduler == "local"
        assert t.is_remote is False

    def test_remote_target(self) -> None:
        t = ComputeTarget(
            name="hpc",
            host="me@host",
            scheduler="slurm",
            scratch_root="/scratch",
        )
        assert t.is_remote is True
        assert t.scheduler == "slurm"

    def test_scratch_root_is_required(self) -> None:
        with pytest.raises(ValueError, match="scratch_root"):
            ComputeTarget(name="x", scratch_root="")

    def test_ssh_options_without_host_rejected(self) -> None:
        with pytest.raises(ValueError, match="require host"):
            ComputeTarget(name="x", scratch_root="/tmp", port=22)

    def test_unknown_scheduler_rejected(self) -> None:
        with pytest.raises(ValueError):
            ComputeTarget(name="x", scratch_root="/tmp", scheduler="invalid")  # type: ignore[arg-type]

    def test_target_is_frozen(self) -> None:
        t = ComputeTarget(name="x", scratch_root="/tmp")
        with pytest.raises(Exception):
            t.name = "y"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Registry CRUD round-trip
# ---------------------------------------------------------------------------


class TestRegistry:
    def test_empty_registry(self, tmp_path: Path) -> None:
        ws = Workspace(tmp_path)
        ws.materialize()
        assert list_targets(ws) == []
        assert not has_target(ws, "anything")

    def test_add_and_list(self, tmp_path: Path) -> None:
        ws = Workspace(tmp_path)
        ws.materialize()
        add_target(ws, ComputeTarget(name="a", scratch_root="/tmp"))
        add_target(ws, ComputeTarget(name="b", host="me@h", scheduler="slurm", scratch_root="/s"))
        names = [t.name for t in list_targets(ws)]
        assert names == ["a", "b"]

    def test_round_trip_via_disk(self, tmp_path: Path) -> None:
        """Targets must survive workspace.json reload."""
        ws = Workspace(tmp_path)
        ws.materialize()
        add_target(
            ws,
            ComputeTarget(
                name="hpc",
                host="me@cluster",
                port=2222,
                identity_file="/k",
                ssh_opts=["-o", "ServerAliveInterval=30"],
                scheduler="slurm",
                scratch_root="/scratch/me",
                default_resources={"cpus": 8, "mem": "16G"},
            ),
        )

        ws2 = Workspace(tmp_path)  # fresh load
        t = get_target(ws2, "hpc")
        assert t.host == "me@cluster"
        assert t.port == 2222
        assert t.identity_file == "/k"
        assert t.ssh_opts == ["-o", "ServerAliveInterval=30"]
        assert t.scheduler == "slurm"
        assert t.default_resources == {"cpus": 8, "mem": "16G"}

    def test_add_duplicate_name_rejected(self, tmp_path: Path) -> None:
        ws = Workspace(tmp_path)
        ws.materialize()
        add_target(ws, ComputeTarget(name="a", scratch_root="/tmp"))
        with pytest.raises(ValueError, match="already exists"):
            add_target(ws, ComputeTarget(name="a", scratch_root="/other"))

    def test_remove(self, tmp_path: Path) -> None:
        ws = Workspace(tmp_path)
        ws.materialize()
        add_target(ws, ComputeTarget(name="a", scratch_root="/tmp"))
        add_target(ws, ComputeTarget(name="b", scratch_root="/tmp"))
        remove_target(ws, "a")
        assert [t.name for t in list_targets(ws)] == ["b"]

    def test_remove_missing_raises(self, tmp_path: Path) -> None:
        ws = Workspace(tmp_path)
        ws.materialize()
        with pytest.raises(KeyError):
            remove_target(ws, "ghost")

    def test_get_missing_raises(self, tmp_path: Path) -> None:
        ws = Workspace(tmp_path)
        ws.materialize()
        with pytest.raises(KeyError):
            get_target(ws, "ghost")


# ---------------------------------------------------------------------------
# Transport bridge
# ---------------------------------------------------------------------------


class TestToTransport:
    def test_local_target_yields_local_transport(self) -> None:
        t = ComputeTarget(name="x", scratch_root="/tmp")
        assert isinstance(to_transport(t), LocalTransport)

    def test_remote_target_yields_ssh_transport(self) -> None:
        t = ComputeTarget(
            name="x",
            host="me@h",
            port=2222,
            identity_file="/k",
            ssh_opts=["-o", "X=Y"],
            scheduler="slurm",
            scratch_root="/s",
        )
        tr = to_transport(t)
        assert isinstance(tr, SshTransport)
        assert tr.options.host == "me@h"
        assert tr.options.port == 2222
        assert tr.options.identity_file == "/k"
        assert tr.options.ssh_opts == ("-o", "X=Y")
