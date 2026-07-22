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
    list_targets,
    remove_target,
    to_transport,
)


class TestComputeTargetValidation:
    """The ``_validate_axes`` model_validator on ``ComputeTarget``."""

    def test_scratch_root_is_required(self) -> None:
        with pytest.raises(ValueError, match="scratch_root"):
            ComputeTarget(name="x", scratch_root="")

    def test_ssh_options_without_host_rejected(self) -> None:
        with pytest.raises(ValueError, match="require host"):
            ComputeTarget(name="x", scratch_root="/tmp", port=22)


class TestRegistry:
    """CRUD over ``WorkspaceMetadata.targets`` via the ``targets`` helpers."""

    def test_add_get_round_trips_through_workspace_json(self, tmp_path: Path) -> None:
        """A registered target survives a fresh ``Workspace`` load unchanged."""
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

    def test_remove_drops_named_target(self, tmp_path: Path) -> None:
        ws = Workspace(tmp_path)
        ws.materialize()
        add_target(ws, ComputeTarget(name="a", scratch_root="/tmp"))
        add_target(ws, ComputeTarget(name="b", scratch_root="/tmp"))
        remove_target(ws, "a")
        assert [t.name for t in list_targets(ws)] == ["b"]

    def test_get_missing_raises(self, tmp_path: Path) -> None:
        ws = Workspace(tmp_path)
        ws.materialize()
        with pytest.raises(KeyError):
            get_target(ws, "ghost")


class TestToTransport:
    """``to_transport`` maps a ``ComputeTarget``'s host axis onto a molq Transport."""

    def test_local_target_yields_local_transport(self) -> None:
        t = ComputeTarget(name="x", scratch_root="/tmp")
        assert isinstance(to_transport(t), LocalTransport)

    def test_remote_target_yields_ssh_transport_with_options(self) -> None:
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
