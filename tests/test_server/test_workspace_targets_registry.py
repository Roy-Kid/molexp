"""Tests for the server-process WorkspaceTarget registry
(``molexp.server.workspace_targets``).

The registry holds the descriptors that point the active workspace at a remote
root. It is server-process scope (lives at ``~/.molexp/workspace_targets.json``
in production), not workspace scope — a descriptor exists before any workspace
is open. The backing store is crash-safe via
:func:`molexp.workspace.base.atomic_write_json`; these tests inject a
``store_path`` under ``tmp_path`` to avoid the developer's real home directory.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import patch

import pytest
from pydantic import ValidationError

from molexp.server.workspace_targets import (
    WorkspaceTarget,
    WorkspaceTargetRegistry,
)


@pytest.fixture
def registry_path(tmp_path: Path) -> Path:
    return tmp_path / "workspace_targets.json"


@pytest.fixture
def registry(registry_path: Path) -> WorkspaceTargetRegistry:
    return WorkspaceTargetRegistry(store_path=registry_path)


def _make_target(name: str = "hpc1", host: str = "me@hpc.example.org") -> WorkspaceTarget:
    return WorkspaceTarget(name=name, host=host, root_path=f"/scratch/{name}")


class TestWorkspaceTargetValidation:
    """The custom ``name`` slug validator (molexp-owned, not pydantic mechanics)."""

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "bad_name",
        ["", "has space", "/abs/path"],
        ids=["empty", "internal-space", "leading-slash"],
    )
    def test_rejects_names_that_are_not_slugs(self, bad_name: str):
        with pytest.raises(ValidationError):
            WorkspaceTarget(name=bad_name, host="h", root_path="/r")


class TestWorkspaceTargetRegistry:
    @pytest.mark.unit
    def test_add_then_get(self, registry: WorkspaceTargetRegistry):
        t = _make_target()
        registry.add(t)
        assert registry.get("hpc1") == t

    @pytest.mark.unit
    def test_list_returns_defensive_snapshot(self, registry: WorkspaceTargetRegistry):
        """``list()`` hands back a fresh copy — mutating it can't corrupt the registry."""
        registry.add(_make_target("hpc1"))
        snapshot = registry.list()
        snapshot.append(_make_target("hpc2"))
        assert [t.name for t in registry.list()] == ["hpc1"]

    @pytest.mark.unit
    def test_add_duplicate_name_raises_value_error(self, registry: WorkspaceTargetRegistry):
        registry.add(_make_target("dup"))
        with pytest.raises(ValueError, match="dup"):
            registry.add(_make_target("dup", host="other.host"))

    @pytest.mark.unit
    def test_get_unknown_name_raises_key_error(self, registry: WorkspaceTargetRegistry):
        with pytest.raises(KeyError):
            registry.get("missing")

    @pytest.mark.unit
    def test_remove(self, registry: WorkspaceTargetRegistry):
        registry.add(_make_target("a"))
        registry.add(_make_target("b"))
        registry.remove("a")
        assert [t.name for t in registry.list()] == ["b"]

    @pytest.mark.unit
    def test_persists_all_fields_across_construction(self, registry_path: Path):
        """A fresh registry (= process restart) reloads every target in insertion
        order with all fields intact; ``ssh_opts`` persists as a JSON list."""
        rich = WorkspaceTarget(
            name="hpc1",
            host="me@host.example",
            port=2222,
            identity_file="/home/me/.ssh/id_ed25519",
            ssh_opts=("-o", "StrictHostKeyChecking=accept-new"),
            root_path="/scratch/me/molexp",
            cache_dir="/var/cache/molexp/hpc1",
            cache_ttl_seconds=600,
        )
        r1 = WorkspaceTargetRegistry(store_path=registry_path)
        r1.add(rich)
        r1.add(_make_target("hpc2"))

        r2 = WorkspaceTargetRegistry(store_path=registry_path)
        assert [t.name for t in r2.list()] == ["hpc1", "hpc2"]
        assert r2.get("hpc1") == rich

        raw = json.loads(registry_path.read_text())
        assert raw["targets"][0]["ssh_opts"] == ["-o", "StrictHostKeyChecking=accept-new"]

    @pytest.mark.unit
    def test_missing_store_file_reads_empty_without_creating_it(self, tmp_path: Path):
        store = tmp_path / "nested" / "deep" / "workspace_targets.json"
        r = WorkspaceTargetRegistry(store_path=store)
        assert r.list() == []
        assert not store.exists()

    @pytest.mark.unit
    def test_atomic_write_failure_rolls_back_disk_and_memory(self, registry_path: Path):
        """If ``atomic_write_json`` blows up, neither disk nor in-memory state changes
        (the cache is mutated only after the disk write succeeds)."""
        r = WorkspaceTargetRegistry(store_path=registry_path)
        r.add(_make_target("existing"))
        before_disk = registry_path.read_text()

        with patch("molexp.server.workspace_targets.atomic_write_json") as bad_write:
            bad_write.side_effect = OSError("simulated disk full")
            with pytest.raises(OSError, match="simulated disk full"):
                r.add(_make_target("doomed"))

        assert registry_path.read_text() == before_disk
        assert [t.name for t in r.list()] == ["existing"]

    @pytest.mark.unit
    def test_corrupt_store_file_raises_typed_error(self, registry_path: Path):
        """A corrupt store file surfaces a clear error rather than silent truncation."""
        registry_path.parent.mkdir(parents=True, exist_ok=True)
        registry_path.write_text("{not json")

        r = WorkspaceTargetRegistry(store_path=registry_path)
        with pytest.raises(ValueError, match=r"workspace_targets\.json"):
            r.list()

    @pytest.mark.unit
    def test_loads_v1_envelope_defaulting_new_cache_fields(self, registry_path: Path):
        """A pre-cache-bump (v1) store file still deserializes, defaulting the new fields."""
        registry_path.parent.mkdir(parents=True, exist_ok=True)
        registry_path.write_text(
            json.dumps(
                {
                    "version": 1,
                    "targets": [
                        {
                            "name": "legacy",
                            "host": "me@old.host",
                            "root_path": "/legacy/root",
                            "port": None,
                            "identity_file": None,
                            "ssh_opts": [],
                        }
                    ],
                }
            )
        )
        r = WorkspaceTargetRegistry(store_path=registry_path)
        target = r.get("legacy")
        assert target.cache_dir is None
        assert target.cache_ttl_seconds == 300


class TestLayerBoundary:
    @pytest.mark.unit
    def test_import_does_not_pull_agent_or_workflow(self, monkeypatch):
        """Importing molexp.server.workspace_targets must not eagerly load
        molexp.agent or molexp.workflow (Layer-4 import-boundary guard)."""
        forbidden = ["molexp.agent", "molexp.workflow"]
        already = {m for m in sys.modules if any(m.startswith(p) for p in forbidden)}

        for key in list(sys.modules):
            if key == "molexp.server.workspace_targets":
                del sys.modules[key]

        import molexp.server.workspace_targets  # noqa: F401

        new = {m for m in sys.modules if any(m.startswith(p) for p in forbidden)} - already
        assert not new, f"workspace_targets pulled in forbidden modules: {sorted(new)}"
