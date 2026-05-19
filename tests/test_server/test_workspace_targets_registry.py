"""Tests for the server-process WorkspaceTarget registry.

The registry holds the descriptors that point the active workspace at a
remote root.  It is server-process scope (lives at ``~/.molexp/
workspace_targets.json`` in production), not workspace scope — you need
a descriptor before any workspace is open.

Backing store is :func:`molexp.workspace.base.atomic_write_json` so the
file is crash-safe.  These tests inject a ``store_path`` under
``tmp_path`` to avoid touching the developer's real home directory.
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

# ── WorkspaceTarget value-type ─────────────────────────────────────────


@pytest.mark.unit
def test_workspace_target_is_frozen():
    """WorkspaceTarget must be frozen so we never mutate a registered descriptor."""
    target = WorkspaceTarget(
        name="hpc1",
        host="me@hpc.example.org",
        root_path="/scratch/me/molexp",
    )
    with pytest.raises(ValidationError):
        target.name = "renamed"  # type: ignore[misc]


@pytest.mark.unit
@pytest.mark.parametrize(
    "bad_name",
    [
        "",
        "   ",
        "has space",
        "trailing-slash/",
        "/abs/path",
        "name\twith\ttab",
        "with\nnewline",
    ],
)
def test_workspace_target_rejects_invalid_names(bad_name: str):
    with pytest.raises(ValidationError):
        WorkspaceTarget(name=bad_name, host="h", root_path="/r")


@pytest.mark.unit
@pytest.mark.parametrize(
    "ok_name",
    [
        "a",
        "hpc1",
        "prod-cluster",
        "lab_node_2",
        "x.y.z",
    ],
)
def test_workspace_target_accepts_slug_names(ok_name: str):
    target = WorkspaceTarget(name=ok_name, host="h", root_path="/r")
    assert target.name == ok_name


@pytest.mark.unit
def test_workspace_target_ssh_opts_is_tuple():
    """ssh_opts must be a tuple so the model stays frozen-immutable."""
    target = WorkspaceTarget(
        name="hpc1",
        host="h",
        root_path="/r",
        ssh_opts=("-o", "StrictHostKeyChecking=no"),
    )
    assert target.ssh_opts == ("-o", "StrictHostKeyChecking=no")
    # Mutating the original list does not leak in
    src = ["-o", "X"]
    target2 = WorkspaceTarget(name="h2", host="h", root_path="/r", ssh_opts=src)  # type: ignore[arg-type]
    src.append("-o")
    src.append("Y")
    assert target2.ssh_opts == ("-o", "X")


# ── WorkspaceTargetRegistry CRUD ───────────────────────────────────────


@pytest.fixture
def registry_path(tmp_path: Path) -> Path:
    return tmp_path / "workspace_targets.json"


@pytest.fixture
def registry(registry_path: Path) -> WorkspaceTargetRegistry:
    return WorkspaceTargetRegistry(store_path=registry_path)


def _make_target(name: str = "hpc1", host: str = "me@hpc.example.org") -> WorkspaceTarget:
    return WorkspaceTarget(name=name, host=host, root_path=f"/scratch/{name}")


@pytest.mark.unit
def test_registry_empty_on_first_construction(registry: WorkspaceTargetRegistry):
    assert registry.list() == []


@pytest.mark.unit
def test_registry_add_then_get(registry: WorkspaceTargetRegistry):
    t = _make_target()
    registry.add(t)
    assert registry.get("hpc1") == t


@pytest.mark.unit
def test_registry_get_returns_frozen_instance(registry: WorkspaceTargetRegistry):
    registry.add(_make_target())
    fetched = registry.get("hpc1")
    with pytest.raises(ValidationError):
        fetched.name = "renamed"  # type: ignore[misc]


@pytest.mark.unit
def test_registry_list_preserves_insertion_order(registry: WorkspaceTargetRegistry):
    registry.add(_make_target("first"))
    registry.add(_make_target("second"))
    registry.add(_make_target("third"))
    assert [t.name for t in registry.list()] == ["first", "second", "third"]


@pytest.mark.unit
def test_registry_list_returns_snapshot_not_internal_state(
    registry: WorkspaceTargetRegistry,
):
    registry.add(_make_target("hpc1"))
    snapshot = registry.list()
    snapshot.append(_make_target("hpc2"))  # mutate caller's copy
    assert [t.name for t in registry.list()] == ["hpc1"]  # registry untouched


@pytest.mark.unit
def test_registry_add_duplicate_raises_value_error(registry: WorkspaceTargetRegistry):
    registry.add(_make_target("dup"))
    with pytest.raises(ValueError, match="dup"):
        registry.add(_make_target("dup", host="other.host"))


@pytest.mark.unit
def test_registry_get_unknown_raises_key_error(registry: WorkspaceTargetRegistry):
    with pytest.raises(KeyError):
        registry.get("missing")


@pytest.mark.unit
def test_registry_remove_unknown_raises_key_error(registry: WorkspaceTargetRegistry):
    with pytest.raises(KeyError):
        registry.remove("missing")


@pytest.mark.unit
def test_registry_remove(registry: WorkspaceTargetRegistry):
    registry.add(_make_target("a"))
    registry.add(_make_target("b"))
    registry.remove("a")
    assert [t.name for t in registry.list()] == ["b"]


# ── Persistence + atomicity ────────────────────────────────────────────


@pytest.mark.unit
def test_registry_persists_across_construction(registry_path: Path):
    """Restarting the process (= constructing a fresh registry) must keep state."""
    r1 = WorkspaceTargetRegistry(store_path=registry_path)
    r1.add(_make_target("hpc1"))
    r1.add(_make_target("hpc2"))

    r2 = WorkspaceTargetRegistry(store_path=registry_path)
    assert [t.name for t in r2.list()] == ["hpc1", "hpc2"]


@pytest.mark.unit
def test_registry_first_read_of_missing_file_yields_empty(tmp_path: Path):
    store = tmp_path / "nested" / "deep" / "workspace_targets.json"
    r = WorkspaceTargetRegistry(store_path=store)
    assert r.list() == []
    assert not store.exists()


@pytest.mark.unit
def test_registry_atomic_write_rolls_back_on_failure(
    registry_path: Path,
):
    """If atomic_write_json blows up, neither disk nor in-memory state changes."""
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
def test_registry_corrupt_json_raises_typed_error(registry_path: Path):
    """A corrupt store file surfaces a clear error rather than silent truncation."""
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    registry_path.write_text("{not json")

    r = WorkspaceTargetRegistry(store_path=registry_path)
    with pytest.raises(ValueError, match=r"workspace_targets\.json"):
        r.list()


@pytest.mark.unit
def test_registry_writes_through_atomic_write_json(registry_path: Path, monkeypatch):
    """Verify the registry calls workspace.base.atomic_write_json, not raw open."""
    from molexp.workspace import base as ws_base

    calls: list[Path] = []
    real = ws_base.atomic_write_json

    def spy(path, data):
        calls.append(path)
        return real(path, data)

    monkeypatch.setattr(
        "molexp.server.workspace_targets.atomic_write_json",
        spy,
    )

    r = WorkspaceTargetRegistry(store_path=registry_path)
    r.add(_make_target())
    assert len(calls) == 1
    assert calls[0] == registry_path


# ── Layer-boundary invariants ─────────────────────────────────────────


@pytest.mark.unit
def test_module_does_not_pull_agent_or_workflow(monkeypatch):
    """Importing molexp.server.workspace_targets must not eagerly load
    molexp.agent or molexp.workflow."""
    forbidden = ["molexp.agent", "molexp.workflow"]
    # Snapshot what's already loaded — we only police what *this import* adds.
    already = {m for m in sys.modules if any(m.startswith(p) for p in forbidden)}

    # Force a fresh import.
    for key in list(sys.modules):
        if key == "molexp.server.workspace_targets":
            del sys.modules[key]

    import molexp.server.workspace_targets  # noqa: F401

    new = {m for m in sys.modules if any(m.startswith(p) for p in forbidden)} - already
    assert not new, f"workspace_targets pulled in forbidden modules: {sorted(new)}"


@pytest.mark.unit
def test_no_arbitrary_types_allowed_in_module():
    """Project rule: arbitrary_types_allowed is banned in agent/server schema modules."""
    src = Path(__file__).resolve().parents[2] / "src" / "molexp" / "server" / "workspace_targets.py"
    assert "arbitrary_types_allowed" not in src.read_text()


@pytest.mark.unit
def test_no_os_environ_writes_in_module():
    """Project rule: no env-var plumbing for runtime config."""
    src = Path(__file__).resolve().parents[2] / "src" / "molexp" / "server" / "workspace_targets.py"
    text = src.read_text()
    # Allow reads (os.environ.get / [] in module-level constants are fine if we ever need them),
    # but disallow writes.
    assert "os.environ[" not in text or "os.environ['" not in text
    assert "os.environ.setdefault" not in text


# ── Sanity: registry survives a JSON round-trip with all fields ───────


@pytest.mark.unit
def test_registry_round_trip_preserves_all_fields(registry_path: Path):
    rich = WorkspaceTarget(
        name="rich",
        host="me@host.example",
        port=2222,
        identity_file="/home/me/.ssh/id_ed25519",
        ssh_opts=("-o", "StrictHostKeyChecking=accept-new"),
        root_path="/scratch/me/molexp",
        cache_dir="/var/cache/molexp/rich",
        cache_ttl_seconds=600,
    )
    r1 = WorkspaceTargetRegistry(store_path=registry_path)
    r1.add(rich)

    r2 = WorkspaceTargetRegistry(store_path=registry_path)
    got = r2.get("rich")
    assert got == rich
    # Sanity-check on-disk shape: ssh_opts is a list in JSON, tuple in memory.
    raw = json.loads(registry_path.read_text())
    assert raw["targets"][0]["ssh_opts"] == ["-o", "StrictHostKeyChecking=accept-new"]
    assert raw["targets"][0]["cache_dir"] == "/var/cache/molexp/rich"
    assert raw["targets"][0]["cache_ttl_seconds"] == 600


# ── v1 → v2 forward compatibility ─────────────────────────────────────


@pytest.mark.unit
def test_registry_loads_v1_envelope_without_cache_fields(registry_path: Path):
    """A pre-cache-bump store file must still deserialize, defaulting the new fields."""
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


@pytest.mark.unit
def test_registry_rejects_negative_cache_ttl():
    from pydantic import ValidationError as _VE

    with pytest.raises(_VE):
        WorkspaceTarget(
            name="bad", host="h", root_path="/r", cache_ttl_seconds=-1
        )
