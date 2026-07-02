"""Behavior locks for the built-in ``local`` compute target defaults.

Written BEFORE the targets-merge refactor to pin the server-facing surface
of the built-in local target. After the merge the logic lives in
``molexp.workspace.targets`` (the ``server/target_defaults`` shim was
removed once ``routes/run.py`` switched over) — every assertion here held
both before and after the move.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from molexp.workspace import (
    LOCAL_TARGET_NAME,
    ComputeTarget,
    Workspace,
    add_target,
    builtin_local_target,
    effective_targets,
)
from molexp.workspace import (
    resolve_compute_target as resolve_target,
)


@pytest.fixture
def ws(tmp_path: Path) -> Workspace:
    w = Workspace(tmp_path / "lab")
    w.materialize()
    return w


def test_builtin_local_target_shape(ws: Workspace) -> None:
    target = builtin_local_target(ws)
    assert target.name == LOCAL_TARGET_NAME == "local"
    assert target.scratch_root == str(ws.root)
    assert target.scheduler == "local"
    assert target.is_remote is False


def test_effective_targets_empty_registry_prepends_builtin(ws: Workspace) -> None:
    targets = effective_targets(ws)
    assert [t.name for t in targets] == [LOCAL_TARGET_NAME]


def test_effective_targets_builtin_first_then_registered(ws: Workspace) -> None:
    add_target(ws, ComputeTarget(name="laptop", scratch_root="/tmp/molexp"))
    names = [t.name for t in effective_targets(ws)]
    assert names == [LOCAL_TARGET_NAME, "laptop"]


def test_effective_targets_registered_local_overrides_builtin(ws: Workspace) -> None:
    add_target(ws, ComputeTarget(name=LOCAL_TARGET_NAME, scratch_root="/custom/scratch"))
    targets = effective_targets(ws)
    assert [t.name for t in targets] == [LOCAL_TARGET_NAME]
    assert targets[0].scratch_root == "/custom/scratch"


def test_resolve_target_registered(ws: Workspace) -> None:
    add_target(ws, ComputeTarget(name="laptop", scratch_root="/tmp/molexp"))
    assert resolve_target(ws, "laptop").scratch_root == "/tmp/molexp"


def test_resolve_target_local_falls_back_to_builtin(ws: Workspace) -> None:
    target = resolve_target(ws, LOCAL_TARGET_NAME)
    assert target.name == LOCAL_TARGET_NAME
    assert target.scratch_root == str(ws.root)


def test_resolve_target_unknown_raises_keyerror(ws: Workspace) -> None:
    with pytest.raises(KeyError):
        resolve_target(ws, "ghost")
