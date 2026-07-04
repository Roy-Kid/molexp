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


def test_effective_targets_registered_local_overrides_builtin(ws: Workspace) -> None:
    add_target(ws, ComputeTarget(name=LOCAL_TARGET_NAME, scratch_root="/custom/scratch"))
    targets = effective_targets(ws)
    assert [t.name for t in targets] == [LOCAL_TARGET_NAME]
    assert targets[0].scratch_root == "/custom/scratch"
