"""Behavior locks for the built-in ``local`` compute-target defaults.

``builtin_local_target`` / ``effective_targets`` live in
``molexp.workspace.targets``. These two behaviors are their **sole owner** —
``test_workspace.test_targets`` covers the registry CRUD helpers and
``to_transport`` but not the built-in-local default path exercised here.
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


class TestTargetDefaults:
    def test_builtin_local_target_shape(self, ws: Workspace) -> None:
        target = builtin_local_target(ws)
        assert target.name == LOCAL_TARGET_NAME == "local"
        assert target.scratch_root == str(ws.root)
        assert target.scheduler == "local"
        assert target.is_remote is False

    def test_effective_targets_registered_local_overrides_builtin(self, ws: Workspace) -> None:
        add_target(ws, ComputeTarget(name=LOCAL_TARGET_NAME, scratch_root="/custom/scratch"))
        targets = effective_targets(ws)
        assert [t.name for t in targets] == [LOCAL_TARGET_NAME]
        assert targets[0].scratch_root == "/custom/scratch"
