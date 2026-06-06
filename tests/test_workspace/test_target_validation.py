"""Target-reference validation (workflow-workspace-hardening P2-4).

``RunMetadata.target`` / ``ExperimentMetadata.default_target`` name a compute
target that must exist in the workspace's ``WorkspaceMetadata.targets``
registry (models.py: "Validated against WorkspaceMetadata.targets at write
time"). When the registry is non-empty, ``add_run`` / ``add_experiment`` now
reject an unregistered target name. A registry-less workspace keeps accepting
free-form target strings (back-compat).
"""

from __future__ import annotations

import pytest

from molexp.workspace import Workspace
from molexp.workspace.models import ComputeTarget
from molexp.workspace.targets import add_target


@pytest.fixture
def ws(tmp_path):
    return Workspace(root=tmp_path / "lab", name="lab")


def test_unregistered_target_rejected_when_registry_nonempty(ws):
    add_target(ws, ComputeTarget(name="laptop", scratch_root="/tmp/molexp"))
    exp = ws.add_project("p").add_experiment("e")

    # Registered target is accepted.
    r = exp.add_run(target="laptop")
    assert r.metadata.target == "laptop"

    # Unregistered target is rejected.
    with pytest.raises(ValueError, match="compute target"):
        exp.add_run(target="cluster")


def test_default_target_validated_on_add_experiment(ws):
    add_target(ws, ComputeTarget(name="laptop", scratch_root="/tmp/molexp"))
    proj = ws.add_project("p")
    with pytest.raises(ValueError, match="compute target"):
        proj.add_experiment("bad", default_target="ghost")


def test_freeform_target_allowed_when_registry_empty(ws):
    # No targets registered → free-form target accepted (back-compat).
    exp = ws.add_project("p").add_experiment("e")
    r = exp.add_run(target="adhoc")
    assert r.metadata.target == "adhoc"
