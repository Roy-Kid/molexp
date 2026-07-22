"""``resolve_plan_compute_target`` — the ONE plan compute-target resolution path.

The single name→``ComputeTarget`` resolution shared by the CLI (``molexp plan``)
and the server's plan-tasks route (Python = UI law). Semantics under test:

* an explicit ``name`` wins over the run's ``metadata.target``;
* an unknown explicit ``name`` raises ``ValueError`` listing the known names
  (fail-fast, no silent fallback);
* with no explicit name the run's ``metadata.target`` is looked up in
  ``WorkspaceMetadata.targets`` — an unregistered/stale metadata target
  resolves to ``None`` (never raises);
* neither names a target → ``None`` (the documented local default).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from molexp.services.plan_runtime.targets import resolve_plan_compute_target
from molexp.workspace import Workspace
from molexp.workspace.models import ComputeTarget
from molexp.workspace.targets import add_target

if TYPE_CHECKING:
    from pathlib import Path

    from molexp.workspace.experiment import Experiment


@pytest.fixture
def workspace(tmp_path: Path) -> Workspace:
    """A workspace with two registered compute targets (``gpu-a`` / ``gpu-b``)."""
    ws = Workspace(root=tmp_path / "lab", name="targets-lab")
    ws.materialize()
    add_target(ws, ComputeTarget(name="gpu-a", scratch_root="/scratch/a"))
    add_target(ws, ComputeTarget(name="gpu-b", scratch_root="/scratch/b"))
    return ws


@pytest.fixture
def experiment(workspace: Workspace) -> Experiment:
    return workspace.add_project("proj").add_experiment("exp")


class TestResolvePlanComputeTarget:
    def test_explicit_name_wins_over_run_metadata_target(
        self, experiment: Experiment, workspace: Workspace
    ) -> None:
        run = experiment.add_run(params={"case": "explicit-wins"}, target="gpu-a")

        resolved = resolve_plan_compute_target(run, workspace, name="gpu-b")

        assert resolved is not None
        assert resolved.name == "gpu-b"

    def test_falls_back_to_run_metadata_target_when_no_explicit_name(
        self, experiment: Experiment, workspace: Workspace
    ) -> None:
        run = experiment.add_run(params={"case": "metadata-fallback"}, target="gpu-a")

        resolved = resolve_plan_compute_target(run, workspace)

        assert isinstance(resolved, ComputeTarget)
        assert resolved.name == "gpu-a"

    def test_unknown_explicit_name_raises_value_error_listing_known_names(
        self, experiment: Experiment, workspace: Workspace
    ) -> None:
        run = experiment.add_run(params={"case": "unknown-explicit"})

        with pytest.raises(ValueError) as excinfo:
            resolve_plan_compute_target(run, workspace, name="no-such-target")

        message = str(excinfo.value)
        assert "gpu-a" in message
        assert "gpu-b" in message

    def test_returns_none_when_neither_name_nor_metadata_target(
        self, experiment: Experiment, workspace: Workspace
    ) -> None:
        run = experiment.add_run(params={"case": "neither"})

        assert resolve_plan_compute_target(run, workspace) is None

    def test_unregistered_metadata_target_resolves_to_none_not_error(
        self, experiment: Experiment, workspace: Workspace
    ) -> None:
        """A stale ``metadata.target`` that no longer matches a registered target
        degrades to ``None`` (only an *explicit* unknown name raises)."""
        run = experiment.add_run(params={"case": "stale-metadata"})
        # ``add_run(target=...)`` validates registration, so a stale name can only
        # arise post-hoc — emulate it via the metadata setter (immutable copy).
        run.metadata = run.metadata.model_copy(update={"target": "ghost-target"})

        assert resolve_plan_compute_target(run, workspace) is None
