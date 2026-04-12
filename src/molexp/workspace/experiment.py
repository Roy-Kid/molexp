"""Experiment entity with run management.

An Experiment binds a workflow definition to a project and holds
configuration for parameter sweeps. Runs are individual executions.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .project import Project
    from .workspace import Workspace

from .asset import AssetLibrary
from .base import _list_children, _load_metadata, _reconstruct, _save_metadata
from .models import ExperimentMetadata, RunMetadata, WorkflowSnapshotRef
from .run import Run
from .utils import generate_id


class Experiment:
    """Repeatable experiment bound to a workflow.

    Example::

        exp = Experiment(
            name="lr-sweep",
            project=project,
            workflow_source="train.py",
            parameter_space={"lr": [1e-4, 1e-3]},
        )
        exp.materialize()
        run = exp.create_run(parameters={"lr": 1e-4})
    """

    def __init__(
        self,
        name: str,
        project: Project,
        id: str | None = None,
        workflow_source: str | None = None,
        workflow_type: str | None = None,
        parameter_space: dict[str, Any] | None = None,
        git_commit: str | None = None,
    ) -> None:
        self.project = project
        self.metadata = ExperimentMetadata(
            id=id if id is not None else generate_id(),
            name=name,
            workflow_source=workflow_source,
            workflow_type=workflow_type,
            parameter_space=parameter_space or {},
            git_commit=git_commit,
        )
        self._assets_lib: AssetLibrary | None = None

    # ── Properties ──────────────────────────────────────────────────────

    @property
    def id(self) -> str:
        return self.metadata.id

    @property
    def name(self) -> str:
        return self.metadata.name

    @property
    def created_at(self):
        return self.metadata.created_at

    @property
    def description(self) -> str:
        return self.metadata.description

    @property
    def tags(self) -> list[str]:
        return self.metadata.tags

    @property
    def workflow_source(self) -> str | None:
        return self.metadata.workflow_source

    @property
    def parameter_space(self) -> dict[str, Any]:
        return self.metadata.parameter_space

    @property
    def workspace(self) -> Workspace:
        return self.project.workspace

    @property
    def experiment_dir(self) -> Path:
        return self.project.project_dir / "experiments" / self.id

    @property
    def assets(self) -> AssetLibrary:
        if self._assets_lib is None:
            self._assets_lib = AssetLibrary(self.experiment_dir / "assets")
        return self._assets_lib

    # ── Persistence ─────────────────────────────────────────────────────

    def materialize(self) -> None:
        """Create filesystem structure and persist metadata."""
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        _save_metadata(self.metadata, self.experiment_dir / "experiment.json")

    def save(self) -> None:
        """Persist current metadata to disk."""
        _save_metadata(self.metadata, self.experiment_dir / "experiment.json")

    # ── Run operations ──────────────────────────────────────────────────

    def create_run(
        self,
        parameters: dict[str, Any] | None = None,
        *,
        id: str | None = None,
        exist_ok: bool = False,
    ) -> Run:
        """Create a run (materialized immediately).

        A workflow snapshot is automatically captured from the experiment's
        metadata at run-creation time.

        Raises:
            ValueError: If run with this ID already exists and *exist_ok* is False.
        """
        # Capture workflow snapshot from experiment
        snapshot = None
        if self.metadata.workflow_source:
            snapshot = WorkflowSnapshotRef(
                source=self.metadata.workflow_source,
                git_commit=self.metadata.git_commit,
            )

        run = Run(
            experiment=self,
            parameters=parameters,
            id=id,
            workflow_snapshot=snapshot,
        )
        run_dir = self.experiment_dir / "runs" / run.id
        if run_dir.exists():
            if exist_ok:
                return self._load_run_from_dir(run_dir)
            raise ValueError(f"Run '{run.id}' already exists")
        run.materialize()
        return run

    def get_run(self, run_id: str) -> Run | None:
        """Get run by ID."""
        run_dir = self.experiment_dir / "runs" / f"run-{run_id}"
        if not run_dir.exists():
            return None
        return self._load_run_from_dir(run_dir)

    def list_runs(self) -> list[Run]:
        """List all runs by scanning the ``runs/`` directory."""
        return _list_children(
            children_dir=self.experiment_dir / "runs",
            metadata_filename="run.json",
            metadata_cls=RunMetadata,
            child_cls=Run,
            attrs_factory=lambda m: {"experiment": self, "metadata": m},
        )

    # ── Internal ────────────────────────────────────────────────────────

    def _load_run_from_dir(self, run_dir: Path) -> Run:
        meta = _load_metadata(RunMetadata, run_dir / "run.json")
        return _reconstruct(Run, {"experiment": self, "metadata": meta})
