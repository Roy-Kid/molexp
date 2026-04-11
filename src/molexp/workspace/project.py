"""Project entity with experiment management.

Construction is side-effect free; call ``materialize()`` to write to disk.
Children are discovered by scanning the filesystem.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .workspace import Workspace

from .asset import AssetLibrary
from .base import _list_children, _load_metadata, _reconstruct, _save_metadata
from .experiment import Experiment
from .models import ExperimentMetadata, ProjectMetadata
from .utils import slugify


class Project:
    """Research project container.

    Example::

        project = Project(name="QM9 Energy Prediction", workspace=workspace)
        project.materialize()
        exp = project.create_experiment(name="baseline", workflow_source="train.py")
    """

    def __init__(self, name: str, workspace: Workspace) -> None:
        self.workspace = workspace
        self.metadata = ProjectMetadata(id=slugify(name), name=name)
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
    def owner(self) -> str:
        return self.metadata.owner

    @property
    def tags(self) -> list[str]:
        return self.metadata.tags

    @property
    def config(self) -> dict[str, Any]:
        return self.metadata.config

    @property
    def project_dir(self) -> Path:
        return self.workspace.root / "projects" / self.id

    @property
    def assets(self) -> AssetLibrary:
        if self._assets_lib is None:
            self._assets_lib = AssetLibrary(self.project_dir / "assets")
        return self._assets_lib

    # ── Persistence ─────────────────────────────────────────────────────

    def materialize(self) -> None:
        """Create filesystem structure and persist metadata."""
        self.project_dir.mkdir(parents=True, exist_ok=True)
        _save_metadata(self.metadata, self.project_dir / "project.json")

    def save(self) -> None:
        """Persist current metadata to disk."""
        _save_metadata(self.metadata, self.project_dir / "project.json")

    def import_asset(
        self,
        name: str,
        src: str | Path,
        action: str = "copy",
        meta: dict[str, Any] | None = None,
    ):
        """Import an asset into the project library."""
        return self.assets.import_asset(name, src, action, meta)

    # ── Experiment operations ───────────────────────────────────────────

    def create_experiment(
        self,
        name: str,
        *,
        id: str | None = None,
        workflow_source: str | None = None,
        workflow_type: str | None = None,
        parameter_space: dict[str, Any] | None = None,
        git_commit: str | None = None,
        exist_ok: bool = False,
    ) -> Experiment:
        """Create an experiment (materialized immediately).

        Args:
            name: Human-readable experiment name.
            id: Optional custom ID (UUID generated if omitted).
            workflow_source: Path to the workflow definition file.
            workflow_type: ``"python"`` | ``"yaml"`` | …
            parameter_space: Parameter search space definition.
            git_commit: Git commit hash for reproducibility.
            exist_ok: Return existing experiment instead of raising.

        Raises:
            ValueError: If experiment already exists and *exist_ok* is False.
        """
        experiment = Experiment(
            name=name,
            project=self,
            id=id,
            workflow_source=workflow_source,
            workflow_type=workflow_type,
            parameter_space=parameter_space,
            git_commit=git_commit,
        )
        exp_dir = self.project_dir / "experiments" / experiment.id
        if exp_dir.exists():
            if exist_ok:
                return self._load_experiment_from_dir(exp_dir)
            raise ValueError(f"Experiment '{experiment.id}' already exists")
        experiment.materialize()
        return experiment

    def get_experiment(self, experiment_id: str) -> Experiment | None:
        """Get experiment by ID."""
        exp_dir = self.project_dir / "experiments" / experiment_id
        if not exp_dir.exists():
            return None
        return self._load_experiment_from_dir(exp_dir)

    def list_experiments(self) -> list[Experiment]:
        """List all experiments by scanning the ``experiments/`` directory."""
        return _list_children(
            children_dir=self.project_dir / "experiments",
            metadata_filename="experiment.json",
            metadata_cls=ExperimentMetadata,
            child_cls=Experiment,
            attrs_factory=lambda m: {
                "project": self,
                "metadata": m,
                "_assets_lib": None,
            },
        )

    # ── Internal ────────────────────────────────────────────────────────

    def _load_experiment_from_dir(self, exp_dir: Path) -> Experiment:
        meta = _load_metadata(ExperimentMetadata, exp_dir / "experiment.json")
        return _reconstruct(
            Experiment,
            {"project": self, "metadata": meta, "_assets_lib": None},
        )
