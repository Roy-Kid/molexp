"""Project entity with experiment management.

Construction is side-effect free; ``workspace.project(...)`` materializes
on disk at call-time (idempotent: existing projects are loaded, missing
ones are created).
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

        ws = Workspace("./lab")
        project = ws.project("QM9")
        exp = project.experiment("baseline", params={"lr": 1e-3})
    """

    def __init__(self, name: str, workspace: Workspace) -> None:
        self.workspace = workspace
        self.metadata = ProjectMetadata(id=slugify(name), name=name)
        self._assets_lib: AssetLibrary | None = None
        self._experiments_cache: dict[str, Experiment] = {}

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
        """Create filesystem structure and persist metadata (non-recursive)."""
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

    def experiment(
        self,
        name: str,
        *,
        id: str | None = None,
        params: dict[str, Any] | None = None,
        n_replicas: int = 1,
        seeds: list[int] | None = None,
        workflow_source: str | None = None,
        workflow_type: str | None = None,
        git_commit: str | None = None,
    ) -> Experiment:
        """Get-or-create an experiment (idempotent, materialized immediately).

        If an experiment with the same ID (or slug from *name*) exists on
        disk, it is loaded and returned.  Otherwise a new experiment is
        constructed and materialized.
        """
        exp_id = id if id is not None else slugify(name)
        if exp_id in self._experiments_cache:
            return self._experiments_cache[exp_id]
        exp_dir = self.project_dir / "experiments" / exp_id
        if exp_dir.exists():
            exp = self._load_experiment_from_dir(exp_dir)
        else:
            exp = Experiment(
                name=name,
                project=self,
                id=exp_id,
                params=params,
                n_replicas=n_replicas,
                seeds=seeds,
                workflow_source=workflow_source,
                workflow_type=workflow_type,
                git_commit=git_commit,
            )
            exp.materialize()
        self._experiments_cache[exp.id] = exp
        return exp

    def get_experiment(self, experiment_id: str) -> Experiment | None:
        """Get experiment by ID."""
        if experiment_id in self._experiments_cache:
            return self._experiments_cache[experiment_id]
        exp_dir = self.project_dir / "experiments" / experiment_id
        if not exp_dir.exists():
            return None
        exp = self._load_experiment_from_dir(exp_dir)
        self._experiments_cache[exp.id] = exp
        return exp

    def list_experiments(self) -> list[Experiment]:
        """List all experiments (disk scan merged with in-memory cache)."""
        seen: dict[str, Experiment] = dict(self._experiments_cache)
        scanned = _list_children(
            children_dir=self.project_dir / "experiments",
            metadata_filename="experiment.json",
            metadata_cls=ExperimentMetadata,
            child_cls=Experiment,
            attrs_factory=lambda m: {
                "project": self,
                "metadata": m,
                "_assets_lib": None,
                "_workflow": None,
            },
        )
        for e in scanned:
            seen.setdefault(e.id, e)
        return list(seen.values())

    # ── Internal ────────────────────────────────────────────────────────

    def _load_experiment_from_dir(self, exp_dir: Path) -> Experiment:
        meta = _load_metadata(ExperimentMetadata, exp_dir / "experiment.json")
        return _reconstruct(
            Experiment,
            {
                "project": self,
                "metadata": meta,
                "_assets_lib": None,
                "_workflow": None,
            },
        )
