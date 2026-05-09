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

from .assets import AssetScope, AssetsView, DataAssetLibrary, ImportAction
from .base import (
    _list_children,
    _load_metadata,
    _rebuild_container_index,
    _reconstruct,
    _save_metadata,
)
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
        self._data_assets: DataAssetLibrary | None = None
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
    def scope(self) -> AssetScope:
        return AssetScope(kind="project", ids=(self.id,))

    @property
    def assets(self) -> AssetsView:
        """Scope-filtered catalog view (read-only queries)."""
        return AssetsView(self.workspace.catalog, self.scope)

    @property
    def data_assets(self) -> DataAssetLibrary:
        if self._data_assets is None:
            self._data_assets = DataAssetLibrary(
                self.project_dir, self.scope, self.workspace.catalog
            )
        return self._data_assets

    # ── Persistence ─────────────────────────────────────────────────────

    def materialize(self) -> None:
        """Create filesystem structure and persist metadata (non-recursive)."""
        self.project_dir.mkdir(parents=True, exist_ok=True)
        _save_metadata(self.metadata, self.project_dir / "project.json")
        self._catalog_upsert()

    def save(self) -> None:
        """Persist current metadata to disk."""
        _save_metadata(self.metadata, self.project_dir / "project.json")
        self._catalog_upsert()

    def _catalog_upsert(self) -> None:
        self.workspace.catalog.upsert_project(
            {
                "project_id": self.metadata.id,
                "workspace_id": self.workspace.id,
                "name": self.metadata.name,
                "description": self.metadata.description,
                "owner": self.metadata.owner,
                "tags": list(self.metadata.tags),
                "path": str(self.project_dir.relative_to(self.workspace.root)),
                "created_at": self.metadata.created_at.isoformat(),
                "updated_at": self.metadata.created_at.isoformat(),
            }
        )

    def import_asset(
        self,
        name: str,
        src: str | Path,
        action: ImportAction = "copy",
        meta: dict[str, Any] | None = None,
    ):
        """Import a ``DataAsset`` into the project library."""
        return self.data_assets.import_asset(name, src, action, meta)

    # ── Experiment operations ───────────────────────────────────────────

    def Experiment(
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
        description: str = "",
        tags: list[str] | None = None,
        default_target: str | None = None,
    ) -> Experiment:
        """Idempotent constructor — return existing experiment if found, else create.

        If an experiment with the same ID (or slug from *name*) exists on
        disk, it is loaded and returned.  Otherwise a new experiment is
        constructed and materialized.

        ``description`` and ``tags`` are only applied on first creation;
        reloading an existing experiment does not overwrite them.

        For "must be new" semantics, use :meth:`create_experiment`. For
        "must already exist", use :meth:`experiment`.
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
                description=description,
                tags=tags,
                default_target=default_target,
            )
            exp.materialize()
            self._refresh_experiments_index()
        self._experiments_cache[exp.id] = exp
        return exp

    def create_experiment(
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
        description: str = "",
        tags: list[str] | None = None,
        default_target: str | None = None,
    ) -> Experiment:
        """Strict constructor — raise :class:`ExperimentExistsError` if exists."""
        from .errors import ExperimentExistsError

        exp_id = id if id is not None else slugify(name)
        if exp_id in self._experiments_cache:
            raise ExperimentExistsError(exp_id)
        exp_dir = self.project_dir / "experiments" / exp_id
        if exp_dir.exists():
            raise ExperimentExistsError(exp_id)
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
            description=description,
            tags=tags,
            default_target=default_target,
        )
        exp.materialize()
        self._refresh_experiments_index()
        self._experiments_cache[exp.id] = exp
        return exp

    def experiment(self, experiment_id: str) -> Experiment:
        """Strict getter — raise :class:`ExperimentNotFoundError` if absent."""
        from .errors import ExperimentNotFoundError

        if experiment_id in self._experiments_cache:
            return self._experiments_cache[experiment_id]
        exp_dir = self.project_dir / "experiments" / experiment_id
        if not exp_dir.exists():
            raise ExperimentNotFoundError(experiment_id)
        exp = self._load_experiment_from_dir(exp_dir)
        self._experiments_cache[exp.id] = exp
        return exp

    def registered_experiments(self) -> list[Experiment]:
        """Return only experiments explicitly registered in the current process.

        Unlike :meth:`list_experiments`, this never scans the disk — it
        returns exactly the ``Experiment`` instances that were constructed
        via ``project.experiment(...)`` in the running script. Use this
        when the caller's source of truth is the script (e.g.
        ``molexp run``); orphan experiment dirs left by prior runs or
        unrelated scripts are excluded.
        """
        return list(self._experiments_cache.values())

    def list_experiments(self) -> list[Experiment]:
        """List all experiments (disk scan merged with in-memory cache).

        For UI/CRUD/discovery. If you want only the experiments registered
        by the current script, use :meth:`registered_experiments` instead.
        """
        seen: dict[str, Experiment] = dict(self._experiments_cache)
        scanned = _list_children(
            children_dir=self.project_dir / "experiments",
            metadata_filename="experiment.json",
            metadata_cls=ExperimentMetadata,
            child_cls=Experiment,
            attrs_factory=lambda m: {
                "project": self,
                "metadata": m,
                "_data_assets": None,
                "_workflow": None,
                "_workflow_entrypoint": None,
                "_workflow_ir": None,
            },
        )
        for e in scanned:
            seen.setdefault(e.id, e)
        return list(seen.values())

    def delete_experiment(self, experiment_id: str) -> None:
        """Delete an experiment directory and cascade-drop its catalog rows.

        Raises:
            KeyError: If the experiment is not found.
        """
        import shutil

        exp_dir = self.project_dir / "experiments" / experiment_id
        if not exp_dir.exists():
            raise KeyError(f"Experiment '{experiment_id}' not found")
        shutil.rmtree(exp_dir)
        self._experiments_cache.pop(experiment_id, None)
        self.workspace.catalog.remove_experiment(experiment_id)
        self._refresh_experiments_index()

    # ── Internal ────────────────────────────────────────────────────────

    def _refresh_experiments_index(self) -> None:
        _rebuild_container_index(
            container_dir=self.project_dir / "experiments",
            index_filename="experiments.json",
            metadata_filename="experiment.json",
            fields=["id", "name", "description", "tags", "n_replicas", "created_at"],
        )

    def _load_experiment_from_dir(self, exp_dir: Path) -> Experiment:
        meta = _load_metadata(ExperimentMetadata, exp_dir / "experiment.json")
        return _reconstruct(
            Experiment,
            {
                "project": self,
                "metadata": meta,
                "_data_assets": None,
                "_workflow": None,
                "_workflow_entrypoint": None,
                "_workflow_ir": None,
            },
        )
