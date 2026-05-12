"""Project entity with experiment management.

Inherits :class:`Folder` (sub-spec 02) so it participates in the
unified workspace folder abstraction: ``kind`` is
:data:`WORKSPACE_PROJECT_KIND`, ``parent`` is the owning
:class:`Workspace`. Construction is side-effect free;
``workspace.Project(...)`` materializes on disk at call-time
(idempotent: existing projects are loaded, missing ones are created).
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from .workspace import Workspace

from .assets import AssetScope, AssetsView, DataAssetLibrary, ImportAction
from .base import (
    _load_metadata,
    _rebuild_container_index,
    _reconstruct,
    _save_metadata,
)
from .errors import (
    ProjectExistsError,
    ProjectNotFoundError,
)
from .experiment import Experiment
from .folder import (
    WORKSPACE_EXPERIMENT_KIND,
    WORKSPACE_PROJECT_KIND,
    Folder,
)
from .models import FolderMetadata, ProjectMetadata
from .utils import slugify


class Project(Folder):
    """Research project container.

    Example::

        ws = Workspace("./lab")
        project = ws.Project("QM9")
        exp = project.Experiment("baseline", params={"lr": 1e-3})
    """

    _exists_error_cls = ProjectExistsError
    _not_found_error_cls = ProjectNotFoundError

    def __init__(
        self,
        *,
        parent: Workspace | None = None,
        name: str,
        kind: str = WORKSPACE_PROJECT_KIND,
        id: str | None = None,
        workspace: Workspace | None = None,
        _entity_metadata: ProjectMetadata | None = None,
    ) -> None:
        # ``parent`` (Folder convention) and ``workspace`` (entity alias)
        # are accepted interchangeably for backwards compatibility with
        # in-tree call sites; new code should pass ``parent=``.
        resolved_parent = parent if parent is not None else workspace
        if resolved_parent is None:
            raise ValueError("Project: parent (or workspace) is required")

        meta = (
            _entity_metadata
            if _entity_metadata is not None
            else ProjectMetadata(
                id=id if id is not None else slugify(name),
                name=name,
            )
        )

        # Bypass Folder.__init__ — entity name may contain characters
        # the kind-pattern rejects; the slugified id (entity-managed) is
        # the kind-safe form.
        self._parent = resolved_parent
        self._name = meta.id
        self._kind = kind
        self._root_path = None
        self._metadata = FolderMetadata(
            id=meta.id,
            name=meta.name,
            kind=kind,
            created_at=meta.created_at,
            updated_at=meta.created_at,
        )
        self._children_cache = {}

        # Entity-specific state
        self._entity_metadata: ProjectMetadata = meta
        self._data_assets: DataAssetLibrary | None = None
        self._experiments_cache: dict[str, Experiment] = {}

    # ── Folder hooks ─────────────────────────────────────────────────────

    def _compute_path(self) -> Path:
        return self.project_dir

    @classmethod
    def _child_dir(cls, parent: Folder, derived_id: str) -> Path:
        """:class:`Folder.attach` hook — projects live under ``projects/<id>/``."""
        return parent.path() / "projects" / derived_id

    @classmethod
    def _from_disk(cls, child_dir: Path, parent: Folder) -> Project:
        """:class:`Folder.attach` hook — load ``project.json`` and rebuild entity state."""
        meta = _load_metadata(ProjectMetadata, child_dir / "project.json")
        return _reconstruct(
            cls,
            {
                "_parent": parent,
                "_name": meta.id,
                "_kind": WORKSPACE_PROJECT_KIND,
                "_root_path": None,
                "_metadata": FolderMetadata(
                    id=meta.id,
                    name=meta.name,
                    kind=WORKSPACE_PROJECT_KIND,
                    created_at=meta.created_at,
                    updated_at=meta.created_at,
                ),
                "_children_cache": {},
                "_entity_metadata": meta,
                "_data_assets": None,
                "_experiments_cache": {},
            },
        )

    # ── Properties (entity-specific) ─────────────────────────────────────

    @property
    def workspace(self) -> Workspace:
        """The owning :class:`Workspace` (alias for :attr:`Folder.parent`)."""
        if self._parent is None:  # pragma: no cover — Project always has a parent
            raise RuntimeError("Project has no parent workspace")
        return cast("Workspace", self._parent)

    @property
    def metadata(self) -> ProjectMetadata:  # type: ignore[override]
        """Project-entity metadata (shadows :attr:`Folder.metadata`)."""
        return self._entity_metadata

    @metadata.setter
    def metadata(self, value: ProjectMetadata) -> None:
        self._entity_metadata = value

    @property
    def id(self) -> str:
        return self._entity_metadata.id

    @property
    def name(self) -> str:
        return self._entity_metadata.name

    @property
    def created_at(self):
        return self._entity_metadata.created_at

    @property
    def description(self) -> str:
        return self._entity_metadata.description

    @property
    def owner(self) -> str:
        return self._entity_metadata.owner

    @property
    def tags(self) -> list[str]:
        return self._entity_metadata.tags

    @property
    def config(self) -> dict[str, Any]:
        return self._entity_metadata.config

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
        _save_metadata(self._entity_metadata, self.project_dir / "project.json")
        self._catalog_upsert()

    def save(self) -> None:
        """Persist current metadata to disk."""
        _save_metadata(self._entity_metadata, self.project_dir / "project.json")
        self._catalog_upsert()

    def _catalog_upsert(self) -> None:
        meta = self._entity_metadata
        self.workspace.catalog.upsert_project(
            {
                "project_id": meta.id,
                "workspace_id": self.workspace.id,
                "name": meta.name,
                "description": meta.description,
                "owner": meta.owner,
                "tags": list(meta.tags),
                "path": str(self.project_dir.relative_to(self.workspace.root)),
                "created_at": meta.created_at.isoformat(),
                "updated_at": meta.created_at.isoformat(),
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

    # ── Experiment operations (typed wrappers over attach/create_child/get_child) ──

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
        exp = self.attach(
            name,
            kind=WORKSPACE_EXPERIMENT_KIND,
            child_cls=Experiment,
            id=id,
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
        if not isinstance(exp, Experiment):  # pragma: no cover — defensive
            raise TypeError(f"attach returned {type(exp).__name__}, expected Experiment")
        self._experiments_cache[exp.id] = exp
        self._refresh_experiments_index()
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
        exp = self.create_child(
            name,
            kind=WORKSPACE_EXPERIMENT_KIND,
            child_cls=Experiment,
            id=id,
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
        if not isinstance(exp, Experiment):  # pragma: no cover — defensive
            raise TypeError(f"create_child returned {type(exp).__name__}, expected Experiment")
        self._experiments_cache[exp.id] = exp
        self._refresh_experiments_index()
        return exp

    def experiment(self, experiment_id: str) -> Experiment:
        """Strict getter — raise :class:`ExperimentNotFoundError` if absent."""
        exp = self.get_child(experiment_id, kind=WORKSPACE_EXPERIMENT_KIND, child_cls=Experiment)
        if not isinstance(exp, Experiment):  # pragma: no cover — defensive
            raise TypeError(f"get_child returned {type(exp).__name__}, expected Experiment")
        self._experiments_cache[exp.id] = exp
        return exp

    def registered_experiments(self) -> list[Experiment]:
        """Return only experiments explicitly registered in the current process.

        Unlike :meth:`list_experiments`, this never scans the disk — it
        returns exactly the ``Experiment`` instances that were constructed
        via ``project.Experiment(...)`` in the running script.
        """
        return list(self._experiments_cache.values())

    def list_experiments(self) -> list[Experiment]:
        """List all experiments (disk scan merged with in-memory cache).

        For UI/CRUD/discovery. If you want only the experiments registered
        by the current script, use :meth:`registered_experiments` instead.
        """
        seen: dict[str, Experiment] = dict(self._experiments_cache)
        exp_dir_parent = self.project_dir / "experiments"
        if exp_dir_parent.exists():
            for entry in sorted(exp_dir_parent.iterdir()):
                if entry.is_dir() and (entry / "experiment.json").exists():
                    e = Experiment._from_disk(entry, self)
                    seen.setdefault(e.id, e)
        return list(seen.values())

    def children(self, kind: str | None = None) -> list[Folder]:
        """List child folders, optionally filtered by ``kind``.

        Project's only entity children are :class:`Experiment` instances
        under ``experiments/``.
        """
        if kind is not None and kind != WORKSPACE_EXPERIMENT_KIND:
            return []
        return list(self.list_experiments())

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
        self._children_cache.pop(experiment_id, None)
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
