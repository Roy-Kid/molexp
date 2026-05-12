"""Project entity with experiment management.

Inherits :class:`Folder` (sub-spec 02) so it participates in the
unified workspace folder abstraction: ``kind`` is
:data:`WORKSPACE_PROJECT_KIND`, ``parent`` is the owning
:class:`Workspace`. Construction is side-effect free;
``workspace.add_project(...)`` materializes on disk at call-time
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
        project = ws.add_project("QM9")
        exp = project.add_experiment("baseline", params={"lr": 1e-3})
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

    # ── Experiment CRUD: typed semantic sugar over generic Folder CRUD ─────

    def add_experiment(self, name: str, **kwargs: Any) -> Experiment:
        """Mount an experiment under this project (idempotent on slug).

        One-line wrapper over generic ``add_folder``. The slugified
        ``name`` doubles as the experiment id when no explicit ``id=``
        kwarg is given — matching the legacy ``Project.Experiment``
        factory semantics so ``add_experiment("counter")`` twice returns
        the same instance.
        """
        slug = slugify(name)
        explicit_id = kwargs.pop("id", None)
        resolved_id = explicit_id if explicit_id is not None else slug
        cached = self._experiments_cache.get(resolved_id)
        if cached is not None:
            return cached
        child_dir = Experiment._child_dir(self, resolved_id)
        if child_dir.is_dir():
            existing = Experiment._from_disk(child_dir, self)
            self._experiments_cache[existing.id] = existing
            self._children_cache[existing.id] = existing
            return existing
        exp = Experiment(parent=self, name=name, id=resolved_id, **kwargs)
        exp.materialize()
        self._experiments_cache[exp.id] = exp
        self._children_cache[exp.id] = exp
        self._upsert_index_row(exp)
        return exp

    def get_experiment(self, name: str) -> Experiment:
        exp = self.get_folder(name, cls=Experiment)
        self._experiments_cache[exp.id] = exp
        return exp

    def has_experiment(self, name: str) -> bool:
        return self.has_folder(name, cls=Experiment)

    def remove_experiment(self, name: str) -> None:
        slug = slugify(name)
        self._experiments_cache.pop(slug, None)
        self.remove_folder(name, cls=Experiment)
        self.workspace.catalog.remove_experiment(slug)
        self._refresh_experiments_index()

    def list_experiments(self) -> list[Experiment]:
        """List all experiments in this project via the typed CRUD view."""
        return self.list_folders(cls=Experiment)

    def children(self, kind: str | None = None) -> list[Folder]:
        """List entity children (currently only :class:`Experiment`)."""
        if kind is not None and kind != WORKSPACE_EXPERIMENT_KIND:
            return []
        return list(self.list_experiments())

    def _refresh_experiments_index(self) -> None:
        _rebuild_container_index(
            container_dir=self.project_dir / "experiments",
            index_filename="experiments.json",
            metadata_filename="experiment.json",
            fields=["id", "name", "description", "tags", "n_replicas", "created_at"],
        )
