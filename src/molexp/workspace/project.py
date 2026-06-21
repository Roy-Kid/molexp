"""Project entity with experiment management.

Inherits :class:`Folder` (sub-spec 02) so it participates in the
unified workspace folder abstraction: ``kind`` is
:data:`WORKSPACE_PROJECT_KIND`, ``parent`` is the owning
:class:`Workspace`. Construction is side-effect free;
``workspace.add_project(...)`` materializes on disk at call-time
(idempotent: existing projects are loaded, missing ones are created).
"""

from __future__ import annotations

from pathlib import Path as _LocalPath
from typing import TYPE_CHECKING, Any, cast

from molexp._typing import JSONValue
from molexp.path import Path

if TYPE_CHECKING:
    from .catalog import AssetCatalog
    from .fs import FileSystem
    from .workspace import Workspace

from molexp.knowledge.types import concept_type

from .assets import AssetScope, AssetsView, DataAssetLibrary, ImportAction
from .base import (
    _load_metadata,
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
    _validate_target_registered,
)
from .fs import PathArg
from .library import Library
from .models import FolderMetadata, ProjectMetadata
from .utils import slugify


@concept_type(WORKSPACE_PROJECT_KIND)
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
        fs: FileSystem | None = None,
        _entity_metadata: ProjectMetadata | None = None,
    ) -> None:
        from .fs_local import LocalFileSystem

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

        self._parent = resolved_parent
        self._name = meta.id
        self._kind = kind
        self._root_path = None
        self._fs = fs or getattr(resolved_parent, "_fs", LocalFileSystem())
        self._metadata = FolderMetadata(
            id=meta.id,
            name=meta.name,
            kind=kind,
            created_at=meta.created_at,
            updated_at=meta.created_at,
        )
        self._children_cache = {}

        self._entity_metadata: ProjectMetadata = meta
        self._data_assets: DataAssetLibrary | None = None
        self._library: Library | None = None

    # ── Folder hooks ─────────────────────────────────────────────────────

    def resolve(self) -> Path:
        return self.project_dir

    @classmethod
    def child_dir(cls, parent: Folder, derived_id: str) -> Path:
        """Folder hook — projects live under ``projects/<id>/``."""
        return Path(parent._fs.join(parent.path(), "projects", derived_id))

    @classmethod
    def from_disk(cls, child_dir: PathArg, parent: Folder) -> Project:
        """Load ``project.json`` and rebuild entity state. See Folder.from_disk hook docs."""
        meta = _load_metadata(
            ProjectMetadata, parent._fs.join(child_dir, "project.json"), fs=parent._fs
        )
        folder_meta = FolderMetadata(
            id=meta.id,
            name=meta.name,
            kind=WORKSPACE_PROJECT_KIND,
            created_at=meta.created_at,
            updated_at=meta.created_at,
        )
        attrs = cls.base_from_disk_attrs(parent, folder_meta) | {
            "_entity_metadata": meta,
            "_data_assets": None,
            "_library": None,
        }
        return _reconstruct(cls, attrs)

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
    def created_at(self):  # noqa: ANN201
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
        ws_root = self.workspace.resolve()
        return Path(self._fs.join(ws_root, "projects", self.id))

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

    @property
    def library(self) -> Library:
        """Notes + references store for this project scope."""
        if self._library is None:
            self._library = Library(self.project_dir, self.scope, self.workspace.catalog)
        return self._library

    # ── Persistence ─────────────────────────────────────────────────────

    def materialize(self) -> None:
        """Create filesystem structure and persist metadata (non-recursive)."""
        d = self.project_dir
        self._fs.mkdir(d, parents=True, exist_ok=True)
        meta_path = self._fs.join(d, "project.json")
        _save_metadata(self._entity_metadata, meta_path, fs=self._fs)
        self._catalog_upsert()

    def save(self) -> None:
        """Persist current metadata to disk."""
        meta_path = self._fs.join(self.project_dir, "project.json")
        _save_metadata(self._entity_metadata, meta_path, fs=self._fs)
        self._catalog_upsert()

    def _write_catalog_row(self, catalog: AssetCatalog) -> None:
        meta = self._entity_metadata
        catalog.upsert_project(
            {
                "project_id": meta.id,
                "workspace_id": self.workspace.id,
                "name": meta.name,
                "description": meta.description,
                "owner": meta.owner,
                "tags": list(meta.tags),
                "path": "projects/" + self.id,
                "created_at": meta.created_at.isoformat(),
                "updated_at": meta.created_at.isoformat(),
            }
        )

    def import_asset(  # noqa: ANN201
        self,
        name: str,
        src: str | _LocalPath,
        action: ImportAction = "copy",
        meta: dict[str, Any] | None = None,
    ):
        """Import a ``DataAsset`` into the project library."""
        return self.data_assets.import_asset(name, src, action, meta)

    # ── Experiment CRUD: typed semantic sugar over generic Folder CRUD ─────

    def add_experiment(
        self,
        name: str,
        *,
        id: str | None = None,
        params: dict[str, JSONValue] | None = None,
        n_replicas: int = 1,
        seeds: list[int] | None = None,
        workflow_source: str | None = None,
        workflow_type: str | None = None,
        git_commit: str | None = None,
        description: str = "",
        tags: list[str] | None = None,
        default_target: str | None = None,
    ) -> Experiment:
        """Mount an experiment under this project (idempotent on slug).

        One-line wrapper over generic ``add_folder``. The slugified
        ``name`` doubles as the experiment id when no explicit ``id=``
        is given — matching the legacy ``Project.Experiment``
        factory semantics so ``add_experiment("counter")`` twice returns
        the same instance.

        The signature is spelled out explicitly (mirroring the
        :class:`~molexp.workspace.models.ExperimentMetadata` fields the
        :class:`Experiment` constructor accepts) so a typo such as
        ``prams=`` raises ``TypeError`` instead of flowing silently.
        ``params`` is the canonical spelling for the parameter dict.
        """
        resolved_id = id if id is not None else slugify(name)
        _validate_target_registered(self.workspace, default_target)
        child = self._construct_child(
            Experiment,
            name,
            id=resolved_id,
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
        return cast(Experiment, self.add_folder(child))

    def experiment(self, name: str, **kwargs: Any) -> Experiment:  # noqa: ANN401
        """Fluent create-or-get alias for :meth:`add_experiment` (idempotent)."""
        return self.add_experiment(name, **kwargs)

    def get_experiment(self, name: str) -> Experiment:
        return self.get_folder(name, cls=Experiment)

    def has_experiment(self, name: str) -> bool:
        return self.has_folder(name, cls=Experiment)

    def remove_experiment(self, name: str) -> None:
        slug = slugify(name)
        self.remove_folder(name, cls=Experiment)
        self.workspace.catalog.remove_experiment(slug)

    def list_experiments(self) -> list[Experiment]:
        """List all experiments in this project via the typed CRUD view."""
        return self.list_folders(cls=Experiment)

    def children(self, kind: str | None = None) -> list[Folder]:
        """List entity children (currently only :class:`Experiment`)."""
        if kind is not None and kind != WORKSPACE_EXPERIMENT_KIND:
            return []
        return list(self.list_experiments())
