"""Workspace: top-level container with project management.

The Workspace is the root of the hierarchy and the only :class:`Folder`
whose ``parent`` is ``None``. Unlike lower levels, the workspace
constructor **does** ensure its root directory + ``workspace.json``
exist, because every other level needs a materialized workspace as
anchor.

Child factories (``.add_project(...)``) are idempotent: they load existing
children from disk or create + materialize new ones.
"""

from __future__ import annotations

import shutil
from pathlib import Path

from .assets import AssetScope, AssetsView, DataAssetLibrary
from .base import (
    _load_metadata,
    _rebuild_container_index,
    _save_metadata,
)
from .cache import CacheFolder
from .catalog import AssetCatalog
from .errors import ProjectExistsError, ProjectNotFoundError
from .folder import (
    WORKSPACE_PROJECT_KIND,
    WORKSPACE_ROOT_KIND,
    Folder,
)
from .models import FolderMetadata, WorkspaceMetadata
from .project import Project
from .utils import slugify

# CLI-level root override: set by ``molexp run -w PATH`` before executing the
# user script, so every ``me.Workspace(...)`` in that script resolves to the
# overridden path instead of its hardcoded argument. ``None`` means no
# override — ``Workspace(root)`` uses the caller-supplied root as-is.
_cli_root_override: Path | None = None


def set_cli_root_override(path: Path | str | None) -> None:
    """Set (or clear) the CLI-level workspace root override.

    When set, :class:`Workspace` constructors use this path instead of the
    ``root`` argument they were called with. Intended solely for ``molexp
    run -w PATH`` to make the CLI flag authoritative over script-hardcoded
    workspace roots.
    """
    global _cli_root_override
    _cli_root_override = Path(path).resolve() if path is not None else None


class Workspace(Folder):
    """Top-level workspace with project management and global asset library.

    Inherits :class:`Folder` (sub-spec 02): ``kind`` is
    :data:`WORKSPACE_ROOT_KIND`, ``parent`` is ``None``. The workspace
    is its own root — :meth:`_compute_path` returns :attr:`root`
    directly rather than nesting one level deeper like other Folder
    subclasses.

    Example::

        ws = Workspace("./lab")
        project = ws.add_project("QM9")
        exp = project.add_experiment("baseline", params={"lr": 1e-3}, n_replicas=3)
    """

    _exists_error_cls = ProjectExistsError
    _not_found_error_cls = ProjectNotFoundError

    def __init__(self, root: Path | str, name: str | None = None) -> None:
        # CLI --workspace wins over script-hardcoded roots.
        resolved_root = (
            _cli_root_override if _cli_root_override is not None else Path(root).resolve()
        )

        metadata_file = resolved_root / "workspace.json"
        if metadata_file.exists():
            entity_meta = _load_metadata(WorkspaceMetadata, metadata_file)
        else:
            display_name = name if name is not None else resolved_root.name
            entity_meta = WorkspaceMetadata(id=slugify(display_name), name=display_name)

        # Workspace bypasses ``Folder.__init__`` because the human-readable
        # ``name`` may contain characters (e.g. spaces, uppercase) that the
        # ``_KIND_PATTERN`` validator rejects — the slugified ``id`` is the
        # kind-safe form persisted in :class:`FolderMetadata`.
        self._parent = None
        self._name = entity_meta.id
        self._kind = WORKSPACE_ROOT_KIND
        self._root_path = resolved_root
        self._metadata = FolderMetadata(
            id=entity_meta.id,
            name=entity_meta.name,
            kind=WORKSPACE_ROOT_KIND,
            created_at=entity_meta.created_at,
            updated_at=entity_meta.created_at,
        )
        self._children_cache = {}

        # Entity-specific state
        self.root = resolved_root
        self._entity_metadata: WorkspaceMetadata = entity_meta
        self._data_assets: DataAssetLibrary | None = None
        self._catalog: AssetCatalog | None = None
        self._projects_cache: dict[str, Project] = {}
        self._cache_folder: CacheFolder | None = None

    # ── Folder hooks ─────────────────────────────────────────────────────

    def _compute_path(self) -> Path:
        """Workspace IS its own on-disk dir; no parent nesting."""
        return self.root

    def _ensure_materialized(self) -> None:
        if not (self.root / "workspace.json").exists():
            self.materialize()

    # ── Properties (entity-specific) ─────────────────────────────────────

    @property
    def metadata(self) -> WorkspaceMetadata:  # type: ignore[override]
        """Workspace-entity metadata (shadows :attr:`Folder.metadata`).

        :attr:`Folder.folder_metadata` still returns the kind-uniform
        :class:`FolderMetadata` view.
        """
        return self._entity_metadata

    @metadata.setter
    def metadata(self, value: WorkspaceMetadata) -> None:
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
    def scope(self) -> AssetScope:
        return AssetScope(kind="workspace", ids=())

    @property
    def assets(self) -> AssetsView:
        """Scope-filtered catalog view (read-only queries)."""
        return AssetsView(self.catalog, self.scope)

    @property
    def data_assets(self) -> DataAssetLibrary:
        """Library for importing ``DataAsset`` inputs."""
        if self._data_assets is None:
            self._data_assets = DataAssetLibrary(self.root, self.scope, self.catalog)
        return self._data_assets

    @property
    def catalog(self) -> AssetCatalog:
        """Workspace-wide JSON asset + entity catalog.

        Lazily constructed.  The catalog directory is created on first write.
        """
        if self._catalog is None:
            self._catalog = AssetCatalog(self.root)
        return self._catalog

    # ── System folder accessors (singletons via lowercase property) ──────

    @property
    def cache(self) -> CacheFolder:  # type: ignore[override]
        """The (single) :class:`CacheFolder` for this workspace.

        Lazily constructed; identity-stable. ``ws.cache is ws.cache``.
        The underlying ``<root>/cache/`` directory is created on first
        read/write through the folder.
        """
        if self._cache_folder is None:
            self._cache_folder = CacheFolder(
                parent=self,
                name="cache",
                kind="workspace.cache",
            )
        return self._cache_folder

    # ── Persistence ─────────────────────────────────────────────────────

    def materialize(self) -> None:
        """Create filesystem structure and persist metadata (non-recursive)."""
        self.root.mkdir(parents=True, exist_ok=True)
        _save_metadata(self._entity_metadata, self.root / "workspace.json")
        self._catalog_upsert()

    def save(self) -> None:
        """Persist current metadata to disk."""
        _save_metadata(self._entity_metadata, self.root / "workspace.json")
        self._catalog_upsert()

    def _catalog_upsert(self) -> None:
        meta = self._entity_metadata
        self.catalog.upsert_workspace(
            {
                "workspace_id": meta.id,
                "root_path": str(self.root),
                "name": meta.name,
                "created_at": meta.created_at.isoformat(),
                "updated_at": meta.created_at.isoformat(),
            }
        )

    # ── Alternative constructors (kept for callers that still use them) ──

    @classmethod
    def load(cls, root: Path | str) -> Workspace:
        """Load an existing workspace from disk (raises if missing)."""
        root = Path(root).resolve()
        metadata_file = root / "workspace.json"
        if not metadata_file.exists():
            raise FileNotFoundError(f"Workspace metadata not found at {metadata_file}")
        # Constructor handles load-or-create idempotently.
        return cls(root)

    # ── Project CRUD: typed semantic sugar over generic Folder CRUD ────────

    def add_project(self, name: str) -> Project:
        """Mount a project under this workspace (idempotent on slug).

        One-line wrapper over generic ``add_folder``:
        ``ws.add_folder(Project(parent=ws, name=name))``.

        Returns the existing project on slug collision (in-memory cache
        or on-disk dir); fresh-creates + materializes otherwise.
        """
        self._ensure_materialized()
        # Project's __init__ currently requires parent; pass self so the
        # entity can compute its on-disk path without going through the
        # unmounted state. add_folder then validates parent is self and
        # skips the wire-up step.
        slug = slugify(name)
        cached = self._children_cache.get(slug)
        if isinstance(cached, Project):
            return cached
        child_dir = Project._child_dir(self, slug)
        if child_dir.is_dir():
            existing = Project._from_disk(child_dir, self)
            self._children_cache[slug] = existing
            self._projects_cache[existing.id] = existing
            return existing
        proj = Project(parent=self, name=name)
        proj.materialize()
        self._children_cache[slug] = proj
        self._projects_cache[proj.id] = proj
        self._upsert_index_row(proj)
        return proj

    def get_project(self, name: str) -> Project:
        """Strict getter — raise :class:`ProjectNotFoundError` if absent."""
        proj = self.get_folder(name, cls=Project)
        self._projects_cache[proj.id] = proj
        return proj

    def has_project(self, name: str) -> bool:
        return self.has_folder(name, cls=Project)

    # ``list_projects`` is defined below as the legacy method; semantics
    # match the generic ``list_folders(cls=Project)`` view, so we keep
    # the legacy implementation for now to avoid behavioral drift.

    def remove_project(self, name: str) -> None:
        """Delete project directory + cascade-drop catalog rows + drop indices."""
        slug = slugify(name)
        if slug in self._projects_cache:
            self._projects_cache.pop(slug, None)
        self.remove_folder(name, cls=Project)
        self.catalog.remove_project(slug)
        self._refresh_projects_index()

    def list_projects(self) -> list[Project]:
        """List all projects in this workspace via the typed CRUD view."""
        return self.list_folders(cls=Project)

    def children(self, kind: str | None = None) -> list[Folder]:
        """List entity children (currently only :class:`Project`)."""
        if kind is not None and kind != WORKSPACE_PROJECT_KIND:
            return []
        return list(self.list_projects())

    def _refresh_projects_index(self) -> None:
        _rebuild_container_index(
            container_dir=self.root / "projects",
            index_filename="projects.json",
            metadata_filename="project.json",
            fields=["id", "name", "description", "created_at"],
        )
