"""Workspace: top-level container with project management.

The Workspace is the root of the hierarchy and the only :class:`Folder`
whose ``parent`` is ``None``. Unlike lower levels, the workspace
constructor **does** ensure its root directory + ``workspace.json``
exist, because every other level needs a materialized workspace as
anchor.

Child factories (``.Project(...)``) are idempotent: they load existing
children from disk or create + materialize new ones.
"""

from __future__ import annotations

import shutil
import warnings
from pathlib import Path

from .assets import AssetCatalog, AssetScope, AssetsView, DataAssetLibrary
from .base import (
    _load_metadata,
    _rebuild_container_index,
    _save_metadata,
)
from .errors import ProjectExistsError, ProjectNotFoundError
from .folder import (
    WORKSPACE_PROJECT_KIND,
    WORKSPACE_ROOT_KIND,
    Folder,
)
from .models import FolderMetadata, WorkspaceMetadata
from .project import Project
from .subsystem import SubsystemStore
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
        project = ws.Project("QM9")
        exp = project.Experiment("baseline", params={"lr": 1e-3}, n_replicas=3)
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
        self._subsystem_stores: dict[str, SubsystemStore] = {}

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

    def subsystem_store(self, kind: str) -> SubsystemStore:
        """Vend a private :class:`SubsystemStore` for ``kind``.

        Same kind returns the same instance per workspace. Construction
        is side-effect-free; the directory is created on first
        :meth:`SubsystemStore.dir` / :meth:`SubsystemStore.file` call.

        Deprecated:
            Use :meth:`attach` or a typed ``*Folder`` subclass. Slated
            for removal in ``unify-folder-abstraction-03``; the
            ``DeprecationWarning`` is the bridge that lets workflow /
            agent callers migrate.
        """
        warnings.warn(
            "Workspace.subsystem_store(kind=...) is deprecated; "
            "use workspace.attach(...) or a typed *Folder subclass. "
            "Will be removed in unify-folder-abstraction-03.",
            DeprecationWarning,
            stacklevel=2,
        )
        cached = self._subsystem_stores.get(kind)
        if cached is not None:
            return cached
        store = SubsystemStore(self.root, kind)
        self._subsystem_stores[kind] = store
        return store

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

    # ── Project operations (typed wrappers over attach/create_child/get_child) ──

    def Project(self, name: str) -> Project:
        """Idempotent constructor — return existing project if found, else create.

        Within the same process, repeated calls with the same name return
        the **same** Project instance — preserving in-memory state such as
        bound workflows on child experiments.

        For "must be new" semantics, use :meth:`create_project`. For
        "must already exist" semantics, use :meth:`project`.
        """
        self._ensure_materialized()
        proj = self.attach(name, kind=WORKSPACE_PROJECT_KIND, child_cls=Project)
        if not isinstance(proj, Project):  # pragma: no cover — defensive
            raise TypeError(f"attach returned {type(proj).__name__}, expected Project")
        # Folder.attach caches on first create; mirror in legacy cache so
        # any caller reaching for ``_projects_cache`` keeps working until
        # sub-spec 03 cleans it up.
        self._projects_cache[proj.id] = proj
        self._refresh_projects_index()
        return proj

    def create_project(self, name: str) -> Project:
        """Strict constructor — raise :class:`ProjectExistsError` if exists.

        Mirror of :meth:`Project` for callers (CLI / API server) that
        require a fresh project and treat collision as an error.
        """
        self._ensure_materialized()
        proj = self.create_child(name, kind=WORKSPACE_PROJECT_KIND, child_cls=Project)
        if not isinstance(proj, Project):  # pragma: no cover — defensive
            raise TypeError(f"create_child returned {type(proj).__name__}, expected Project")
        self._projects_cache[proj.id] = proj
        self._refresh_projects_index()
        return proj

    def project(self, name_or_id: str) -> Project:
        """Strict getter — raise :class:`ProjectNotFoundError` if absent.

        Accepts a project name or its slugified ID.
        """
        proj = self.get_child(name_or_id, kind=WORKSPACE_PROJECT_KIND, child_cls=Project)
        if not isinstance(proj, Project):  # pragma: no cover — defensive
            raise TypeError(f"get_child returned {type(proj).__name__}, expected Project")
        self._projects_cache[proj.id] = proj
        return proj

    def registered_projects(self) -> list[Project]:
        """Return only projects explicitly registered in the current process.

        Unlike :meth:`list_projects`, this never scans the disk — it returns
        exactly the ``Project`` instances that were constructed via
        ``ws.Project(...)`` in the running script. Use this when the caller's
        source of truth is the script, not the workspace directory (e.g.
        ``molexp run`` dispatching workflows). Orphan project dirs left by
        unrelated scripts are excluded.
        """
        return list(self._projects_cache.values())

    def list_projects(self) -> list[Project]:
        """List all projects (disk scan merged with in-memory cache).

        For UI/CRUD/discovery. If you want only the projects registered by
        the current script, use :meth:`registered_projects` instead.
        """
        seen: dict[str, Project] = dict(self._projects_cache)
        projects_dir = self.root / "projects"
        if projects_dir.exists():
            for entry in sorted(projects_dir.iterdir()):
                if entry.is_dir() and (entry / "project.json").exists():
                    proj = Project._from_disk(entry, self)
                    seen.setdefault(proj.id, proj)
        return list(seen.values())

    def children(self, kind: str | None = None) -> list[Folder]:
        """List child folders, optionally filtered by ``kind``.

        Workspace's only entity children are :class:`Project` instances
        under ``projects/``. Subsystem dirs (under ``.subsystems/``) and
        the asset catalog (under ``.catalog/``) are deliberately excluded
        — they're not entity children and don't have the lifecycle
        contract that :meth:`Folder.children` returns. Sub-spec 03 makes
        them typed system-folder subclasses.
        """
        if kind is not None and kind != WORKSPACE_PROJECT_KIND:
            return []
        return list(self.list_projects())

    def delete_project(self, project_id: str) -> None:
        """Delete project directory and cascade-drop its catalog rows.

        Raises:
            KeyError: If project not found.
        """
        project_dir = self.root / "projects" / project_id
        if not project_dir.exists():
            raise KeyError(f"Project '{project_id}' not found")
        shutil.rmtree(project_dir)
        self._projects_cache.pop(project_id, None)
        self._children_cache.pop(project_id, None)
        self.catalog.remove_project(project_id)
        self._refresh_projects_index()

    # ── Internal ────────────────────────────────────────────────────────

    def _refresh_projects_index(self) -> None:
        _rebuild_container_index(
            container_dir=self.root / "projects",
            index_filename="projects.json",
            metadata_filename="project.json",
            fields=["id", "name", "description", "created_at"],
        )
