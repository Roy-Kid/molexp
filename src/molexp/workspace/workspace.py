"""Workspace: top-level container with project management.

The Workspace is the root of the hierarchy.  Unlike lower levels, the
workspace constructor **does** ensure its root directory + ``workspace.json``
exist, because every other level needs a materialized workspace as anchor.

Child factories (``.project()``) are idempotent: they load existing
children from disk or create + materialize new ones.
"""

from __future__ import annotations

import shutil
from pathlib import Path

from .assets import AssetCatalog, AssetScope, AssetsView, DataAssetLibrary
from .base import (
    _list_children,
    _load_metadata,
    _rebuild_container_index,
    _reconstruct,
    _save_metadata,
)
from .models import ProjectMetadata, WorkspaceMetadata
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


class Workspace:
    """Top-level workspace with project management and global asset library.

    Example::

        ws = Workspace("./lab")
        project = ws.project("QM9")
        exp = project.experiment("baseline", params={"lr": 1e-3}, n_replicas=3)
    """

    def __init__(self, root: Path | str, name: str | None = None) -> None:
        # CLI --workspace wins over script-hardcoded roots.
        if _cli_root_override is not None:
            self.root = _cli_root_override
        else:
            self.root = Path(root).resolve()
        metadata_file = self.root / "workspace.json"
        if metadata_file.exists():
            self.metadata = _load_metadata(WorkspaceMetadata, metadata_file)
        else:
            if name is None:
                name = self.root.name
            self.metadata = WorkspaceMetadata(id=slugify(name), name=name)
        self._data_assets: DataAssetLibrary | None = None
        self._catalog: AssetCatalog | None = None
        self._projects_cache: dict[str, Project] = {}

    def _ensure_materialized(self) -> None:
        if not (self.root / "workspace.json").exists():
            self.materialize()

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

    # ── Persistence ─────────────────────────────────────────────────────

    def materialize(self) -> None:
        """Create filesystem structure and persist metadata (non-recursive)."""
        self.root.mkdir(parents=True, exist_ok=True)
        _save_metadata(self.metadata, self.root / "workspace.json")
        self._catalog_upsert()

    def save(self) -> None:
        """Persist current metadata to disk."""
        _save_metadata(self.metadata, self.root / "workspace.json")
        self._catalog_upsert()

    def _catalog_upsert(self) -> None:
        self.catalog.upsert_workspace(
            {
                "workspace_id": self.metadata.id,
                "root_path": str(self.root),
                "name": self.metadata.name,
                "created_at": self.metadata.created_at.isoformat(),
                "updated_at": self.metadata.created_at.isoformat(),
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
        meta = _load_metadata(WorkspaceMetadata, metadata_file)
        return _reconstruct(
            cls,
            {
                "root": root,
                "metadata": meta,
                "_data_assets": None,
                "_catalog": None,
                "_projects_cache": {},
            },
        )

    # ── Project operations ──────────────────────────────────────────────

    def project(self, name: str) -> Project:
        """Get-or-create a project (idempotent, materialized immediately).

        Within the same process, repeated calls with the same name return
        the **same** Project instance — preserving in-memory state such as
        bound workflows on child experiments.
        """
        self._ensure_materialized()
        project_id = slugify(name)
        if project_id in self._projects_cache:
            return self._projects_cache[project_id]
        project_dir = self.root / "projects" / project_id
        if project_dir.exists():
            project = self._load_project_from_dir(project_dir)
        else:
            project = Project(name=name, workspace=self)
            project.materialize()
            self._refresh_projects_index()
        self._projects_cache[project_id] = project
        return project

    def get_project(self, name_or_id: str) -> Project | None:
        """Get project by name or slugified ID."""
        for candidate in (name_or_id, slugify(name_or_id)):
            if candidate in self._projects_cache:
                return self._projects_cache[candidate]
            project_dir = self.root / "projects" / candidate
            if project_dir.exists():
                project = self._load_project_from_dir(project_dir)
                self._projects_cache[project.id] = project
                return project
        return None

    def registered_projects(self) -> list[Project]:
        """Return only projects explicitly registered in the current process.

        Unlike :meth:`list_projects`, this never scans the disk — it returns
        exactly the ``Project`` instances that were constructed via
        ``ws.project(...)`` in the running script. Use this when the caller's
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
        scanned = _list_children(
            children_dir=self.root / "projects",
            metadata_filename="project.json",
            metadata_cls=ProjectMetadata,
            child_cls=Project,
            attrs_factory=lambda m: {
                "workspace": self,
                "metadata": m,
                "_data_assets": None,
                "_experiments_cache": {},
            },
        )
        for p in scanned:
            seen.setdefault(p.id, p)
        return list(seen.values())

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

    def _load_project_from_dir(self, project_dir: Path) -> Project:
        meta = _load_metadata(ProjectMetadata, project_dir / "project.json")
        return _reconstruct(
            Project,
            {
                "workspace": self,
                "metadata": meta,
                "_data_assets": None,
                "_experiments_cache": {},
            },
        )
