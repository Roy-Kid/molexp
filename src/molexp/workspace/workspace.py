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

from .asset import AssetLibrary
from .base import _list_children, _load_metadata, _reconstruct, _save_metadata
from .models import ProjectMetadata, WorkspaceMetadata
from .project import Project
from .utils import slugify


class Workspace:
    """Top-level workspace with project management and global asset library.

    Example::

        ws = Workspace("./lab")
        project = ws.project("QM9")
        exp = project.experiment("baseline", params={"lr": 1e-3}, n_replicas=3)
    """

    def __init__(self, root: Path | str, name: str | None = None) -> None:
        self.root = Path(root).resolve()
        metadata_file = self.root / "workspace.json"
        if metadata_file.exists():
            self.metadata = _load_metadata(WorkspaceMetadata, metadata_file)
        else:
            if name is None:
                name = self.root.name
            self.metadata = WorkspaceMetadata(id=slugify(name), name=name)
        self._assets_lib: AssetLibrary | None = None
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
    def assets(self) -> AssetLibrary:
        if self._assets_lib is None:
            self._assets_lib = AssetLibrary(self.root / "assets")
        return self._assets_lib

    # ── Persistence ─────────────────────────────────────────────────────

    def materialize(self) -> None:
        """Create filesystem structure and persist metadata (non-recursive)."""
        self.root.mkdir(parents=True, exist_ok=True)
        _save_metadata(self.metadata, self.root / "workspace.json")

    def save(self) -> None:
        """Persist current metadata to disk."""
        _save_metadata(self.metadata, self.root / "workspace.json")

    # ── Alternative constructors (kept for callers that still use them) ──

    @classmethod
    def load(cls, root: Path | str) -> Workspace:
        """Load an existing workspace from disk (raises if missing)."""
        root = Path(root).resolve()
        metadata_file = root / "workspace.json"
        if not metadata_file.exists():
            raise FileNotFoundError(
                f"Workspace metadata not found at {metadata_file}"
            )
        meta = _load_metadata(WorkspaceMetadata, metadata_file)
        return _reconstruct(
            cls,
            {
                "root": root,
                "metadata": meta,
                "_assets_lib": None,
                "_projects_cache": {},
            },
        )

    @classmethod
    def from_path(cls, root: Path | str) -> Workspace:
        """Compatibility alias for ``Workspace(root)``."""
        return cls(root)

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

    def list_projects(self) -> list[Project]:
        """List all projects (disk scan merged with in-memory cache)."""
        seen: dict[str, Project] = dict(self._projects_cache)
        scanned = _list_children(
            children_dir=self.root / "projects",
            metadata_filename="project.json",
            metadata_cls=ProjectMetadata,
            child_cls=Project,
            attrs_factory=lambda m: {
                "workspace": self,
                "metadata": m,
                "_assets_lib": None,
                "_experiments_cache": {},
            },
        )
        for p in scanned:
            seen.setdefault(p.id, p)
        return list(seen.values())

    def delete_project(self, project_id: str) -> None:
        """Delete project directory.

        Raises:
            KeyError: If project not found.
        """
        project_dir = self.root / "projects" / project_id
        if not project_dir.exists():
            raise KeyError(f"Project '{project_id}' not found")
        shutil.rmtree(project_dir)
        self._projects_cache.pop(project_id, None)

    # ── Internal ────────────────────────────────────────────────────────

    def _load_project_from_dir(self, project_dir: Path) -> Project:
        meta = _load_metadata(ProjectMetadata, project_dir / "project.json")
        return _reconstruct(
            Project,
            {
                "workspace": self,
                "metadata": meta,
                "_assets_lib": None,
                "_experiments_cache": {},
            },
        )
