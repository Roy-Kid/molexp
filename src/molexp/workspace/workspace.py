"""Workspace: top-level container with project management.

Construction is side-effect free; call ``materialize()`` to write to disk.
Children are discovered by scanning the filesystem — no child list in metadata.
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

        workspace = Workspace(root="./lab", name="My Lab")
        workspace.materialize()
        project = workspace.create_project(name="QM9")
    """

    def __init__(self, root: Path | str, name: str | None = None) -> None:
        self.root = Path(root).resolve()
        if name is None:
            name = self.root.name
        self.metadata = WorkspaceMetadata(id=slugify(name), name=name)
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
    def assets(self) -> AssetLibrary:
        if self._assets_lib is None:
            self._assets_lib = AssetLibrary(self.root / "assets")
        return self._assets_lib

    # ── Persistence ─────────────────────────────────────────────────────

    def materialize(self) -> None:
        """Create filesystem structure and persist metadata."""
        self.root.mkdir(parents=True, exist_ok=True)
        _save_metadata(self.metadata, self.root / "workspace.json")

    def save(self) -> None:
        """Persist current metadata to disk."""
        _save_metadata(self.metadata, self.root / "workspace.json")

    # ── Constructors ────────────────────────────────────────────────────

    @classmethod
    def from_config(cls, config: object) -> Workspace:
        """Load workspace from a molcfg ``Config`` object.

        Args:
            config: A ``molcfg.Config`` with a ``workspace_root`` key.

        Returns:
            Loaded or created workspace.
        """
        root = Path(config["workspace_root"]).resolve()  # type: ignore[index]
        return cls.from_path(root)

    @classmethod
    def load(cls, root: Path | str) -> Workspace:
        """Load an existing workspace from disk."""
        root = Path(root).resolve()
        metadata_file = root / "workspace.json"
        if not metadata_file.exists():
            raise FileNotFoundError(
                f"Workspace metadata not found at {metadata_file}"
            )
        meta = _load_metadata(WorkspaceMetadata, metadata_file)
        return _reconstruct(cls, {"root": root, "metadata": meta, "_assets_lib": None})

    @classmethod
    def from_path(cls, root: Path | str) -> Workspace:
        """Load existing workspace, or create & materialize a new one."""
        root = Path(root).resolve()
        if (root / "workspace.json").exists():
            return cls.load(root)
        ws = cls(root=root, name=root.name)
        ws.materialize()
        return ws

    # ── Project operations ──────────────────────────────────────────────

    def create_project(self, name: str, exist_ok: bool = False) -> Project:
        """Create a new project (materialized immediately).

        Raises:
            ValueError: If project already exists and *exist_ok* is False.
        """
        project = Project(name=name, workspace=self)
        project_dir = self.root / "projects" / project.id
        if project_dir.exists():
            if exist_ok:
                return self._load_project_from_dir(project_dir)
            raise ValueError(f"Project '{project.id}' already exists")
        project.materialize()
        return project

    def get_project(self, name_or_id: str) -> Project | None:
        """Get project by name or slugified ID."""
        for candidate in (name_or_id, slugify(name_or_id)):
            project_dir = self.root / "projects" / candidate
            if project_dir.exists():
                return self._load_project_from_dir(project_dir)
        return None

    def list_projects(self) -> list[Project]:
        """List all projects by scanning the ``projects/`` directory."""
        return _list_children(
            children_dir=self.root / "projects",
            metadata_filename="project.json",
            metadata_cls=ProjectMetadata,
            child_cls=Project,
            attrs_factory=lambda m: {
                "workspace": self,
                "metadata": m,
                "_assets_lib": None,
            },
        )

    def delete_project(self, project_id: str) -> None:
        """Delete project directory.

        Raises:
            KeyError: If project not found.
        """
        project_dir = self.root / "projects" / project_id
        if not project_dir.exists():
            raise KeyError(f"Project '{project_id}' not found")
        shutil.rmtree(project_dir)

    # ── Internal ────────────────────────────────────────────────────────

    def _load_project_from_dir(self, project_dir: Path) -> Project:
        meta = _load_metadata(ProjectMetadata, project_dir / "project.json")
        return _reconstruct(
            Project,
            {"workspace": self, "metadata": meta, "_assets_lib": None},
        )
