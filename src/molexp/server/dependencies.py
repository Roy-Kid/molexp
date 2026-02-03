"""Dependency injection for MolExp API.

This module provides FastAPI dependencies for:
- Workspace instances
- WorkspaceFolderStore
- Configuration settings

Using dependency injection instead of global variables improves:
- Testability (easy to mock dependencies)
- Separation of concerns
- Configuration management
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from molexp.workspace import Workspace


class Settings(BaseModel):
    """Application settings loaded from environment."""

    workspace_path: str = ""
    debug: bool = False

    @classmethod
    def from_env(cls) -> "Settings":
        """Create settings from environment variables."""
        return cls(
            workspace_path=os.environ.get("MOLEXP_WORKSPACE", ""),
            debug=os.environ.get("MOLEXP_DEBUG", "").lower() in ("true", "1", "yes"),
        )

    def get_workspace_path(self) -> Path:
        """Get workspace path, defaulting to current directory."""
        if self.workspace_path:
            return Path(self.workspace_path)
        return Path.cwd()


@lru_cache
def get_settings() -> Settings:
    """Get cached application settings."""
    return Settings.from_env()


# Import Workspace lazily to avoid circular imports
def get_workspace():
    """FastAPI dependency to get Workspace instance.

    Usage:
        @app.get("/api/projects")
        def list_projects(workspace: Workspace = Depends(get_workspace)):
            ...
    """
    from molexp.workspace import Workspace

    workspace_path = get_workspace_path()
    return Workspace.from_path(workspace_path)


# ============================================================================
# Workspace Folder Store (in-memory singleton)
# ============================================================================


class WorkspaceFolderStore:
    """In-memory storage for workspace folders.

    Thread-safe storage for tracking added workspace folders.
    Each folder has an ID, path, name, and timestamp.
    """

    def __init__(self) -> None:
        self._folders: dict[str, dict[str, Any]] = {}

    def add(self, folder_id: str, path: str, name: str, added_at: str) -> None:
        """Add a folder to the store."""
        self._folders[folder_id] = {
            "id": folder_id,
            "path": path,
            "name": name,
            "added_at": added_at,
        }

    def remove(self, folder_id: str) -> bool:
        """Remove a folder from the store.

        Returns:
            True if folder was removed, False if not found
        """
        if folder_id in self._folders:
            del self._folders[folder_id]
            return True
        return False

    def get(self, folder_id: str) -> dict[str, Any] | None:
        """Get a folder by ID."""
        return self._folders.get(folder_id)

    def list_all(self) -> list[dict[str, Any]]:
        """List all folders."""
        return list(self._folders.values())

    def find_by_path(self, path: Path) -> dict[str, Any] | None:
        """Find a folder by its path."""
        resolved = path.resolve()
        for folder in self._folders.values():
            if Path(folder["path"]).resolve() == resolved:
                return folder
        return None


# Global singleton instance
_workspace_folder_store: WorkspaceFolderStore | None = None
_workspace_path_override: Path | None = None


def get_workspace_folder_store() -> WorkspaceFolderStore:
    """FastAPI dependency to get WorkspaceFolderStore instance.

    Usage:
        @app.post("/api/workspace/folders")
        def add_folder(store: WorkspaceFolderStore = Depends(get_workspace_folder_store)):
            ...
    """
    global _workspace_folder_store
    if _workspace_folder_store is None:
        _workspace_folder_store = WorkspaceFolderStore()
    return _workspace_folder_store


def reset_workspace_folder_store() -> None:
    """Reset the workspace folder store (for testing)."""
    global _workspace_folder_store
    _workspace_folder_store = None


def set_workspace_path_override(path: Path | None) -> None:
    """Override workspace path at runtime."""
    global _workspace_path_override
    _workspace_path_override = path


def get_workspace_path() -> Path:
    """Resolve workspace path, honoring override when set."""
    if _workspace_path_override is not None:
        return _workspace_path_override
    settings = get_settings()
    return settings.get_workspace_path()
