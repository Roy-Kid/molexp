"""Workspace folder store (in-memory process singleton).

This module is the **single owner** of the mutable ``_workspace_folder_store``
global; writes go through :func:`get_workspace_folder_store` /
:func:`reset_workspace_folder_store` only.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


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
