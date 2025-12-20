"""Folder scanning and classification for indexed entities."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from ..repositories.indexed import IndexFileManager

if TYPE_CHECKING:
    from ..models import Asset, Experiment, Project, Run


class FolderScanner:
    """Scans workspace and classifies folders as indexed entities.

    This service provides discovery and classification of molexp-managed
    folders by detecting and parsing their index files.
    """

    def __init__(self, workspace_root: Path) -> None:
        """Initialize scanner with workspace root.

        Args:
            workspace_root: Root directory of the workspace
        """
        self.root = Path(workspace_root)

    def scan_folder(self, folder_path: Path) -> dict[str, Any] | None:
        """Scan a folder and return its entity metadata if it's indexed.

        Args:
            folder_path: Path to the folder to scan

        Returns:
            Dict with entity metadata or None if not an indexed folder.
            The dict contains:
                - kind: Entity kind (project, experiment, run, asset)
                - path: Relative path from workspace root
                - entity: Parsed entity model instance
                - is_indexed: Always True for indexed folders
        """
        # Detect entity kind
        kind = IndexFileManager.detect_entity_kind(folder_path)

        if not kind:
            return None

        # Import models dynamically to avoid circular imports
        from ..models import Asset, Experiment, Project, Run

        # Map kinds to model classes
        model_map = {
            "project": Project,
            "experiment": Experiment,
            "run": Run,
            "asset": Asset,
        }

        model_class = model_map.get(kind)
        if not model_class:
            return None

        # Read and parse index
        entity = IndexFileManager.read_index(folder_path, kind, model_class)

        if not entity:
            return None

        return {
            "kind": kind,
            "path": str(folder_path.relative_to(self.root)),
            "entity": entity,
            "is_indexed": True,
        }

    def scan_workspace(self) -> list[dict[str, Any]]:
        """Scan entire workspace and return all indexed entities.

        Returns:
            List of entity info dicts (see scan_folder for structure)
        """
        indexed_entities = []

        # Scan projects
        projects_dir = self.root / "projects"
        if projects_dir.exists():
            for project_dir in projects_dir.iterdir():
                if project_dir.is_dir():
                    entity_info = self.scan_folder(project_dir)
                    if entity_info:
                        indexed_entities.append(entity_info)

                        # Scan experiments within this project
                        experiments_dir = project_dir / "experiments"
                        if experiments_dir.exists():
                            for exp_dir in experiments_dir.iterdir():
                                if exp_dir.is_dir():
                                    exp_info = self.scan_folder(exp_dir)
                                    if exp_info:
                                        indexed_entities.append(exp_info)

                                        # Scan runs within this experiment
                                        runs_dir = exp_dir / "runs"
                                        if runs_dir.exists():
                                            for run_dir in runs_dir.iterdir():
                                                if run_dir.is_dir():
                                                    run_info = self.scan_folder(run_dir)
                                                    if run_info:
                                                        indexed_entities.append(
                                                            run_info
                                                        )

        # Scan assets
        assets_dir = self.root / "assets"
        if assets_dir.exists():
            for asset_dir in assets_dir.iterdir():
                if asset_dir.is_dir():
                    entity_info = self.scan_folder(asset_dir)
                    if entity_info:
                        indexed_entities.append(entity_info)

        return indexed_entities

    def is_indexed_folder(self, folder_path: Path) -> bool:
        """Check if a folder is an indexed entity.

        Args:
            folder_path: Path to check

        Returns:
            True if folder contains a valid index file
        """
        return self.scan_folder(folder_path) is not None

    def get_entity_kind(self, folder_path: Path) -> str | None:
        """Get the entity kind for a folder.

        Args:
            folder_path: Path to check

        Returns:
            Entity kind string or None if not indexed
        """
        entity_info = self.scan_folder(folder_path)
        return entity_info["kind"] if entity_info else None
