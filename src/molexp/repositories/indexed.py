"""Index file management for indexed folder entities."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, TypeVar

import yaml
from pydantic import BaseModel

if TYPE_CHECKING:
    from ..models import Asset, Experiment, Project, Run

T = TypeVar("T", bound=BaseModel)


class IndexFileManager:
    """Manages reading and writing index files for indexed folders.
    
    This service provides a centralized way to serialize and deserialize
    entity metadata to/from index files, following molexp conventions.
    """
    
    # Map entity kinds to their models and file patterns
    ENTITY_CONFIG = {
        "project": {
            "filename": "project.yaml",
            "format": "yaml",
        },
        "experiment": {
            "filename": "experiment.yaml",
            "format": "yaml",
        },
        "run": {
            "filename": "run.json",
            "format": "json",
        },
        "asset": {
            "filename": "meta.yaml",
            "format": "yaml",
        },
    }
    
    @classmethod
    def get_index_path(cls, folder_path: Path, kind: str) -> Path:
        """Get the index file path for a given folder and entity kind.
        
        Args:
            folder_path: Path to the entity folder
            kind: Entity kind (project, experiment, run, asset)
            
        Returns:
            Path to the index file
            
        Raises:
            ValueError: If kind is unknown
        """
        config = cls.ENTITY_CONFIG.get(kind)
        if not config:
            raise ValueError(f"Unknown entity kind: {kind}")
        return folder_path / config["filename"]
    
    @classmethod
    def read_index(cls, folder_path: Path, kind: str, model_class: type[T]) -> T | None:
        """Read and parse an index file.
        
        Args:
            folder_path: Path to the entity folder
            kind: Entity kind (project, experiment, run, asset)
            model_class: Pydantic model class to validate against
            
        Returns:
            Parsed entity model or None if file doesn't exist or is invalid
        """
        index_path = cls.get_index_path(folder_path, kind)
        
        if not index_path.exists():
            return None
        
        config = cls.ENTITY_CONFIG[kind]
        
        try:
            with open(index_path, "r") as f:
                if config["format"] == "yaml":
                    data = yaml.safe_load(f)
                else:  # json
                    data = json.load(f)
            
            return model_class.model_validate(data)
        except Exception as e:
            # Log error but don't crash
            logger = logging.getLogger(__name__)
            logger.error(f"Error reading index file {index_path}: {e}")
            return None
    
    @classmethod
    def write_index(cls, folder_path: Path, entity: BaseModel) -> None:
        """Write an entity model to its index file.
        
        Args:
            folder_path: Path to the entity folder
            entity: Entity model instance with a 'kind' property
        """
        kind = entity.kind  # type: ignore
        config = cls.ENTITY_CONFIG[kind]
        index_path = cls.get_index_path(folder_path, kind)
        
        # Ensure folder exists
        folder_path.mkdir(parents=True, exist_ok=True)
        
        # Serialize model
        data = entity.model_dump(mode="json")
        
        with open(index_path, "w") as f:
            if config["format"] == "yaml":
                yaml.safe_dump(data, f, sort_keys=False)
            else:  # json
                json.dump(data, f, indent=2)
    
    @classmethod
    def detect_entity_kind(cls, folder_path: Path) -> str | None:
        """Detect what kind of entity a folder represents by checking for index files.
        
        Args:
            folder_path: Path to check
            
        Returns:
            Entity kind string or None if no index file found
        """
        for kind, config in cls.ENTITY_CONFIG.items():
            index_path = folder_path / config["filename"]
            if index_path.exists():
                return kind
        return None
