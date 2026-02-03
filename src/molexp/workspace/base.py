"""Base class for workspace hierarchy nodes.

Provides shared JSON persistence, metadata loading, child reconstruction,
and child listing for Workspace, Project, Experiment, and Run.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, TypeVar

from pydantic import BaseModel

T = TypeVar("T")


def _save_metadata(metadata: BaseModel, path: Path) -> None:
    """Write a Pydantic metadata model to a JSON file.

    Args:
        metadata: Pydantic model to serialize.
        path: Destination file path.
    """
    with open(path, "w") as f:
        json.dump(metadata.model_dump(mode="json"), f, indent=2, default=str)


def _load_metadata(metadata_cls: type[T], path: Path) -> T:
    """Read a JSON file into a Pydantic metadata model.

    Args:
        metadata_cls: Target Pydantic model class.
        path: Source JSON file path.

    Returns:
        Deserialized metadata model instance.
    """
    with open(path, "r") as f:
        data = json.load(f)
    return metadata_cls(**data)


def _reconstruct(
    cls: type[T],
    attrs: dict[str, Any],
) -> T:
    """Reconstruct a hierarchy object without calling ``__init__``.

    Args:
        cls: Target class.
        attrs: Mapping of attribute names to values to set on the instance.

    Returns:
        Reconstructed instance with attributes set.
    """
    obj = cls.__new__(cls)
    for key, value in attrs.items():
        setattr(obj, key, value)
    return obj


def _list_children(
    children_dir: Path,
    metadata_filename: str,
    metadata_cls: type[BaseModel],
    child_cls: type[T],
    attrs_factory: Any,
) -> list[T]:
    """List child nodes by scanning a directory for metadata files.

    Args:
        children_dir: Directory containing child subdirectories.
        metadata_filename: Name of the metadata JSON file (e.g. "project.json").
        metadata_cls: Pydantic model class for deserialization.
        child_cls: Class of the child to reconstruct.
        attrs_factory: Callable(metadata) -> dict of attrs to pass to _reconstruct.

    Returns:
        List of reconstructed child instances.
    """
    if not children_dir.exists():
        return []

    children: list[T] = []
    for child_dir in children_dir.iterdir():
        if child_dir.is_dir():
            metadata_file = child_dir / metadata_filename
            if metadata_file.exists():
                metadata = _load_metadata(metadata_cls, metadata_file)
                attrs = attrs_factory(metadata)
                child = _reconstruct(child_cls, attrs)
                children.append(child)
    return children
