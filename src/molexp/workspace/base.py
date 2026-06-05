"""Base class for workspace hierarchy nodes.

Provides shared JSON persistence, metadata loading, child reconstruction,
and child listing for Workspace, Project, Experiment, and Run.
"""

from __future__ import annotations

import contextlib
import json
import os
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, TypeVar

from pydantic import BaseModel

if TYPE_CHECKING:
    from .fs import FileSystem

T = TypeVar("T")


def atomic_write_json(path: Path, data: object) -> None:
    """Write JSON data to a file atomically via write-to-temp + rename.

    On POSIX systems, os.replace is atomic — if the process crashes
    mid-write, the original file remains intact. This prevents data
    corruption for critical files like run.json, metadata files, and
    workflow-state checkpoints.

    The ``data`` parameter is the structural top-type ``object`` rather
    than ``JSONValue`` because :func:`json.dumps` is invoked with
    ``default=str``, which accepts anything that has a string repr.
    Callers are responsible for ensuring the value is meaningful as
    JSON; ``json.dumps`` raises at write time if not.

    Public surface — re-exported through ``molexp.workspace`` so the
    workflow layer's ``RunStorePersistence`` and the agent layer's
    :class:`Agent` / :class:`AgentSession` folder subclasses can write
    through workspace's atomicity guarantee without reaching into a
    private helper.

    Args:
        path: Destination file path.
        data: JSON-serializable value (or anything ``str()``-coercible).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    # Write to a temp file in the same directory (same filesystem for atomic rename)
    fd, tmp_path = tempfile.mkstemp(dir=path.parent, suffix=".tmp", prefix=f".{path.stem}_")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(data, f, indent=2, default=str)
        os.replace(tmp_path, path)  # noqa: PTH105
    except BaseException:
        # Clean up temp file on any failure (including KeyboardInterrupt)
        with contextlib.suppress(OSError):
            os.unlink(tmp_path)  # noqa: PTH108
        raise


# Backwards-compatible private alias — every existing call site uses
# the underscore name. New code should reach for ``atomic_write_json``.
_atomic_write_json = atomic_write_json


def atomic_write_text(path: Path, content: str, *, encoding: str = "utf-8") -> None:
    """Write text to a file atomically via write-to-temp + rename.

    Companion to :func:`atomic_write_json` for plain-text artifacts —
    markdown reports, generated source previews, log snapshots — that
    are read back as strings rather than parsed as JSON. Same temp-file
    + ``os.replace`` pattern; if the process crashes mid-write the
    original file remains intact.

    Public surface — re-exported through ``molexp.workspace`` so the
    harness layer (validation reports / audit records) and any future
    plain-text consumer can write through workspace's atomicity
    guarantee.

    Args:
        path: Destination file path.
        content: Text to write.
        encoding: Text encoding (default ``"utf-8"``).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(dir=path.parent, suffix=".tmp", prefix=f".{path.stem}_")
    tmp = Path(tmp_path)
    try:
        with os.fdopen(fd, "w", encoding=encoding) as f:
            f.write(content)
        tmp.replace(path)
    except BaseException:
        # Clean up temp file on any failure (including KeyboardInterrupt).
        with contextlib.suppress(OSError):
            tmp.unlink()
        raise


def _save_metadata(metadata: BaseModel, path: str, *, fs: FileSystem | None = None) -> None:
    """Write a Pydantic metadata model to a JSON file atomically."""
    from .schema_version import write_versioned_json

    write_versioned_json(path, metadata.model_dump(mode="json"), fs=fs)


def _load_metadata[T](metadata_cls: type[T], path: str, *, fs: FileSystem | None = None) -> T:
    """Read a JSON file into a Pydantic metadata model."""
    from .schema_version import read_versioned_json

    data = read_versioned_json(path, fs=fs)
    return metadata_cls(**data)


def _reconstruct[T](
    cls: type[T],
    attrs: dict[str, object],
) -> T:
    """Reconstruct a hierarchy object without calling ``__init__``.

    The attribute values are heterogeneous — pydantic metadata models,
    parent containers (``Workspace`` / ``Project`` / ``Experiment``),
    paths — so the parameter type is the structural top-type ``object``.
    Each call site already knows the concrete shape it is reconstituting.

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


def _list_children[T](
    children_dir: Path,
    metadata_filename: str,
    metadata_cls: type[BaseModel],
    child_cls: type[T],
    attrs_factory: Callable[[BaseModel], dict[str, object]],
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
