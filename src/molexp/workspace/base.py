"""Base class for workspace hierarchy nodes.

Provides shared JSON persistence, metadata loading, child reconstruction,
and child listing for Workspace, Project, Experiment, and Run.

The atomic write helpers ``atomic_write_json`` / ``atomic_write_text``
(and the legacy private alias ``_atomic_write_json``) are re-exported from
the cross-layer primitive :mod:`molexp.atomicio` — they live there so the
OKF ``knowledge`` bottom layer can cite them without importing workspace.
These names remain importable from ``molexp.workspace.base`` (same function
objects) for back-compat; new code may reach for ``molexp.atomicio``
directly.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, TypeVar

from pydantic import BaseModel

from molexp.atomicio import atomic_write_json, atomic_write_text

if TYPE_CHECKING:
    from pathlib import Path

    from .fs import FileSystem

T = TypeVar("T")

# Backwards-compatible private alias — many existing call sites use the
# underscore name. New code should reach for ``atomic_write_json``.
_atomic_write_json = atomic_write_json


def _save_metadata(metadata: BaseModel, path: str | Path, *, fs: FileSystem | None = None) -> None:
    """Write a Pydantic metadata model to a JSON file atomically."""
    from .schema_version import write_versioned_json

    write_versioned_json(path, metadata.model_dump(mode="json"), fs=fs)


def _load_metadata[T](
    metadata_cls: type[T], path: str | Path, *, fs: FileSystem | None = None
) -> T:
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


# Re-export surface (back-compat): the atomic writers live in
# ``molexp.atomicio`` now but stay importable from here as the same objects.
__all__ = ["atomic_write_json", "atomic_write_text"]
