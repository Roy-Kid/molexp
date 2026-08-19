"""Workspace JSON schema versioning."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .fs import FileSystem

MOLEXP_SCHEMA_VERSION = 1


def versioned_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Wrap *payload* with the ``schema_version`` envelope FileStore writes."""
    return {"schema_version": MOLEXP_SCHEMA_VERSION, **payload}


class IncompatibleSchemaError(RuntimeError):
    """Raised when a JSON file's schema_version exceeds what this build understands."""


def write_versioned_json(
    path: str | Path, payload: dict[str, Any], *, fs: FileSystem | None = None
) -> None:
    """Atomically write *payload* with a ``schema_version`` envelope."""
    versioned = {"schema_version": MOLEXP_SCHEMA_VERSION, **payload}
    if fs is not None:
        fs.atomic_write_json(str(path), versioned)
    else:
        from .base import atomic_write_json

        atomic_write_json(Path(path), versioned)


def read_versioned_json(path: str | Path, *, fs: FileSystem | None = None) -> dict[str, Any]:
    """Read a JSON file and return the payload with ``schema_version`` stripped."""
    if fs is not None:
        with fs.open(str(path)) as fh:
            data = json.load(fh)
    else:
        with open(path) as fh:  # noqa: PTH123
            data = json.load(fh)
    if "schema_version" not in data:
        raise IncompatibleSchemaError(
            f"{path} is missing schema_version; this molexp requires "
            f"schema_version={MOLEXP_SCHEMA_VERSION}."
        )
    sv = data.pop("schema_version")
    if isinstance(sv, int) and sv > MOLEXP_SCHEMA_VERSION:
        raise IncompatibleSchemaError(
            f"{path} has schema_version={sv}; this molexp supports up to "
            f"schema_version={MOLEXP_SCHEMA_VERSION}."
        )
    return data
