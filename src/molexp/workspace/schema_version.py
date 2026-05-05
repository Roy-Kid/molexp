"""Workspace JSON schema versioning.

Every entity-level JSON file written by the workspace layer
(``workspace.json``, ``project.json``, ``experiment.json``,
``run.json``, ``executions/<id>/execution.json``, and the per-container
index files like ``runs.json``) is wrapped with a ``schema_version``
integer at the top level.

Compatibility rules on load:

* **missing** ``schema_version`` → treated as ``v0`` (legacy, predating
  this module). The payload is accepted as-is; the next save bumps it
  to :data:`MOLEXP_SCHEMA_VERSION`.
* **==** :data:`MOLEXP_SCHEMA_VERSION` → loaded as-is.
* **>** :data:`MOLEXP_SCHEMA_VERSION` → :class:`IncompatibleSchemaError`
  (a future molexp wrote this; the current build cannot interpret it).

The asset manifest (``assets/manifest.json``) and global catalog
(``.catalog/index.json``) carry their own ``schema_version`` constants
that predate this module — they are not routed through here.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

MOLEXP_SCHEMA_VERSION = 1


class IncompatibleSchemaError(RuntimeError):
    """Raised when a JSON file's ``schema_version`` exceeds what this
    molexp build understands.

    The forward-compatibility contract is one-way: a newer molexp can
    read older files (auto-upgrade), but an older molexp must refuse to
    interpret a newer payload rather than silently lose fields.
    """


def write_versioned_json(path: Path, payload: dict[str, Any]) -> None:
    """Atomically write *payload* with a ``schema_version`` envelope.

    The current :data:`MOLEXP_SCHEMA_VERSION` is injected at the top
    level; any incoming ``schema_version`` key in *payload* is
    overwritten so callers cannot accidentally write a stale value.

    Args:
        path: Destination JSON file.
        payload: JSON-serializable mapping (typically a Pydantic
            ``model_dump(mode="json")`` output).
    """
    from .base import _atomic_write_json

    versioned = {"schema_version": MOLEXP_SCHEMA_VERSION, **payload}
    versioned["schema_version"] = MOLEXP_SCHEMA_VERSION  # ensure ours wins
    _atomic_write_json(path, versioned)


def read_versioned_json(path: Path) -> dict[str, Any]:
    """Read a JSON file and return the payload with ``schema_version`` stripped.

    Args:
        path: JSON file to read.

    Returns:
        The decoded payload as a mapping, with the ``schema_version``
        key removed (callers consume the model fields only).

    Raises:
        IncompatibleSchemaError: When ``schema_version`` exceeds
            :data:`MOLEXP_SCHEMA_VERSION`.
    """
    with open(path) as fh:
        data = json.load(fh)
    sv = data.pop("schema_version", 0)
    if isinstance(sv, int) and sv > MOLEXP_SCHEMA_VERSION:
        raise IncompatibleSchemaError(
            f"{path} has schema_version={sv}; this molexp supports up to "
            f"schema_version={MOLEXP_SCHEMA_VERSION}. Upgrade molexp or "
            f"downgrade the workspace."
        )
    return data
