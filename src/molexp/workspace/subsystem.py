"""Per-subsystem private storage under ``<workspace_root>/.subsystems/<kind>/``.

A subsystem is any consumer of the workspace that needs a private,
namespaced location for its own files. Workspace knows nothing about
specific consumers — it just vends the path through
:meth:`Workspace.subsystem_store` and validates that the requested
*kind* is shape-safe. The consumer owns the layout inside the
directory it is handed and the schema of the files it writes.

Construction is side-effect-free: ``SubsystemStore(root, kind)`` does
not create any directory. ``.dir()`` and ``.file(name)`` create the
parent on demand. ``kind`` is dotted-lowercase ASCII; validation
rejects path-traversal, leading dots, uppercase, and non-ASCII strings
so the namespace can never escape ``<workspace_root>/.subsystems/`` or
shadow another dotfile. Workspace assigns no semantics to particular
``kind`` values.
"""

from __future__ import annotations

from pathlib import Path

# Stable kind grammar (lowercase ASCII, dot-separated, no path
# traversal) is owned by ``molexp.workspace.folder`` since the
# unify-folder-abstraction-01 sub-spec — subsystem reverse-imports the
# canonical ``_KIND_PATTERN`` and ``_validate_kind`` so behaviour stays
# identical until sub-spec 03 deletes this module. The redundant
# ``as`` aliases mark the re-export intent so ruff F401 stays clean.
from .folder import _KIND_PATTERN as _KIND_PATTERN
from .folder import _validate_kind as _validate_kind

SUBSYSTEMS_DIRNAME = ".subsystems"


class SubsystemStore:
    """A per-kind private directory rooted at ``<root>/.subsystems/<kind>/``.

    The store does not interpret the contents of its directory — it
    only vendors paths and creates parent directories on demand.
    Consumers own whatever schema lives inside.

    Examples:
        >>> store = SubsystemStore(workspace_root, "my.subsystem")
        >>> store_dir = store.dir()
        >>> store_file = SubsystemStore(workspace_root, "my.subsystem").file("data.json")
    """

    def __init__(self, workspace_root: Path | str, kind: str) -> None:
        _validate_kind(kind)
        self._root = Path(workspace_root)
        self._kind = kind

    @property
    def kind(self) -> str:
        return self._kind

    @property
    def workspace_root(self) -> Path:
        return self._root

    def dir(self) -> Path:
        """Return the kind's directory, creating it if missing."""
        path = self._root / SUBSYSTEMS_DIRNAME / self._kind
        path.mkdir(parents=True, exist_ok=True)
        return path

    def file(self, name: str) -> Path:
        """Return ``<root>/.subsystems/<kind>/<name>``, mkdir-ing the parent.

        The file itself is *not* created; only the parent kind directory.
        Caller writes the file content (typically through
        :func:`~molexp.workspace.base._atomic_write_json`).
        """
        if not isinstance(name, str) or not name:
            raise ValueError("subsystem file name must be a non-empty string")
        if "/" in name or "\\" in name or name in {".", ".."}:
            raise ValueError(f"invalid subsystem file name {name!r}")
        return self.dir() / name


__all__ = ["SUBSYSTEMS_DIRNAME", "SubsystemStore"]
