"""Per-subsystem private storage under ``<workspace_root>/.subsystems/<kind>/``.

A subsystem is any consumer of the workspace that needs a private,
namespaced location for its own files — for example the agent layer's
sessions / skills / tools / mcp stores. The workspace vendors the path
through :meth:`Workspace.subsystem_store`; the subsystem owns the layout
inside the directory it is handed.

Construction is side-effect-free: ``SubsystemStore(root, kind)`` does
not create any directory. ``.dir()`` and ``.file(name)`` create the
parent on demand. ``kind`` is dotted-lowercase (``agent.sessions``,
``agent.skills``); validation rejects path-traversal, leading dots,
uppercase, and non-ASCII strings so the namespace can never escape
``<workspace_root>/.subsystems/`` or shadow another dotfile.
"""

from __future__ import annotations

import re
from pathlib import Path

SUBSYSTEMS_DIRNAME = ".subsystems"

# Stable kind keys: lowercase ASCII, segments separated by ``.``,
# segment characters ``[a-z0-9_-]``, no leading dot, no path traversal.
_KIND_PATTERN = re.compile(r"^[a-z0-9][a-z0-9_-]*(?:\.[a-z0-9][a-z0-9_-]*)*$")


def _validate_kind(kind: str) -> None:
    if not isinstance(kind, str) or not kind:
        raise ValueError("subsystem kind must be a non-empty string")
    if not _KIND_PATTERN.fullmatch(kind):
        raise ValueError(
            f"invalid subsystem kind {kind!r}: must be dotted lowercase ASCII "
            "(e.g. 'agent.sessions'); no path separators, leading dots, "
            "uppercase, or whitespace allowed"
        )


class SubsystemStore:
    """A per-kind private directory rooted at ``<root>/.subsystems/<kind>/``.

    The store does not interpret the contents of its directory — it
    only vendors paths and creates parent directories on demand.
    Consumers (the agent layer's sessions store, skills store, etc.)
    own whatever schema lives inside.

    Examples:
        >>> store = SubsystemStore(workspace_root, "agent.sessions")
        >>> sessions_dir = store.dir()
        >>> skills_file = SubsystemStore(workspace_root, "agent.skills").file("skills.json")
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
