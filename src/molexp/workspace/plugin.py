"""Workspace host plugin — publishes ``ctx.workspace`` (and ``ctx.fs``).

This module is the only workspace→host edge. It does not import
``molexp.harness``; compose mounts it by duck-typed ``name`` / ``inject`` /
``apply``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

__all__ = ["WorkspacePlugin"]


class WorkspacePlugin:
    """Publish a :class:`~molexp.workspace.Workspace` on the host context."""

    name = "workspace"
    inject: tuple[str, ...] = ()

    def __init__(self, root: str | Path) -> None:
        self._root = Path(root)

    def apply(self, ctx: Any) -> None:  # noqa: ANN401 — duck-typed host Context
        """Provide ``workspace`` and, if free, ``fs``."""
        from molexp.workspace.workspace import Workspace

        workspace = Workspace(self._root)
        ctx.provide("workspace", workspace)
        if not ctx.has("fs"):
            ctx.provide("fs", workspace.fs)
