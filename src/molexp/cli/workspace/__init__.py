"""``molexp.cli.workspace`` — workspace command implementations.

Command modules (``run`` / ``serve`` / ``monitor`` / ``explore`` / ``sync`` /
``lifecycle`` / ``resources``) register their commands on the shared top-level
app (:mod:`molexp.cli._app`); the flat tree is assembled in :mod:`molexp.cli`.
This package no longer owns a Typer group or a target-resolving callback —
target resolution moved to :mod:`molexp.cli._target`.
"""

from __future__ import annotations

from molexp.workspace.target import RemoteTarget


class RemoteWorkspaceError(Exception):
    """Full workspace CRUD on remote targets is not yet implemented.

    Use ``exec``, ``shell``, or ``sync`` for remote operations.
    """

    def __init__(self, target: RemoteTarget | None) -> None:
        super().__init__(
            f"Remote workspace CRUD not yet supported for {target}. "
            "Use 'exec', 'shell', or 'sync' commands."
        )
