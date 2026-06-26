"""The built-in ``local`` compute target — always available.

A run can always be started on *this machine* without first registering a
target: the server offers a default ``local`` target (``host=None`` →
``LocalTransport``, ``scheduler="local"``) on top of whatever the workspace
registry holds. The registry stays the source of truth — this is purely a
server-side convenience (the CLI already covers local execution via
``molexp run``). A workspace that registers its own ``local`` target overrides
the built-in.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from molexp.workspace import ComputeTarget, get_target, list_targets

if TYPE_CHECKING:
    from molexp.workspace import Workspace

__all__ = [
    "LOCAL_TARGET_NAME",
    "builtin_local_target",
    "effective_targets",
    "resolve_target",
]

LOCAL_TARGET_NAME = "local"


def builtin_local_target(workspace: Workspace) -> ComputeTarget:
    """The default local target: this machine, molq ``local`` scheduler."""
    return ComputeTarget(name=LOCAL_TARGET_NAME, scratch_root=str(workspace.root))


def effective_targets(workspace: Workspace) -> list[ComputeTarget]:
    """Registered targets, with the built-in ``local`` prepended unless the
    workspace already registers a target named ``local``."""
    registered = list_targets(workspace)
    if any(t.name == LOCAL_TARGET_NAME for t in registered):
        return list(registered)
    return [builtin_local_target(workspace), *registered]


def resolve_target(workspace: Workspace, name: str) -> ComputeTarget:
    """``get_target`` with the built-in ``local`` fallback.

    Raises ``KeyError`` for any unregistered name other than ``local``.
    """
    try:
        return get_target(workspace, name)
    except KeyError:
        if name == LOCAL_TARGET_NAME:
            return builtin_local_target(workspace)
        raise
