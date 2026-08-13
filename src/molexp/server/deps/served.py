"""Served-workspace set (the workspaces ``molexp serve`` was pointed at).

The active workspace (overrides in :mod:`.workspace_state`) is whichever one
is *currently* being read; the served set is every workspace the server was
started with, so the UI can list them and switch (via
``POST /api/workspace/open``) between them.  Exactly one served workspace is
the unchanged single-workspace case.

This module is the **single owner** of the mutable ``_served_workspaces``
global; writes go through :func:`set_served_workspaces` only.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from molexp.server.deps.workspace_state import _SAFE_METHODS, _active_workspace_key

if TYPE_CHECKING:
    from molexp.server.workspace_targets import WorkspaceTarget


@dataclass(frozen=True)
class ServedWorkspace:
    """One workspace `molexp serve` is hosting.

    Attributes:
        key: Stable slug, unique within this server process — the switch handle.
        label: Human-facing description (a path, or ``user@host:/path``).
        is_remote: True for an SSH-backed remote workspace.
        path: Absolute local root, or ``None`` when remote.
        target_name: Workspace-target name used as the remote active-workspace
            identifier (registry name or process-scoped slug from CLI SCP).
        remote_target: Optional inline :class:`WorkspaceTarget` for CLI
            ``-ws user@host:/path`` so serve does not have to persist into
            ``~/.molexp/workspace_targets.json``. When set, resolution prefers
            this over the on-disk registry.
    """

    key: str
    label: str
    is_remote: bool
    path: str | None = None
    target_name: str | None = None
    remote_target: WorkspaceTarget | None = None


_served_workspaces: list[ServedWorkspace] = []


def set_served_workspaces(workspaces: list[ServedWorkspace]) -> None:
    """Record the workspaces this server is hosting (called once by ``serve``)."""
    global _served_workspaces
    _served_workspaces = list(workspaces)


def get_served_workspaces() -> list[ServedWorkspace]:
    """Return the served workspace set (empty until ``serve`` populates it)."""
    return list(_served_workspaces)


def _used_keys() -> set[str]:
    return {sw.key for sw in _served_workspaces}


def add_served_workspace(workspace: ServedWorkspace) -> ServedWorkspace:
    """Append *workspace* to the live served set (VS Code "Add Folder").

    Dedupes by absolute local path or remote label. Returns the existing
    entry when already served.
    """
    global _served_workspaces
    for sw in _served_workspaces:
        if (
            not workspace.is_remote
            and not sw.is_remote
            and sw.path
            and workspace.path
            and Path(sw.path).resolve() == Path(workspace.path).resolve()
        ):
            return sw
        if workspace.is_remote and sw.is_remote and sw.label == workspace.label:
            return sw
    _served_workspaces = [*_served_workspaces, workspace]
    return workspace


def remove_served_workspace(key: str) -> bool:
    """Drop a served workspace by key. Returns False if not found."""
    global _served_workspaces
    before = len(_served_workspaces)
    _served_workspaces = [sw for sw in _served_workspaces if sw.key != key]
    return len(_served_workspaces) < before


def _served_by_key(key: str) -> ServedWorkspace | None:
    """Look up a served workspace by its stable key (``None`` if absent)."""
    for sw in _served_workspaces:
        if sw.key == key:
            return sw
    return None


def resolve_served_remote_target(identifier: str) -> WorkspaceTarget:
    """Resolve a remote :class:`WorkspaceTarget` by active-workspace id.

    Prefers an inline descriptor on the served set (CLI ``-ws user@host:/path``)
    so serve need not write the on-disk registry. Accepts the served ``key``,
    ``target_name``, or the target's own ``name`` (UI activate may send any of
    these). Falls back to the process :class:`WorkspaceTargetRegistry`.

    Raises:
        KeyError: no served inline match and the registry has no such name.
    """
    from molexp.server.workspace_targets import WorkspaceTarget as _WT

    for sw in _served_workspaces:
        if not sw.is_remote or sw.remote_target is None:
            continue
        aliases = {sw.key, sw.target_name, sw.remote_target.name}
        aliases.discard(None)
        if identifier in aliases:
            return sw.remote_target
    from molexp.server.deps.targets import get_workspace_target_registry

    target: _WT = get_workspace_target_registry().get(identifier)
    return target


def active_served_key() -> str | None:
    """The key of the served workspace that is currently active, if any.

    Matches the active ``(kind, identifier)`` against the served set so the UI
    can mark which workspace its flat routes / deep tree currently address.
    """
    kind, identifier = _active_workspace_key()
    for sw in _served_workspaces:
        if sw.is_remote and kind == "remote":
            aliases = {sw.key, sw.target_name}
            if sw.remote_target is not None:
                aliases.add(sw.remote_target.name)
            aliases.discard(None)
            if identifier in aliases:
                return sw.key
        if (
            not sw.is_remote
            and kind == "local"
            and sw.path is not None
            and str(Path(sw.path).resolve()) == identifier
        ):
            return sw.key
    return None


def assert_served_workspace(key: str) -> None:
    """Raise :class:`UnknownWorkspaceError` (404) unless ``key`` is served."""
    if _served_by_key(key) is None:
        from molexp.server.exceptions import UnknownWorkspaceError

        raise UnknownWorkspaceError(key)


def assert_workspace_writable(key: str, method: str) -> None:
    """Reject a mutating ``method`` against a remote workspace (read-only v1).

    Raises :class:`RemoteWorkspaceReadOnlyError` (405). Safe methods and local
    workspaces always pass.
    """
    if method in _SAFE_METHODS:
        return
    sw = _served_by_key(key)
    if sw is not None and sw.is_remote:
        from molexp.server.exceptions import RemoteWorkspaceReadOnlyError

        raise RemoteWorkspaceReadOnlyError(key)
