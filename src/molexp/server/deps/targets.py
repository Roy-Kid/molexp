"""Workspace-target registry singleton (server-process scope).

This module is the **single owner** of the mutable
``_workspace_target_registry`` global; tests that need to substitute a
registry should monkeypatch *this* module (or override the FastAPI
dependency), not the ``molexp.server.dependencies`` facade.
"""

from __future__ import annotations

_workspace_target_registry: object | None = None  # lazy singleton; typed via accessor


def get_workspace_target_registry():  # noqa: ANN201 — return type stated in docstring
    """Return the process-singleton :class:`WorkspaceTargetRegistry`.

    The registry is server-process scope (lives at
    ``~/.molexp/workspace_targets.json`` in production) so it must
    exist *before* any workspace is open.  Tests should override this
    dependency via ``app.dependency_overrides`` rather than touching
    the singleton.
    """
    global _workspace_target_registry
    from molexp.server.workspace_targets import WorkspaceTargetRegistry

    if _workspace_target_registry is None:
        _workspace_target_registry = WorkspaceTargetRegistry()
    return _workspace_target_registry


def reset_workspace_target_registry() -> None:
    """Drop the process singleton (test fixture support)."""
    global _workspace_target_registry
    _workspace_target_registry = None


def get_remote_fs_factory():  # noqa: ANN201
    """Return the FS-factory callable used by the workspace-targets probe.

    Separated as a FastAPI dependency so tests can substitute a fake
    factory via ``app.dependency_overrides[get_remote_fs_factory]``.
    """
    from molexp.server.workspace_targets import (
        target_to_filesystem_for_workspace_target,
    )

    return target_to_filesystem_for_workspace_target
