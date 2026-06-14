"""Request-aware workspace resolution (FastAPI dependencies).

Resolves the workspace a request addresses — explicit ``{ws}`` key, query
param, header, or the active/default workspace — caching instances in the
``(kind, identifier)``-keyed cache owned by :mod:`.workspace_state`.
"""

from __future__ import annotations

from pathlib import Path

from fastapi import Request

from molexp.server.deps.served import _served_by_key
from molexp.server.deps.targets import get_workspace_target_registry
from molexp.server.deps.workspace_state import (
    _SAFE_METHODS,
    _active_workspace_key,
    _workspace_cache,
)
from molexp.workspace import Workspace


def _workspace_key_from_request(request: Request | None) -> str | None:
    """Extract an explicit workspace key from a request, if any.

    Resolution order: ``{ws}`` path segment (the aggregate routes) →
    ``?ws=`` query param → ``X-Molexp-Workspace`` header. Returns ``None``
    when the request addresses the active/default workspace (flat routes).
    """
    if request is None:
        return None
    return (
        request.path_params.get("ws")
        or request.query_params.get("ws")
        or request.headers.get("x-molexp-workspace")
        or None
    )


def get_workspace(request: Request):  # noqa: ANN201
    """FastAPI dependency to get a cached Workspace instance.

    When the request carries an explicit workspace key (``{ws}`` path segment,
    ``?ws=``, or ``X-Molexp-Workspace`` header) the named served workspace is
    resolved via :func:`get_workspace_by_key`. Otherwise the **active/default**
    workspace is used — the unchanged single-workspace path.

    Mutating requests (non-GET) against a remote workspace are rejected with
    :class:`RemoteWorkspaceReadOnlyError`.
    """
    explicit_key = _workspace_key_from_request(request)
    if explicit_key is not None:
        if request.method not in _SAFE_METHODS:
            sw = _served_by_key(explicit_key)
            if sw is not None and sw.is_remote:
                from molexp.server.exceptions import RemoteWorkspaceReadOnlyError

                raise RemoteWorkspaceReadOnlyError(explicit_key)
        return get_workspace_by_key(explicit_key)

    # No explicit workspace key → the active/default workspace, with its
    # existing semantics unchanged. The remote read-only policy applies to the
    # aggregate surface only (explicit key / `/workspaces/{ws}`), so the legacy
    # active-switch surface (e.g. cache invalidation on a switched-to remote)
    # keeps working.
    return get_active_workspace()


def get_active_workspace():  # noqa: ANN201
    """Resolve (and cache) the active/default workspace, ignoring any request.

    The cache key is ``(kind, identifier)`` — local-vs-remote workspaces
    coexist without collision. This is the request-free core used by
    :func:`get_workspace` and by direct callers that have no request.
    """
    kind, identifier = _active_workspace_key()
    cache_key = (kind, identifier)
    if cache_key not in _workspace_cache:
        if kind == "remote":
            from molexp.server.workspace_targets import (
                target_to_filesystem_for_workspace_target,
            )

            registry = get_workspace_target_registry()
            try:
                target = registry.get(identifier)
            except KeyError as exc:
                raise KeyError(
                    f"active workspace target {identifier!r} no longer registered"
                ) from exc
            fs = target_to_filesystem_for_workspace_target(target)
            _workspace_cache[cache_key] = Workspace(target.root_path, fs=fs)
        else:
            _workspace_cache[cache_key] = Workspace(Path(identifier))
    return _workspace_cache[cache_key]


def get_workspace_by_key(key: str):  # noqa: ANN201
    """Resolve a served workspace by its stable ``key`` (the ``{ws}`` segment).

    Raises:
        UnknownWorkspaceError: ``key`` names no served workspace (404).
        RemoteWorkspaceUnreachableError: the remote transport failed (502).
    """
    sw = _served_by_key(key)
    if sw is None:
        from molexp.server.exceptions import UnknownWorkspaceError

        raise UnknownWorkspaceError(key)

    if sw.is_remote:
        target_name = sw.target_name or sw.key
        cache_key = ("remote", target_name)
        if cache_key not in _workspace_cache:
            from molexp.server.exceptions import RemoteWorkspaceUnreachableError
            from molexp.server.workspace_targets import (
                target_to_filesystem_for_workspace_target,
            )

            try:
                target = get_workspace_target_registry().get(target_name)
                fs = target_to_filesystem_for_workspace_target(target)
                _workspace_cache[cache_key] = Workspace(target.root_path, fs=fs)
            except Exception as exc:  # connection / auth / unknown target
                raise RemoteWorkspaceUnreachableError(sw.key, str(exc)) from exc
        return _workspace_cache[cache_key]

    assert sw.path is not None  # local always carries a path
    cache_key = ("local", str(Path(sw.path).resolve()))
    if cache_key not in _workspace_cache:
        _workspace_cache[cache_key] = Workspace(Path(sw.path))
    return _workspace_cache[cache_key]
