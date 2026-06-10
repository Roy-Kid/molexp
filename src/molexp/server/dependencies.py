"""Facade over :mod:`molexp.server.deps` — the historical import surface.

The implementation lives in the ``molexp.server.deps`` package, split by
domain (config, workspace state, served set, resolution, targets, folder
store, agent runtime). This module re-exports the full public surface so
existing ``from molexp.server.dependencies import …`` importers keep working
unchanged.

Singleton semantics: every mutable global lives in exactly **one** owner
module under ``deps/``. Public names are re-bound here (function/class
identity is shared, so FastAPI ``app.dependency_overrides`` keyed on either
import path match), while private module-level globals are *forwarded
dynamically* via module ``__getattr__`` — reading e.g.
``dependencies._workspace_path_override`` always reflects the owner module's
current value. Code that needs to *reassign* a private global (test
monkeypatching) must target the owner module, not this facade.
"""

from __future__ import annotations

from typing import Any

from molexp.server.deps import (
    agent_runtime as _agent_runtime_mod,
)
from molexp.server.deps import (
    config as _config_mod,
)
from molexp.server.deps import (
    folder_store as _folder_store_mod,
)
from molexp.server.deps import (
    resolution as _resolution_mod,
)
from molexp.server.deps import (
    served as _served_mod,
)
from molexp.server.deps import (
    targets as _targets_mod,
)
from molexp.server.deps import (
    workspace_state as _workspace_state_mod,
)
from molexp.server.deps.agent_runtime import get_agent_runtime, reset_agent_runtime
from molexp.server.deps.config import Settings, get_settings
from molexp.server.deps.folder_store import (
    WorkspaceFolderStore,
    get_workspace_folder_store,
    reset_workspace_folder_store,
)
from molexp.server.deps.resolution import (
    get_active_workspace,
    get_workspace,
    get_workspace_by_key,
)
from molexp.server.deps.served import (
    ServedWorkspace,
    active_served_key,
    assert_served_workspace,
    assert_workspace_writable,
    get_served_workspaces,
    set_served_workspaces,
)
from molexp.server.deps.targets import (
    get_remote_fs_factory,
    get_workspace_target_registry,
    reset_workspace_target_registry,
)
from molexp.server.deps.workspace_state import (
    get_workspace_path,
    register_workspace_subscriber,
    reset_workspace_cache,
    set_active_workspace_descriptor,
    set_workspace_path_override,
)

__all__ = [
    "ServedWorkspace",
    "Settings",
    "WorkspaceFolderStore",
    "active_served_key",
    "assert_served_workspace",
    "assert_workspace_writable",
    "get_active_workspace",
    "get_agent_runtime",
    "get_remote_fs_factory",
    "get_served_workspaces",
    "get_settings",
    "get_workspace",
    "get_workspace_by_key",
    "get_workspace_folder_store",
    "get_workspace_path",
    "get_workspace_target_registry",
    "register_workspace_subscriber",
    "reset_agent_runtime",
    "reset_workspace_cache",
    "reset_workspace_folder_store",
    "reset_workspace_target_registry",
    "set_active_workspace_descriptor",
    "set_served_workspaces",
    "set_workspace_path_override",
]

# Private module-level names forwarded to their single owner module. Reads of
# mutable globals (e.g. ``dependencies._workspace_cache``) stay live because
# ``__getattr__`` defers to the owner at access time.
_PRIVATE_NAME_OWNERS = {
    # config
    "_SERVER_DEFAULTS": _config_mod,
    "_load_server_config": _config_mod,
    # workspace_state
    "_SAFE_METHODS": _workspace_state_mod,
    "_workspace_cache": _workspace_state_mod,
    "_workspace_path_override": _workspace_state_mod,
    "_workspace_descriptor_override": _workspace_state_mod,
    "_active_workspace_key": _workspace_state_mod,
    "_active_workspace_subscribers": _workspace_state_mod,
    "_drain_workspace_subscribers": _workspace_state_mod,
    "_await_one": _workspace_state_mod,
    # served
    "_served_workspaces": _served_mod,
    "_served_by_key": _served_mod,
    # resolution
    "_workspace_key_from_request": _resolution_mod,
    # targets
    "_workspace_target_registry": _targets_mod,
    # folder_store
    "_workspace_folder_store": _folder_store_mod,
    # agent_runtime
    "_agent_runtime_registry": _agent_runtime_mod,
}


def __getattr__(name: str) -> Any:  # noqa: ANN401 — module-level attribute forwarding
    """Forward private globals to their owner module (live reads)."""
    owner = _PRIVATE_NAME_OWNERS.get(name)
    if owner is not None:
        return getattr(owner, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
