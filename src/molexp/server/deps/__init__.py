"""Dependency injection for the MolExp API, organized by domain.

Modules (each is the single owner of its mutable singleton state):

- :mod:`.config` — molcfg-backed :class:`Settings` + ``get_settings``.
- :mod:`.workspace_state` — active-workspace cache, overrides, switch
  subscribers.
- :mod:`.served` — the served-workspace set (``molexp serve`` targets).
- :mod:`.resolution` — request-aware workspace resolution (FastAPI deps).
- :mod:`.targets` — workspace-target registry singleton + remote FS factory.
- :mod:`.folder_store` — in-memory :class:`WorkspaceFolderStore` singleton.
- :mod:`.agent_runtime` — :class:`AgentSessionRegistry` process singleton.

The historical flat import path ``molexp.server.dependencies`` remains a
facade re-exporting this entire surface; importers may use either path.
"""

from __future__ import annotations

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
