"""MCP support for the agent harness.

Importing this package registers an :class:`McpToolSource` on
:func:`molexp.agent.tools.source.register_tool_source` so the
dispatcher can serve ``mcp:<server>.<tool>`` calls. The MCP SDK and
``httpx`` are *not* loaded at module import — every code path that
talks to a server (probe, source) imports them lazily so the agent
package as a whole stays stdlib-only.

This subpackage owns:

- The MCP server / secrets store (workspace + user-home tiers).
- The OAuth integration.
- The connection probe.
- The :class:`McpToolSource` that bridges the harness ``ToolSource``
  contract to the underlying MCP SDK.
"""

from __future__ import annotations

from molexp.agent.mcp.oauth import (
    START_TIMEOUT_SECONDS,
    FileTokenStorage,
    OAuthFlowSession,
    OAuthSessionRegistry,
    build_oauth_provider,
    default_redirect_uri,
    session_registry,
    storage_for,
)
from molexp.agent.mcp.probe import (
    PROBE_TIMEOUT_SECONDS,
    McpServerToolList,
    McpToolSummary,
    ProbeOutcome,
    list_mcp_tools,
    probe_server,
)
from molexp.agent.mcp.source import SOURCE_NAME, McpToolSource
from molexp.agent.mcp.store import (
    MCP_CONFIG_FILENAME,
    MCP_SECRETS_FILENAME,
    McpScope,
    McpSecretsStore,
    McpServerEntry,
    McpStore,
    UnresolvedSecretError,
)
from molexp.agent.tools.source import register_tool_source


def _register() -> None:
    register_tool_source(McpToolSource())


_register()


__all__ = [
    "FileTokenStorage",
    "MCP_CONFIG_FILENAME",
    "MCP_SECRETS_FILENAME",
    "McpScope",
    "McpSecretsStore",
    "McpServerEntry",
    "McpServerToolList",
    "McpStore",
    "McpToolSource",
    "McpToolSummary",
    "OAuthFlowSession",
    "OAuthSessionRegistry",
    "PROBE_TIMEOUT_SECONDS",
    "ProbeOutcome",
    "SOURCE_NAME",
    "START_TIMEOUT_SECONDS",
    "UnresolvedSecretError",
    "build_oauth_provider",
    "default_redirect_uri",
    "list_mcp_tools",
    "probe_server",
    "session_registry",
    "storage_for",
]
