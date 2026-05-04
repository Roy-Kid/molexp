"""MCP as a :class:`molexp.agent.ToolSource` plugin.

Importing this package registers the source on
:func:`molexp.agent.tools.source.register_tool_source`. The plugin
owns:

- The MCP server / secrets stores (workspace + user-home tiers).
- The OAuth integration (``oauth.py``).
- The connection probe (``probe.py``).
- The harness-facing :class:`McpToolSource` (``source.py``).

OAuth, HTTP clients, MCP SDK access, and secret stores all live here
— never in a model plugin.
"""

from __future__ import annotations

from molexp.agent.tools.source import register_tool_source
from molexp.plugins.tool_mcp.oauth import (
    FileTokenStorage,
    OAuthFlowSession,
    OAuthSessionRegistry,
    START_TIMEOUT_SECONDS,
    build_oauth_provider,
    default_redirect_uri,
    session_registry,
    storage_for,
)
from molexp.plugins.tool_mcp.probe import (
    McpServerToolList,
    McpToolSummary,
    PROBE_TIMEOUT_SECONDS,
    ProbeOutcome,
    list_mcp_tools,
    probe_server,
)
from molexp.plugins.tool_mcp.source import McpToolSource, SOURCE_NAME
from molexp.plugins.tool_mcp.store import (
    MCP_CONFIG_FILENAME,
    MCP_SECRETS_FILENAME,
    McpScope,
    McpSecretsStore,
    McpServerEntry,
    McpStore,
    UnresolvedSecretError,
)


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
