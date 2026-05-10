"""MCP configuration support for the agent.

This subpackage owns the **config layer** for MCP servers — config files,
secrets, OAuth flows, default seeds. It does **not** implement MCP tool
dispatch, server probing, or tool-source registration: those are pydantic-ai
native (``Agent(toolsets=[MCPServerStdio(...)])``) and constructed inside
:mod:`molexp.agent._pydanticai`.

What lives here:

- :class:`McpStore` / :class:`McpSecretsStore` — workspace + user-home tiers
- :class:`McpServerEntry` / :class:`McpScope` — config record / scope enum
- :class:`UnresolvedSecretError` — config-time error type
- OAuth: token storage, session registry, provider builder
- Defaults: :data:`MCP_DEFAULTS`, :func:`seed_user_defaults`, the
  ``molmcp`` seed + ``MOLEXP_MOLMCP_COMMAND`` env-var contract

What used to live here and was deleted by ``agent-pydanticai-rectification``:

- ``source.py`` / ``tool_store.py`` / ``probe.py`` — parallel-to-pydantic-ai
  implementations of tool dispatch and server probing; replaced by
  ``pydantic_ai.Agent(toolsets=[MCPServerStdio(...)])`` and
  ``MCPServerStdio.list_tools()`` respectively.
"""

from __future__ import annotations

from molexp.agent.mcp.defaults import (
    MCP_DEFAULTS,
    MCP_SEEDED_FILENAME,
    MOLMCP_COMMAND_ENV,
    MOLMCP_USAGE_INSTRUCTIONS,
    seed_user_defaults,
)
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
from molexp.agent.mcp.store import (
    MCP_CONFIG_FILENAME,
    MCP_SECRETS_FILENAME,
    McpScope,
    McpSecretsStore,
    McpServerEntry,
    McpStore,
    UnresolvedSecretError,
)

__all__ = [
    "MCP_CONFIG_FILENAME",
    "MCP_DEFAULTS",
    "MCP_SECRETS_FILENAME",
    "MCP_SEEDED_FILENAME",
    "MOLMCP_COMMAND_ENV",
    "MOLMCP_USAGE_INSTRUCTIONS",
    "START_TIMEOUT_SECONDS",
    "FileTokenStorage",
    "McpScope",
    "McpSecretsStore",
    "McpServerEntry",
    "McpStore",
    "OAuthFlowSession",
    "OAuthSessionRegistry",
    "UnresolvedSecretError",
    "build_oauth_provider",
    "default_redirect_uri",
    "seed_user_defaults",
    "session_registry",
    "storage_for",
]
