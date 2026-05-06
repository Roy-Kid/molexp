"""Codex app-server as a coding-agent plugin.

Importing this package exposes :class:`CodexAppServerClient`, which spawns
``codex app-server`` as a long-lived subprocess and drives it via JSON-RPC
over stdio. Translates Codex protocol messages into normalized
:class:`molexp.agent.coding_protocol.TurnResult` plus an event callback.

Reverse-RPC tool calls (where the Codex agent invokes a callback into the
host) are dispatched through a caller-supplied :class:`ToolHandler` —
typically symphony's GitHub GraphQL bridge. The plugin defines the
protocol shape only; concrete implementations live with the caller.
"""

from __future__ import annotations

from molexp.plugins.agent_codex.client import CodexAppServerClient, ToolHandler
from molexp.plugins.agent_codex.config import CodexConfig

__all__ = ["CodexAppServerClient", "CodexConfig", "ToolHandler"]
