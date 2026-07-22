"""``_resolve_agent_mcp_tools`` — per-agent molmcp tool resolution for codegen.

The plan pipeline wires its code-writing agents to consult molmcp mid-run
(so they resolve real molcrafts APIs instead of guessing). The gateway builder
turns each agent's configured MCP *server name* into a concrete SDK-free
``McpToolSpec`` by reading the same ``mcp.json`` config the capability prefetch
uses. A missing/non-stdio server is skipped so a plan without molmcp still runs.
"""

from __future__ import annotations

import json
from pathlib import Path

from molexp.agent.router import McpToolSpec
from molexp.services.plan_runtime.gateway import _resolve_agent_mcp_tools


def _write_workspace_mcp(root: Path, servers: dict) -> None:
    (root / "mcp.json").write_text(json.dumps({"mcpServers": servers}), encoding="utf-8")


def test_resolves_stdio_server_to_toolspec(tmp_path: Path) -> None:
    _write_workspace_mcp(
        tmp_path,
        {"molmcp-test": {"type": "stdio", "command": "molmcp", "args": ["serve"]}},
    )
    resolved = _resolve_agent_mcp_tools({"coder": ("molmcp-test",)}, tmp_path)
    assert set(resolved) == {"coder"}
    (spec,) = resolved["coder"]
    assert isinstance(spec, McpToolSpec)
    assert spec.name == "molmcp-test"
    assert spec.command == "molmcp"
    assert spec.args == ("serve",)


def test_unconfigured_server_yields_no_tools(tmp_path: Path) -> None:
    _write_workspace_mcp(tmp_path, {})
    resolved = _resolve_agent_mcp_tools({"coder": ("definitely-not-configured-xyz",)}, tmp_path)
    # Agent with no resolvable server is absent — the plan still runs ungrounded.
    assert resolved == {}


def test_missing_config_is_graceful(tmp_path: Path) -> None:
    # No mcp.json at all → no tools, no exception.
    resolved = _resolve_agent_mcp_tools({"coder": ("also-not-there-xyz",)}, tmp_path)
    assert resolved == {}
