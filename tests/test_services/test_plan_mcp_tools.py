"""``_resolve_agent_mcp_tools`` — per-agent molmcp tool resolution for codegen.

The plan pipeline wires its code-writing agents to consult molmcp mid-run
(so they resolve real molcrafts APIs instead of guessing). The gateway builder
turns each agent's configured MCP *server name* into a concrete SDK-free
``McpToolSpec`` by reading the same ``mcp.json`` config the capability prefetch
uses. When nothing resolves the plan still runs ungrounded — resolution never
raises.
"""

from __future__ import annotations

import json
from pathlib import Path

from molexp.agent.router import McpToolSpec
from molexp.services.plan_runtime.gateway import _resolve_agent_mcp_tools


def _write_workspace_mcp(root: Path, servers: dict) -> None:
    (root / "mcp.json").write_text(json.dumps({"mcpServers": servers}), encoding="utf-8")


class TestResolveAgentMcpTools:
    def test_resolves_stdio_server_to_toolspec(self, tmp_path: Path) -> None:
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

    def test_unresolvable_server_yields_no_tools_without_raising(self, tmp_path: Path) -> None:
        """No mcp config at all → empty map, no exception; the plan runs
        ungrounded rather than crashing when molmcp is absent."""
        resolved = _resolve_agent_mcp_tools({"coder": ("also-not-there-xyz",)}, tmp_path)
        assert resolved == {}
