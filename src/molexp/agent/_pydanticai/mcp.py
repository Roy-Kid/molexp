"""MCP server builder — wraps pydantic-ai's MCP transports.

Sole site for ``from pydantic_ai.mcp import ...`` (alongside the harness).
``agent/mcp/probe.py`` calls into :func:`build_mcp_server` instead of
importing pydantic-ai directly, keeping the import-boundary firewall
intact.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import httpx
    from pydantic_ai.mcp import MCPServerSSE, MCPServerStdio, MCPServerStreamableHTTP


def build_mcp_server(
    *,
    transport: str,
    name: str,
    command: str = "",
    args: tuple[str, ...] = (),
    env: dict[str, str] | None = None,
    url: str = "",
    http_client: httpx.AsyncClient | None = None,
    headers: dict[str, str] | None = None,
) -> MCPServerStdio | MCPServerStreamableHTTP | MCPServerSSE:
    """Map a resolved transport spec onto the right pydantic-ai MCP class.

    Returns a pydantic-ai ``MCPServerStdio`` / ``MCPServerStreamableHTTP``
    / ``MCPServerSSE`` instance. The caller passes either ``http_client``
    or ``headers`` (never both — pydantic-ai rejects the combination).
    """
    from pydantic_ai.mcp import (
        MCPServerSSE,
        MCPServerStdio,
        MCPServerStreamableHTTP,
    )

    if transport == "stdio":
        return MCPServerStdio(
            command=command,
            args=list(args),
            env=env,
            tool_prefix=name,
        )

    if transport == "http":
        if http_client is not None:
            return MCPServerStreamableHTTP(url=url, tool_prefix=name, http_client=http_client)
        return MCPServerStreamableHTTP(url=url, tool_prefix=name, headers=headers)
    if transport == "sse":
        if http_client is not None:
            return MCPServerSSE(url=url, tool_prefix=name, http_client=http_client)
        return MCPServerSSE(url=url, tool_prefix=name, headers=headers)
    raise ValueError(f"Unknown MCP transport: {transport!r}")


__all__ = ["build_mcp_server"]
