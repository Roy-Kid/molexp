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


async def check_stdio_handshake(
    command: str,
    args: tuple[str, ...] = (),
    *,
    timeout_seconds: float = 5.0,
) -> None:
    """Open and close a stdio MCP server, raising on failure.

    Lives here (inside ``_pydanticai/``) instead of in ``agent/modes/plan/``
    because it imports ``pydantic_ai.mcp.MCPServerStdio`` directly — that
    import is only permitted inside the ``_pydanticai`` subtree per the
    agent-layer firewall (``tests/test_agent/test_import_guard.py``).
    """
    import asyncio
    import contextlib
    import os
    from collections.abc import Iterator

    from pydantic_ai.mcp import MCPServerStdio

    @contextlib.contextmanager
    def _silence_process_stdio() -> Iterator[None]:
        saved_out = os.dup(1)
        saved_err = os.dup(2)
        devnull = os.open(os.devnull, os.O_WRONLY)
        try:
            os.dup2(devnull, 1)
            os.dup2(devnull, 2)
            yield
        finally:
            os.dup2(saved_out, 1)
            os.dup2(saved_err, 2)
            os.close(saved_out)
            os.close(saved_err)
            os.close(devnull)

    async def _run() -> None:
        server = MCPServerStdio(command, list(args))
        async with server:
            pass

    with _silence_process_stdio():
        await asyncio.wait_for(_run(), timeout=timeout_seconds)


__all__ = ["build_mcp_server", "check_stdio_handshake"]
