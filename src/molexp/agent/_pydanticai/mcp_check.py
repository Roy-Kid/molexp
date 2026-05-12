"""pydantic-ai-backed MCP runtime checks.

This file exists to keep the agent-layer pydantic-ai firewall intact:
PlanMode preflight code can ask for an MCP stdio handshake without
importing pydantic-ai outside ``agent/_pydanticai``.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
from collections.abc import Iterator

from pydantic_ai.mcp import MCPServerStdio

__all__ = ["check_mcp_stdio_handshake"]


async def check_mcp_stdio_handshake(
    command: str,
    args: tuple[str, ...] = (),
    *,
    timeout_seconds: float = 5.0,
    quiet: bool = True,
) -> None:
    """Open and close a stdio MCP server, raising on failure."""

    async def _run() -> None:
        server = MCPServerStdio(command, list(args))
        async with server:
            pass

    if not quiet:
        await asyncio.wait_for(_run(), timeout=timeout_seconds)
        return

    with _silence_process_stdio():
        await asyncio.wait_for(_run(), timeout=timeout_seconds)


@contextlib.contextmanager
def _silence_process_stdio() -> Iterator[None]:
    """Temporarily point stdout/stderr fds at ``os.devnull``.

    Some MCP servers print startup banners directly from the child
    process.  Redirecting Python's ``sys.stderr`` is not enough because
    the child inherits OS-level file descriptors.
    """
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
