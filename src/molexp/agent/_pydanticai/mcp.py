"""MCP toolset builder — wraps pydantic-ai's ``MCPToolset`` API.

Sole site for ``from pydantic_ai.mcp import ...`` (alongside the harness).
Callers outside ``_pydanticai`` obtain MCP toolsets via
:func:`build_mcp_server` instead of importing pydantic-ai directly, keeping
the import-boundary firewall intact.

v2 migration note: the deprecated ``MCPServerStdio`` / ``MCPServerSSE`` /
``MCPServerStreamableHTTP`` classes are replaced by ``MCPToolset`` over
explicit fastmcp transports (``StdioTransport`` / ``SSETransport`` /
``StreamableHttpTransport``, re-exported by ``pydantic_ai.mcp``). We build
the transport explicitly rather than passing a URL string because pydantic-ai
infers SSE-vs-streamable-HTTP from the URL shape (``/sse`` suffix), which
would misroute an explicitly-configured transport whose URL doesn't follow
that convention. The v1 ``tool_prefix`` becomes ``.prefixed(name)`` — same
``{name}_{tool}`` naming.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable

    import httpx
    from pydantic_ai.toolsets import PrefixedToolset


def _make_httpx_client_factory(
    http_client: httpx.AsyncClient,
) -> Callable[..., httpx.AsyncClient]:
    """Adapt a pre-built ``httpx.AsyncClient`` to fastmcp's ``httpx_client_factory``.

    fastmcp calls the factory with keyword arguments (``headers``, ``auth``,
    ``timeout``, ``follow_redirects``, …); accepting ``**kwargs`` and ignoring
    them mirrors pydantic-ai's own internal adapter — the user-supplied client
    is authoritative for headers/auth/timeouts.
    """

    def factory(**_kwargs: object) -> httpx.AsyncClient:
        return http_client

    return factory


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
) -> PrefixedToolset[Any]:
    """Map a resolved transport spec onto a pydantic-ai ``MCPToolset``.

    Returns the ``MCPToolset`` wrapped with ``.prefixed(name)`` so every tool
    is exposed as ``{name}_{tool}`` — identical naming to the v1
    ``tool_prefix=name`` behavior. ``name`` is also set as the toolset ``id``.

    The caller passes either ``http_client`` or ``headers`` (never both);
    when ``http_client`` is given it is authoritative and ``headers`` are
    ignored, matching the v1 builder's precedence.
    """
    from pydantic_ai.mcp import (
        MCPToolset,
        SSETransport,
        StdioTransport,
        StreamableHttpTransport,
    )

    if transport == "stdio":
        # keep_alive=False restores v1 ``MCPServerStdio`` lifecycle: the
        # subprocess is terminated when the toolset context exits (fastmcp
        # defaults to keeping it alive for session reuse).
        stdio = StdioTransport(command=command, args=list(args), env=env, keep_alive=False)
        return MCPToolset(stdio, id=name).prefixed(name)

    if transport in ("http", "sse"):
        factory = _make_httpx_client_factory(http_client) if http_client is not None else None
        transport_cls = StreamableHttpTransport if transport == "http" else SSETransport
        fastmcp_transport = transport_cls(
            url=url,
            headers=headers if http_client is None else None,
            httpx_client_factory=factory,
        )
        return MCPToolset(fastmcp_transport, id=name).prefixed(name)

    raise ValueError(f"Unknown MCP transport: {transport!r}")


async def check_stdio_handshake(
    command: str,
    args: tuple[str, ...] = (),
    *,
    timeout_seconds: float = 5.0,
) -> None:
    """Open and close a stdio MCP server, raising on failure.

    Lives here (inside ``_pydanticai/``) instead of in ``agent/modes/plan/``
    because it imports ``pydantic_ai.mcp`` directly — that import is only
    permitted inside the ``_pydanticai`` subtree per the agent-layer
    firewall (``tests/test_agent/test_import_guard.py``).
    """
    import asyncio
    import contextlib
    import os
    from collections.abc import Iterator

    from pydantic_ai.mcp import MCPToolset, StdioTransport

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
        # keep_alive=False so the probe actually terminates the subprocess on
        # exit — fastmcp's default keeps it alive for reuse, which would leak
        # a child process per handshake check.
        transport = StdioTransport(command=command, args=list(args), keep_alive=False)
        server = MCPToolset(transport)
        async with server:
            pass

    with _silence_process_stdio():
        await asyncio.wait_for(_run(), timeout=timeout_seconds)


__all__ = ["build_mcp_server", "check_stdio_handshake"]
