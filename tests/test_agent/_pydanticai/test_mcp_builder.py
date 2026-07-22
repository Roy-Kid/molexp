"""Construction contracts for :func:`build_mcp_server` (pydantic-ai v2 MCP API).

Locks the ``MCPServer*`` → ``MCPToolset`` migration mapping molexp owns:

- the module-wide ``filterwarnings("error")`` fails the build on any
  ``DeprecationWarning`` (no legacy transport classes);
- ``name`` maps to both the toolset ``id`` and the ``{name}_`` tool prefix;
- the configured ``transport`` always wins over pydantic-ai's URL-shape
  inference (``/sse`` suffix), in both directions;
- a caller-supplied ``http_client`` is injected via the fastmcp factory and
  takes precedence over ``headers``.
"""

from __future__ import annotations

import httpx
import pytest

from molexp.agent._pydanticai.mcp import build_mcp_server

pytestmark = pytest.mark.filterwarnings("error::DeprecationWarning")


class TestBuildMcpServer:
    def test_stdio_builds_prefixed_toolset_with_keep_alive_passthrough(self) -> None:
        from pydantic_ai.mcp import MCPToolset, StdioTransport
        from pydantic_ai.toolsets import PrefixedToolset

        toolset = build_mcp_server(
            transport="stdio",
            name="molmcp",
            command="uvx",
            args=("molmcp", "stdio"),
            env={"MOLEXP_X": "1"},
            keep_alive=False,
        )

        assert isinstance(toolset, PrefixedToolset)
        assert toolset.prefix == "molmcp"
        inner = toolset.wrapped
        assert isinstance(inner, MCPToolset)
        assert inner.id == "molmcp"
        transport = inner.client.transport
        assert isinstance(transport, StdioTransport)
        assert transport.command == "uvx"
        assert transport.args == ["molmcp", "stdio"]
        assert transport.env == {"MOLEXP_X": "1"}
        assert transport.keep_alive is False

    def test_http_transport_wins_over_sse_url_shape_and_passes_headers(self) -> None:
        from pydantic_ai.mcp import StreamableHttpTransport

        toolset = build_mcp_server(
            transport="http",
            name="srv",
            # URL ends in /sse: the explicit config transport must win over
            # pydantic-ai's URL-shape inference.
            url="https://example.test/sse",
            headers={"Authorization": "Bearer t"},
        )

        transport = toolset.wrapped.client.transport
        assert isinstance(transport, StreamableHttpTransport)
        assert transport.url == "https://example.test/sse"
        assert transport.headers == {"Authorization": "Bearer t"}

    def test_sse_transport_wins_and_injects_http_client_via_factory(self) -> None:
        from pydantic_ai.mcp import SSETransport

        client = httpx.AsyncClient()
        try:
            toolset = build_mcp_server(
                transport="sse",
                name="srv2",
                # URL does NOT end in /sse — explicit transport still wins.
                url="https://example.test/events",
                http_client=client,
            )

            transport = toolset.wrapped.client.transport
            assert isinstance(transport, SSETransport)
            assert transport.url == "https://example.test/events"
            factory = transport.httpx_client_factory
            assert factory is not None
            assert factory(headers={}, timeout=None, auth=None) is client
        finally:
            del client

    def test_http_client_takes_precedence_over_headers(self) -> None:
        client = httpx.AsyncClient()
        toolset = build_mcp_server(
            transport="http",
            name="srv3",
            url="https://example.test/mcp",
            http_client=client,
            headers={"X-Ignored": "yes"},
        )

        transport = toolset.wrapped.client.transport
        # v1 precedence: http_client is authoritative, headers are dropped.
        assert transport.headers == {}
        factory = transport.httpx_client_factory
        assert factory is not None
        assert factory() is client

    def test_unknown_transport_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="ws"):
            build_mcp_server(transport="ws", name="x")
