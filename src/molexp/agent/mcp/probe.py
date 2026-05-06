"""Probe an MCP server: real connection + list_tools, for the Test button.

Spawns the configured stdio subprocess (or opens the HTTP/SSE/streamable
connection), enters the pydantic-ai async-context, calls ``list_tools()``,
and exits cleanly. Hard-bounded by ``PROBE_TIMEOUT_SECONDS`` so a
misconfigured server cannot hang the request handler.

Errors are returned (not raised) as short, redaction-safe strings — the
secret values that may have been substituted into env/headers are never
echoed back, only the exception class + message head.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any

from mollog import get_logger
from pydantic import BaseModel, ConfigDict

from .store import (
    McpScope,
    McpServerEntry,
    McpStore,
    ResolvedSpec,
    UnresolvedSecretError,
)

logger = get_logger(__name__)

PROBE_TIMEOUT_SECONDS = 10.0


_FROZEN = ConfigDict(frozen=True)


class ProbeOutcome(BaseModel):
    """Result of probing one MCP server."""

    model_config = _FROZEN

    ok: bool
    latency_ms: int
    tool_count: int = 0
    error: str | None = None


class McpToolSummary(BaseModel):
    """Public-safe view of one tool exposed by an MCP server."""

    model_config = _FROZEN

    name: str
    description: str
    server: str  # the entry's logical name (used as group key in UI)
    scope: str  # "user" / "workspace"


class McpServerToolList(BaseModel):
    """Result of enumerating one MCP server's tools.

    Used for the Tools panel — empty ``tools`` + non-null ``error`` means
    the server is misconfigured / offline; the UI renders it as a broken
    group instead of silently dropping it.
    """

    model_config = _FROZEN

    server: str
    scope: str
    ok: bool
    tools: list[McpToolSummary]
    error: str | None = None


async def probe_server(
    store: McpStore,
    entry: McpServerEntry,
) -> ProbeOutcome:
    """Resolve secrets, build a pydantic-ai MCPServer, and call list_tools.

    Returns a friendly ``ProbeOutcome`` even on error. Never raises so the
    route handler can wrap it in a 200 response.
    """
    if not entry.valid:
        return ProbeOutcome(
            ok=False,
            latency_ms=0,
            error=f"Invalid spec: {entry.invalid_reason}",
        )

    try:
        resolved = store.resolve(entry)
    except UnresolvedSecretError as exc:
        return ProbeOutcome(
            ok=False,
            latency_ms=0,
            error=f"Missing secrets: {', '.join(exc.keys)}",
        )
    except Exception as exc:
        return ProbeOutcome(ok=False, latency_ms=0, error=_format_error(exc))

    try:
        server = _build_pydantic_ai_server(resolved, entry.name, entry.scope, store)
    except ImportError:
        return ProbeOutcome(
            ok=False,
            latency_ms=0,
            error="pydantic-ai MCP support unavailable. Install: pip install 'pydantic-ai[mcp]'.",
        )
    except Exception as exc:
        return ProbeOutcome(ok=False, latency_ms=0, error=_format_error(exc))

    started = time.monotonic()
    try:
        tools = await asyncio.wait_for(_connect_and_list(server), timeout=PROBE_TIMEOUT_SECONDS)
    except TimeoutError:
        elapsed = int((time.monotonic() - started) * 1000)
        return ProbeOutcome(
            ok=False,
            latency_ms=elapsed,
            error=f"Timeout after {PROBE_TIMEOUT_SECONDS:.0f}s",
        )
    except Exception as exc:
        elapsed = int((time.monotonic() - started) * 1000)
        return ProbeOutcome(ok=False, latency_ms=elapsed, error=_format_error(exc))

    elapsed = int((time.monotonic() - started) * 1000)
    return ProbeOutcome(ok=True, latency_ms=elapsed, tool_count=len(tools))


async def _connect_and_list(server) -> list:
    """Enter the server's async context, list tools, exit cleanly."""
    async with server:
        return await server.list_tools()


async def list_mcp_tools(
    store: McpStore,
    *,
    timeout: float = PROBE_TIMEOUT_SECONDS,
) -> list[McpServerToolList]:
    """Enumerate tools from every reachable MCP server in the store.

    Probes all valid, non-shadowed, secrets-resolved entries in parallel
    so one slow server can't drag the request handler past ``timeout``.
    Each per-server result is independent — a failed probe returns its
    own :class:`McpServerToolList` with ``ok=False`` and a short error
    string. The total wall-clock is bounded by ``timeout``: stragglers
    are reported as timeouts rather than holding the response open.
    """
    entries = [e for e in store.list() if e.valid and not e.shadowed and not e.unresolved_secrets]
    if not entries:
        return []
    results = await asyncio.gather(
        *(_list_one(store, e, timeout) for e in entries),
        return_exceptions=False,
    )
    return list(results)


async def _list_one(store: McpStore, entry: McpServerEntry, timeout: float) -> McpServerToolList:
    """Probe one server and shape the result into :class:`McpServerToolList`."""
    server_name = entry.name
    scope = entry.scope.value
    try:
        resolved = store.resolve(entry)
    except UnresolvedSecretError as exc:
        return McpServerToolList(
            server=server_name,
            scope=scope,
            ok=False,
            tools=[],
            error=f"Missing secrets: {', '.join(exc.keys)}",
        )
    except Exception as exc:
        return McpServerToolList(
            server=server_name,
            scope=scope,
            ok=False,
            tools=[],
            error=_format_error(exc),
        )

    try:
        srv = _build_pydantic_ai_server(resolved, server_name, entry.scope, store)
    except ImportError:
        return McpServerToolList(
            server=server_name,
            scope=scope,
            ok=False,
            tools=[],
            error="pydantic-ai MCP support unavailable.",
        )
    except Exception as exc:
        return McpServerToolList(
            server=server_name,
            scope=scope,
            ok=False,
            tools=[],
            error=_format_error(exc),
        )

    try:
        tools = await asyncio.wait_for(_connect_and_list(srv), timeout=timeout)
    except TimeoutError:
        return McpServerToolList(
            server=server_name,
            scope=scope,
            ok=False,
            tools=[],
            error=f"Timeout after {timeout:.0f}s",
        )
    except Exception as exc:
        return McpServerToolList(
            server=server_name,
            scope=scope,
            ok=False,
            tools=[],
            error=_format_error(exc),
        )

    return McpServerToolList(
        server=server_name,
        scope=scope,
        ok=True,
        tools=[
            McpToolSummary(
                name=_tool_name(t),
                description=_tool_description(t),
                server=server_name,
                scope=scope,
            )
            for t in tools
        ],
    )


def _tool_name(tool: Any) -> str:
    """Extract a tool's logical name across pydantic-ai/MCP version drift."""
    name = getattr(tool, "name", None)
    if isinstance(name, str):
        return name
    inner = getattr(tool, "tool_def", None) or getattr(tool, "definition", None)
    if inner is not None:
        n = getattr(inner, "name", None)
        if isinstance(n, str):
            return n
    return repr(tool)


def _tool_description(tool: Any) -> str:
    desc = getattr(tool, "description", None)
    if isinstance(desc, str):
        return desc
    inner = getattr(tool, "tool_def", None) or getattr(tool, "definition", None)
    if inner is not None:
        d = getattr(inner, "description", None)
        if isinstance(d, str):
            return d
    return ""


def _build_pydantic_ai_server(spec: ResolvedSpec, name: str, scope: McpScope, store: McpStore):
    """Map the resolved spec onto the right pydantic-ai class.

    OAuth-protected HTTP servers get an ``httpx.AsyncClient(auth=...)``
    wrapping :class:`OAuthClientProvider`; non-OAuth servers stay on the
    simple headers-only path. The probe runs in a non-interactive context
    (no UI session bound), so OAuth here only succeeds when stored tokens
    can be refreshed silently — a fresh authorization will raise and the
    error message tells the user to click Connect.
    """
    from pydantic_ai.mcp import (
        MCPServerSSE,
        MCPServerStdio,
        MCPServerStreamableHTTP,
    )

    if spec.transport == "stdio":
        return MCPServerStdio(
            command=spec.command or "",
            args=list(spec.args),
            env=spec.env or None,
            tool_prefix=name,
        )

    http_client = _maybe_oauth_http_client(spec, name, scope, store)
    # Claude Code .mcp.json convention: ``http`` is streamable HTTP (the
    # modern MCP wire format). ``sse`` is the legacy long-poll transport.
    if spec.transport == "http":
        kwargs = _http_kwargs(spec, http_client)
        return MCPServerStreamableHTTP(url=spec.url or "", tool_prefix=name, **kwargs)
    if spec.transport == "sse":
        kwargs = _http_kwargs(spec, http_client)
        return MCPServerSSE(url=spec.url or "", tool_prefix=name, **kwargs)
    raise ValueError(f"Unknown MCP transport: {spec.transport!r}")


def _http_kwargs(spec: ResolvedSpec, http_client: Any) -> dict[str, Any]:
    """pydantic-ai forbids passing both ``headers`` and ``http_client``;
    pick exactly one based on whether OAuth wrapping is in play."""
    if http_client is not None:
        return {"http_client": http_client}
    return {"headers": spec.headers or None}


def _maybe_oauth_http_client(
    spec: ResolvedSpec, name: str, scope: McpScope, store: McpStore
) -> Any:
    """Build an OAuth-equipped httpx client when the spec requests OAuth.

    Returns ``None`` for non-OAuth specs so the caller falls back to plain
    ``headers=...``. Imports are local so the probe path stays usable when
    only the static-headers code path is exercised (e.g., minimal installs
    that lack the optional ``mcp[client-auth]`` extras).
    """
    if spec.auth is None or spec.auth.type != "oauth2":
        return None
    import httpx

    from .oauth import build_oauth_provider, default_redirect_uri, storage_for

    storage = storage_for(store, scope, name)
    provider = build_oauth_provider(
        server_url=spec.url or "",
        redirect_uri=default_redirect_uri(),
        scopes=list(spec.auth.scopes),
        storage=storage,
    )
    return httpx.AsyncClient(auth=provider)


def _format_error(exc: BaseException) -> str:
    """Produce a short, user-readable string for the probe response.

    Recursively unwraps ``ExceptionGroup`` so the user sees the actual
    cause (e.g. ``HTTPStatusError: 401 Unauthorized``) instead of the
    anyio-default ``"unhandled errors in a TaskGroup (1 sub-exception)"``.
    Falls back to the class name when ``str(exc)`` is empty.
    """
    leaf = _innermost(exc)
    raw = str(leaf) or leaf.__class__.__name__
    if len(raw) > 400:
        raw = raw[:400] + "…"
    return f"{leaf.__class__.__name__}: {raw}"


def _innermost(exc: BaseException) -> BaseException:
    """Walk into ``ExceptionGroup`` / ``__cause__`` to find the root cause.

    Uses ``BaseExceptionGroup`` (Python 3.11+) so we cover both the typed
    and untyped variants. Stops at depth 8 to defend against pathological
    self-referential chains.
    """
    seen: set[int] = set()
    cur: BaseException = exc
    for _ in range(8):
        if id(cur) in seen:
            break
        seen.add(id(cur))
        if isinstance(cur, BaseExceptionGroup) and cur.exceptions:
            cur = cur.exceptions[0]
            continue
        if cur.__cause__ is not None:
            cur = cur.__cause__
            continue
        if cur.__context__ is not None and not cur.__suppress_context__:
            cur = cur.__context__
            continue
        break
    return cur
