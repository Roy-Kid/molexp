"""OAuth 2.0 (Authorization Code + PKCE) integration for remote MCP servers.

This module bridges three pieces:

1. **MCP Python SDK** (``mcp.client.auth.OAuthClientProvider``) — implements
   the actual OAuth flow (PKCE, Dynamic Client Registration, token exchange,
   refresh). It's an ``httpx.Auth`` subclass and plugs into any
   ``httpx.AsyncClient``.
2. **pydantic-ai** ``MCPServerStreamableHTTP`` / ``MCPServerSSE`` — accept a
   pre-built ``http_client`` kwarg, so an httpx client wrapped with the
   OAuth auth handler will transparently negotiate auth on every request.
3. **molexp's web UI** — initiates the flow with a click, gets back an
   authorize URL, opens it in the browser, posts the callback ``code`` back.

The SDK requires the host application to supply two callbacks
(``redirect_handler`` and ``callback_handler``) and a ``TokenStorage``
implementation. We provide:

- :class:`FileTokenStorage` — JSON-backed atomic store, one file per
  ``(scope, server_name)`` tuple.
- :class:`OAuthFlowSession` — futures-based bridge that lets the FastAPI
  ``/oauth/start`` endpoint kick off a flow, return the authorize URL to
  the browser, and have a separate ``/oauth/callback`` endpoint complete
  it on the same in-flight task.
- :func:`build_oauth_provider` — convenience factory that wires storage +
  flow session + SDK client metadata.

# TODO(pydantic-ai): when pydantic-ai ships first-class MCP OAuth support
# (tracked upstream under pydantic/pydantic-ai#mcp-oauth), drop this whole
# module and pass that helper to ``MCPServerStreamableHTTP`` / SSE directly.
# The public surface (build_oauth_provider, FileTokenStorage,
# OAuthFlowSession) should stay stable enough that the swap is local to
# ``mcp_probe.py`` / ``runtime.py``.
"""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from threading import Lock
from typing import TYPE_CHECKING

from mcp.client.auth import OAuthClientProvider, TokenStorage
from mcp.shared.auth import (
    OAuthClientInformationFull,
    OAuthClientMetadata,
    OAuthToken,
)
from mollog import get_logger

if TYPE_CHECKING:
    from .mcp_store import McpScope, McpStore

logger = get_logger(__name__)

# The redirect URI the browser bounces back to. The SPA owns this route
# (``/oauth-callback``); it just extracts ``code`` and ``state`` and posts
# them to ``POST /api/agent/admin/mcp-servers/{name}/oauth/callback``.
DEFAULT_REDIRECT_PATH = "/oauth-callback"
DEFAULT_CLIENT_NAME = "molexp"

# Hard cap on how long the SDK's callback_handler will wait for the user
# to complete the browser dance. The route handler returns 408 if exceeded.
CALLBACK_TIMEOUT_SECONDS = 300.0

# Wait budget for ``/oauth/start`` to obtain the authorize URL from the SDK.
# DCR + metadata discovery typically takes <1s; we give it generous slack
# but never block forever — a misconfigured issuer URL would otherwise hang
# the request handler.
START_TIMEOUT_SECONDS = 30.0


# ── Token storage ──────────────────────────────────────────────────────────


class FileTokenStorage(TokenStorage):
    """JSON-backed implementation of MCP SDK's ``TokenStorage`` Protocol.

    One file per ``(scope, server_name)`` pair. Layout::

        <root>/.mcp_oauth/<server_name>.json
        {
          "tokens": {...OAuthToken JSON...} | null,
          "client_info": {...OAuthClientInformationFull JSON...} | null
        }

    Atomic writes via temp-file + ``os.replace``; chmod 0o600 on POSIX so
    refresh tokens don't leak to other local users. Best-effort on Windows.
    """

    DIRNAME = ".mcp_oauth"

    def __init__(self, root: str | Path, server_name: str) -> None:
        self._root = Path(root)
        self._server_name = server_name
        self._path = self._root / self.DIRNAME / f"{server_name}.json"
        self._lock = Lock()

    @property
    def path(self) -> Path:
        return self._path

    @property
    def server_name(self) -> str:
        return self._server_name

    async def get_tokens(self) -> OAuthToken | None:
        raw = self._read().get("tokens")
        if not isinstance(raw, dict):
            return None
        try:
            return OAuthToken.model_validate(raw)
        except Exception:  # corrupt file — treat as missing
            logger.warning(f"Corrupt OAuth tokens for {self._server_name}, ignoring")
            return None

    async def set_tokens(self, tokens: OAuthToken) -> None:
        with self._lock:
            current = self._read()
            current["tokens"] = tokens.model_dump(mode="json", exclude_none=True)
            self._write(current)

    async def get_client_info(self) -> OAuthClientInformationFull | None:
        raw = self._read().get("client_info")
        if not isinstance(raw, dict):
            return None
        try:
            return OAuthClientInformationFull.model_validate(raw)
        except Exception:
            logger.warning(f"Corrupt OAuth client_info for {self._server_name}, ignoring")
            return None

    async def set_client_info(self, client_info: OAuthClientInformationFull) -> None:
        with self._lock:
            current = self._read()
            current["client_info"] = client_info.model_dump(mode="json", exclude_none=True)
            self._write(current)

    def clear(self) -> bool:
        """Drop all stored OAuth state for this server. Returns True if a
        file existed prior to deletion (i.e. the user had connected before).
        """
        with self._lock:
            if not self._path.exists():
                return False
            self._path.unlink()
            return True

    def _read(self) -> dict:
        if not self._path.exists():
            return {}
        try:
            return json.loads(self._path.read_text())
        except (OSError, json.JSONDecodeError):
            return {}

    def _write(self, payload: dict) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self._path.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
        try:
            os.chmod(tmp, 0o600)
        except OSError:
            pass
        os.replace(tmp, self._path)


# ── Flow session — bridges SDK callbacks to FastAPI request/response ──────


class OAuthFlowSession:
    """Async bridge between SDK callbacks and HTTP request handlers.

    Lifecycle:

    1. ``/oauth/start`` creates a session and launches a background task
       that drives ``OAuthClientProvider``. The provider invokes
       ``redirect_handler(url)`` once it knows the authorize URL — we
       resolve :attr:`authorize_url_future` so ``/oauth/start`` can return.
    2. The browser pops the authorize URL, user logs in, the IdP bounces
       back to ``<molexp>/oauth-callback?code=…&state=…``. The SPA POSTs
       those to ``/oauth/callback`` which calls :meth:`submit_callback`.
    3. The provider's ``callback_handler`` was awaiting that future, so it
       returns ``(code, state)`` and the provider completes token exchange.
       Tokens land in :class:`FileTokenStorage` — the next MCP request
       picks them up automatically.

    Futures are created lazily on first access so a session can be
    constructed outside an event loop (e.g. in synchronous tests or the
    registry's ``Lock``-guarded ``create``); ``asyncio.Future()`` requires
    a running loop and would otherwise raise.
    """

    def __init__(self) -> None:
        self._authorize_url_future: asyncio.Future[str] | None = None
        self._callback_future: asyncio.Future[tuple[str, str | None]] | None = None
        # Cache the (code, state) tuple if it was submitted before the
        # callback_future was lazily created; we replay it on first access.
        self._pending_callback: tuple[str, str | None] | None = None
        self._cancelled: BaseException | None = None
        # Holds the background task that drives ``OAuthClientProvider``
        # so the callback route can await its completion (and so it isn't
        # garbage-collected before token exchange finishes).
        self.flow_task: asyncio.Task[None] | None = None

    @property
    def authorize_url_future(self) -> asyncio.Future[str]:
        if self._authorize_url_future is None:
            self._authorize_url_future = asyncio.get_event_loop().create_future()
            if self._cancelled is not None:
                self._authorize_url_future.set_exception(self._cancelled)
        return self._authorize_url_future

    @property
    def callback_future(self) -> asyncio.Future[tuple[str, str | None]]:
        if self._callback_future is None:
            self._callback_future = asyncio.get_event_loop().create_future()
            if self._pending_callback is not None:
                self._callback_future.set_result(self._pending_callback)
            elif self._cancelled is not None:
                self._callback_future.set_exception(self._cancelled)
        return self._callback_future

    async def redirect_handler(self, authorize_url: str) -> None:
        """SDK calls this with the URL the user must open in a browser."""
        fut = self.authorize_url_future
        if not fut.done():
            fut.set_result(authorize_url)

    async def callback_handler(self) -> tuple[str, str | None]:
        """SDK awaits this to receive ``(code, state)`` from the browser."""
        try:
            return await asyncio.wait_for(
                self.callback_future, timeout=CALLBACK_TIMEOUT_SECONDS
            )
        except TimeoutError:
            raise TimeoutError(
                f"OAuth callback not received within {CALLBACK_TIMEOUT_SECONDS:.0f}s"
            ) from None

    def submit_callback(self, code: str, state: str | None) -> bool:
        """Called by ``/oauth/callback``. Returns True iff the future was
        still pending — duplicate / late submissions return False rather
        than raising so the route handler can convert to 410 Gone."""
        if self._callback_future is None:
            # Future not yet created (no awaiter) — stash the result so
            # the next reader sees it on first access.
            if self._pending_callback is not None:
                return False
            self._pending_callback = (code, state)
            return True
        if self._callback_future.done():
            return False
        self._callback_future.set_result((code, state))
        return True

    def cancel(self, reason: str = "cancelled") -> None:
        """Tear down the flow if the user closes the popup or the request
        handler times out. Safe to call even if futures already resolved
        or before any future has been instantiated."""
        exc = asyncio.CancelledError(reason)
        self._cancelled = exc
        for fut in (self._authorize_url_future, self._callback_future):
            if fut is not None and not fut.done():
                fut.set_exception(exc)

    @property
    def cancelled(self) -> bool:
        """True if :meth:`cancel` has been called (independent of whether
        a future was ever materialized to receive the cancellation)."""
        return self._cancelled is not None


# ── Provider factory ───────────────────────────────────────────────────────


def build_client_metadata(
    redirect_uri: str,
    *,
    scopes: list[str] | None = None,
    client_name: str = DEFAULT_CLIENT_NAME,
) -> OAuthClientMetadata:
    """Construct the ``OAuthClientMetadata`` payload sent to the IdP during
    Dynamic Client Registration. Scopes are joined with spaces per RFC 6749.
    """
    return OAuthClientMetadata(
        redirect_uris=[redirect_uri],  # type: ignore[arg-type]
        token_endpoint_auth_method="none",
        grant_types=["authorization_code", "refresh_token"],
        response_types=["code"],
        scope=" ".join(scopes) if scopes else None,
        client_name=client_name,
    )


def build_oauth_provider(
    *,
    server_url: str,
    redirect_uri: str,
    scopes: list[str] | None,
    storage: TokenStorage,
    session: OAuthFlowSession | None = None,
    client_name: str = DEFAULT_CLIENT_NAME,
) -> OAuthClientProvider:
    """Assemble an ``OAuthClientProvider`` for use with an ``httpx.AsyncClient(auth=...)``.

    Pass a live :class:`OAuthFlowSession` from a UI request handler to make
    the provider drive an interactive PKCE flow (browser pop, callback
    POST). Pass ``session=None`` for non-interactive contexts (runtime
    toolset load, Test button) — the provider will still refresh stored
    tokens silently, but a fresh authorization will raise via the runtime
    no-op handlers and the caller should surface "click Connect" to the
    user.
    """
    redirect = session.redirect_handler if session is not None else _runtime_redirect_handler
    callback = session.callback_handler if session is not None else _runtime_callback_handler
    return OAuthClientProvider(
        server_url=server_url,
        client_metadata=build_client_metadata(
            redirect_uri,
            scopes=scopes,
            client_name=client_name,
        ),
        storage=storage,
        redirect_handler=redirect,
        callback_handler=callback,
    )


# ── Session registry — global keyed by (scope, name) ───────────────────────


class OAuthSessionRegistry:
    """Process-wide registry of in-flight OAuth flows.

    A flow is "in flight" between ``/oauth/start`` and ``/oauth/callback``;
    the registry lets the callback handler find the right
    :class:`OAuthFlowSession` by ``(scope, server_name)`` key.

    In-memory only — restarting the server drops any pending flows. Users
    just click Connect again. Completed tokens persist via
    :class:`FileTokenStorage` so this loss is benign.
    """

    def __init__(self) -> None:
        self._sessions: dict[tuple[str, str], OAuthFlowSession] = {}
        self._lock = Lock()

    def _key(self, scope: McpScope | str, name: str) -> tuple[str, str]:
        scope_str = scope.value if hasattr(scope, "value") else str(scope)
        return (scope_str, name)

    def create(self, scope: McpScope | str, name: str) -> OAuthFlowSession:
        """Register a fresh session, replacing any existing one (the user
        clicking Connect twice cancels the older flow)."""
        session = OAuthFlowSession()
        with self._lock:
            existing = self._sessions.get(self._key(scope, name))
            if existing is not None:
                existing.cancel("superseded")
            self._sessions[self._key(scope, name)] = session
        return session

    def get(self, scope: McpScope | str, name: str) -> OAuthFlowSession | None:
        with self._lock:
            return self._sessions.get(self._key(scope, name))

    def discard(self, scope: McpScope | str, name: str) -> None:
        with self._lock:
            self._sessions.pop(self._key(scope, name), None)


_registry: OAuthSessionRegistry | None = None


def session_registry() -> OAuthSessionRegistry:
    """Process-singleton accessor. Lazy so test runs can monkey-patch."""
    global _registry
    if _registry is None:
        _registry = OAuthSessionRegistry()
    return _registry


# ── Storage helpers ────────────────────────────────────────────────────────


def storage_for(store: McpStore, scope: McpScope, server_name: str) -> FileTokenStorage:
    """Return the :class:`FileTokenStorage` for a given scope+server.

    Workspace scope points at ``<workspace_root>/.mcp_oauth/``; User scope
    points at ``~/.molexp/.mcp_oauth/``. Mirrors the layout of the secrets
    store next to it.
    """
    from .mcp_store import USER_DIR, McpScope as _McpScope

    root = store.workspace_root if scope is _McpScope.WORKSPACE else USER_DIR
    return FileTokenStorage(root, server_name)


# ── No-op handlers for non-interactive contexts (runtime use) ──────────────


async def _runtime_redirect_handler(_url: str) -> None:
    raise RuntimeError(
        "OAuth flow cannot start outside an interactive UI request — "
        "click Connect in the agent settings to (re)authorize this server."
    )


async def _runtime_callback_handler() -> tuple[str, str | None]:
    raise RuntimeError(
        "OAuth callback unavailable at runtime — token expired or never set; "
        "click Connect in the agent settings."
    )


def default_redirect_uri() -> str:
    """Where the IdP bounces the browser back to.

    Defaults to ``http://127.0.0.1:8000/oauth-callback`` (where the SPA
    served by ``molexp serve --port 8000`` lives). Override via the
    ``MOLEXP_OAUTH_REDIRECT_URI`` env var for non-default deployments.
    """
    return os.environ.get(
        "MOLEXP_OAUTH_REDIRECT_URI", "http://127.0.0.1:8000/oauth-callback"
    )


