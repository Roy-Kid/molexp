"""Agent admin routes — provider config + MCP servers over shared services.

The Settings page's backend: ``GET/PUT /api/agent/provider`` read/write the
operator config (``~/.molexp/config.json``) through the ONE shared
loader/writer in :mod:`molexp.services.operator_config` (the same file and
code path ``molexp config set`` uses), ``POST /api/agent/provider/test``
runs the plan preflight (no disk, no network), and
``GET/POST /api/agent/mcp/servers`` wrap the existing two-tier
:class:`~molexp.agent.mcp.store.McpStore`.

Secret rule: API-key **values** are never echoed — every response carries
only ``apiKeySet`` plus a masked ``apiKeyPreview`` (first-2 + "…" + last-4),
and nothing here logs a key.

Registered BEFORE ``agent.router`` in ``routes/__init__`` so these paths win
over the legacy 503 catch-all (FastAPI matches in registration order); the
genuinely retired ``/api/agent/*`` session paths keep 503-ing behind it.

In-code-wins + re-bridge: a PUT persists to disk, then clears the bridged
``molexp.config`` keys it just changed and re-bridges — so the *running*
server picks the change up without a restart, and a previously-bridged stale
value never shadows the update.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field

from molexp.server.dependencies import get_workspace

if TYPE_CHECKING:
    from molexp.agent.mcp.store import McpServerEntry, McpStore
    from molexp.workspace import Workspace

# NOTE: no ``/api`` here — the router is mounted under the global ``/api``
# prefix by ``create_app``.
router = APIRouter(prefix="/agent", tags=["agent-admin"])

SUPPORTED_PROVIDERS = ["anthropic", "openai", "google", "deepseek", "openai-compatible"]


class ProviderResponse(BaseModel):
    """The Settings page's provider view — never carries a key value."""

    provider: str
    model: str
    baseUrl: str
    apiKeyPreview: str
    apiKeySet: bool
    instructions: str
    supportedProviders: list[str] = Field(default_factory=lambda: list(SUPPORTED_PROVIDERS))


class ProviderUpdateRequest(BaseModel):
    """PUT body — only submitted fields are written."""

    provider: str | None = None
    model: str | None = None
    api_key: str | None = None
    base_url: str | None = None
    instructions: str | None = None


class ProviderTestResponse(BaseModel):
    """POST /provider/test result — an honest preflight, not a chat reply."""

    ok: bool
    provider: str
    model: str
    latencyMs: int
    reply: str
    error: str | None = None


def _mask_key(value: str) -> str:
    """``sk-abc…wxyz`` — enough to recognize a key, never enough to use it."""
    if len(value) <= 12:
        return "…"
    return f"{value[:2]}…{value[-4:]}"


def _provider_of(model: str, stored: str | None) -> str:
    """Derive the provider: stored value wins, else the model-id prefix."""
    if stored:
        return stored
    if ":" in model:
        return model.split(":", 1)[0]
    return ""


def _agent_section(config: dict[str, Any]) -> dict[str, Any]:
    section = config.get("agent")
    return section if isinstance(section, dict) else {}


def _provider_response() -> ProviderResponse:
    from molexp.services.operator_config import load_operator_config

    agent = _agent_section(load_operator_config())
    raw_model = agent.get("model")
    model = raw_model if isinstance(raw_model, str) else ""
    provider = _provider_of(model, agent.get("provider"))
    key_name = f"{provider.replace('-', '_')}_api_key" if provider else ""
    key = agent.get(key_name) if key_name else None
    key_set = isinstance(key, str) and bool(key)
    return ProviderResponse(
        provider=provider,
        model=model or "",
        baseUrl=agent.get("base_url") or "",
        apiKeyPreview=_mask_key(key) if key_set else "",
        apiKeySet=key_set,
        instructions=agent.get("instructions") or "",
    )


@router.get("/provider", response_model=ProviderResponse)
def get_provider() -> ProviderResponse:
    """Current provider settings from the operator config (keys masked)."""
    return _provider_response()


@router.put("/provider", response_model=ProviderResponse)
def update_provider(request: ProviderUpdateRequest) -> ProviderResponse:
    """Persist submitted provider fields, then re-bridge the live process.

    The write goes through the shared :func:`set_operator_values` (same file,
    same atomic writer as ``molexp config set``). The bridged
    ``molexp.config`` keys this PUT changes are cleared before re-bridging so
    the running server serves the new values immediately.
    """
    import molexp
    from molexp.services.operator_config import (
        AGENT_MODEL_KEY,
        bridge_operator_config,
        load_operator_config,
        set_operator_values,
    )

    current = _agent_section(load_operator_config())
    provider = request.provider or _provider_of(
        request.model or current.get("model") or "", current.get("provider")
    )
    if request.provider is not None and request.provider not in SUPPORTED_PROVIDERS:
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_CONTENT,
            f"unknown provider {request.provider!r}; supported: {SUPPORTED_PROVIDERS}",
        )

    updates: dict[str, str | int | float | bool] = {}
    touched_config_keys: list[str] = []
    if request.provider is not None:
        updates["agent.provider"] = request.provider
    if request.model is not None:
        updates["agent.model"] = request.model
        touched_config_keys.append(AGENT_MODEL_KEY)
    if request.api_key is not None:
        if not provider:
            raise HTTPException(
                status.HTTP_422_UNPROCESSABLE_CONTENT,
                "cannot store an API key without a provider (set provider or model first)",
            )
        key_name = f"{provider.replace('-', '_')}_api_key"
        updates[f"agent.{key_name}"] = request.api_key
        touched_config_keys.append(key_name)
    if request.base_url is not None:
        updates["agent.base_url"] = request.base_url
    if request.instructions is not None:
        updates["agent.instructions"] = request.instructions

    if updates:
        set_operator_values(updates)
        # Clear the just-changed bridged keys so the re-bridge below refreshes
        # them (bridge never overwrites an existing in-code value).
        for key in touched_config_keys:
            if molexp.config.get(key) is not None:
                del molexp.config[key]
        bridge_operator_config()
    return _provider_response()


@router.post("/provider/test", response_model=ProviderTestResponse)
def test_provider(request: ProviderUpdateRequest) -> ProviderTestResponse:
    """Preflight the agent stack for the submitted (or stored) model.

    Constructor + credential validation only — no disk writes, no network,
    no LLM call (the honest scope of a settings-page "test" that must never
    spend tokens or mutate state). ``reply`` describes what was verified.
    """
    from molexp.services.operator_config import load_operator_config
    from molexp.services.plan_runtime import PlanPreflightError, preflight_plan_router

    agent = _agent_section(load_operator_config())
    stored_model = agent.get("model")
    model = request.model or (stored_model if isinstance(stored_model, str) else "")
    provider = _provider_of(model or "", request.provider or agent.get("provider"))
    if not model:
        return ProviderTestResponse(
            ok=False,
            provider=provider,
            model="",
            latencyMs=0,
            reply="",
            error="no model configured — set a model first",
        )
    started = time.monotonic()
    try:
        preflight_plan_router(model=model)
    except PlanPreflightError as exc:
        return ProviderTestResponse(
            ok=False,
            provider=provider,
            model=model,
            latencyMs=int((time.monotonic() - started) * 1000),
            reply="",
            error=str(exc),
        )
    return ProviderTestResponse(
        ok=True,
        provider=provider,
        model=model,
        latencyMs=int((time.monotonic() - started) * 1000),
        reply="preflight passed — agent stack importable, model accepted, credentials resolved",
        error=None,
    )


# ── MCP servers — list/create; replace/delete land in a later slice ─────────


class McpServerResponse(BaseModel):
    """One MCP server row (no secret values — names/refs only)."""

    name: str
    scope: str
    transport: str
    command: str | None
    args: list[str]
    url: str | None
    envKeys: list[str]
    headerKeys: list[str]
    secretRefs: list[str]
    unresolvedSecrets: list[str]
    shadowed: bool
    valid: bool
    invalidReason: str
    auth: dict[str, Any] | None = None


class McpServerListResponse(BaseModel):
    workspacePath: str
    userPath: str
    servers: list[McpServerResponse]


class McpServerUpsertRequest(BaseModel):
    name: str
    scope: str = "user"
    spec: dict[str, Any]


def _to_mcp_response(entry: McpServerEntry) -> McpServerResponse:
    return McpServerResponse(
        name=entry.name,
        scope=str(entry.scope),
        transport=entry.transport,
        command=entry.command,
        args=list(entry.args),
        url=entry.url,
        envKeys=list(entry.env_keys),
        headerKeys=list(entry.header_keys),
        secretRefs=list(entry.secret_refs),
        unresolvedSecrets=list(entry.unresolved_secrets),
        shadowed=entry.shadowed,
        valid=entry.valid,
        invalidReason=entry.invalid_reason,
        auth=entry.auth.model_dump(mode="json") if entry.auth is not None else None,
    )


def _mcp_store(workspace: Workspace) -> McpStore:
    from molexp.agent.mcp.store import McpStore

    return McpStore(str(workspace.root))


@router.get("/mcp/servers", response_model=McpServerListResponse)
def list_mcp_servers(workspace: Workspace = Depends(get_workspace)) -> McpServerListResponse:
    """Merged user + workspace MCP server entries (workspace shadows user)."""
    from molexp.agent.mcp.store import McpScope

    store = _mcp_store(workspace)
    return McpServerListResponse(
        workspacePath=str(store.config_path(McpScope.WORKSPACE)),
        userPath=str(store.config_path(McpScope.USER)),
        servers=[_to_mcp_response(entry) for entry in store.list()],
    )


@router.post("/mcp/servers", response_model=McpServerResponse, status_code=status.HTTP_201_CREATED)
def create_mcp_server(
    request: McpServerUpsertRequest,
    workspace: Workspace = Depends(get_workspace),
) -> McpServerResponse:
    """Upsert one MCP server entry at the requested scope."""
    from molexp.agent.mcp.store import McpScope

    store = _mcp_store(workspace)
    try:
        scope = McpScope(request.scope)
        entry = store.upsert(scope, request.name, request.spec)
    except ValueError as exc:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, str(exc)) from exc
    return _to_mcp_response(entry)


__all__ = ["router"]
