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
from molexp.server.schemas import (
    AgentHealthResponse,
    AgentToolListResponse,
    CommandListResponse,
    CommandParseResponse,
    CommandSpec,
    SkillListResponse,
)
from molexp.server.schemas.responses import AgentToolResponse, ToolParameterResponse

if TYPE_CHECKING:
    from molexp.agent.mcp.store import McpServerEntry, McpStore
    from molexp.workspace import Workspace

# NOTE: no ``/api`` here — the router is mounted under the global ``/api``
# prefix by ``create_app``.
router = APIRouter(prefix="/agent", tags=["agent-admin"])

SUPPORTED_PROVIDERS = ["anthropic", "openai", "google", "deepseek", "openai-compatible"]


_BUILTIN_COMMANDS: tuple[CommandSpec, ...] = (
    CommandSpec(
        slashName="plan",
        name="Plan mode",
        description="Toggle plan mode for the next message.",
        parameters=[],
        defaultPlanMode=True,
        isBuiltin=True,
        skillId=None,
    ),
    CommandSpec(
        slashName="clear",
        name="Clear conversation",
        description="Discard the current chat transcript and start fresh.",
        parameters=[],
        defaultPlanMode=False,
        isBuiltin=True,
        skillId=None,
    ),
    CommandSpec(
        slashName="model",
        name="Change model",
        description="Show or change the active model.",
        parameters=[],
        defaultPlanMode=False,
        isBuiltin=True,
        skillId=None,
    ),
    CommandSpec(
        slashName="help",
        name="Help",
        description="Show available commands and a short usage reminder.",
        parameters=[],
        defaultPlanMode=False,
        isBuiltin=True,
        skillId=None,
    ),
)

_BUILTIN_SLASH = frozenset(c.slashName for c in _BUILTIN_COMMANDS)

_PROVIDER_ENV_HINT: dict[str, str] = {
    "anthropic": "ANTHROPIC_API_KEY",
    "openai": "OPENAI_API_KEY",
    "google": "GOOGLE_API_KEY",
    "deepseek": "DEEPSEEK_API_KEY",
    "openai-compatible": "OPENAI_API_KEY",
}


@router.get("/health", response_model=AgentHealthResponse)
def agent_health() -> AgentHealthResponse:
    """Agent readiness for the UI banner — always 200, never the 503 catch-all.

    ``ready=False`` is a normal configuration state (no model / no API key).
    The legacy ``agent.router`` catch-all used to 503 unknown ``/api/agent/*``
    paths, which made the UI treat "missing route" as "stack unavailable"
    and permanently stop probing. This endpoint exists so health is always a
    real JSON readiness document.
    """
    from molexp.services.operator_config import (
        configured_agent_model,
        configured_api_keys,
        load_operator_config,
    )

    config = load_operator_config()
    agent = _agent_section(config)
    model = configured_agent_model(config) or ""
    if not model:
        raw = agent.get("model")
        model = raw if isinstance(raw, str) else ""
    provider = _provider_of(
        model, agent.get("provider") if isinstance(agent.get("provider"), str) else None
    )
    keys = configured_api_keys(config)
    key_name = f"{provider.replace('-', '_')}_api_key" if provider else ""
    key_set = bool(key_name and keys.get(key_name))
    # Also accept an active provider card key under agent.<provider>_api_key
    # via the flat map; openai-compatible often stores under openai_api_key.
    if not key_set and provider:
        for alt in (f"{provider.replace('-', '_')}_api_key", "openai_api_key", "api_key"):
            if keys.get(alt):
                key_set = True
                key_name = alt
                break
    # Live process may already have a bridged key without a disk entry.
    if not key_set and provider:
        import molexp

        for flat in (f"{provider.replace('-', '_')}_api_key", "openai_api_key"):
            val = molexp.config.get(flat)
            if isinstance(val, str) and val.strip():
                key_set = True
                key_name = flat
                break

    env_var = _PROVIDER_ENV_HINT.get(provider, "API_KEY")
    source = "stored" if (model or key_set) else "none"
    if model and key_set:
        return AgentHealthResponse(
            ready=True,
            provider=provider,
            model=model,
            source=source,
            reason="",
            envVar=env_var,
        )
    if not model:
        reason = (
            "No agent model configured. Set one in Agent Settings → Provider "
            "(or `molexp config set agent.model <provider:model>`)."
        )
    else:
        reason = (
            f"No API key configured for provider '{provider or 'unknown'}'. "
            "Save one in Agent Settings → Provider."
        )
    return AgentHealthResponse(
        ready=False,
        provider=provider,
        model=model,
        source=source,
        reason=reason,
        envVar=env_var,
    )


@router.get("/commands", response_model=CommandListResponse)
def list_commands() -> CommandListResponse:
    """Slash-command catalog for the chat composer palette.

    Builtins always ship; skill-backed commands join when skill persistence
    is wired (currently an empty skill catalog is valid).
    """
    # Skills surface is still empty (see list_skills); builtins only for now.
    return CommandListResponse(commands=list(_BUILTIN_COMMANDS))


class _CommandParseRequest(BaseModel):
    raw: str = ""


@router.post("/commands/parse", response_model=CommandParseResponse)
def parse_command(request: _CommandParseRequest) -> CommandParseResponse:
    """Parse a raw slash line into a builtin / skill / error result."""
    raw = (request.raw or "").strip()
    if not raw.startswith("/"):
        return CommandParseResponse(
            kind="error",
            error="Slash commands must start with '/'.",
        )
    tokens = raw[1:].strip().split()
    if not tokens:
        return CommandParseResponse(kind="error", error="Empty slash command.")
    head = tokens[0].lower()
    params: dict[str, str] = {}
    for token in tokens[1:]:
        if "=" not in token:
            return CommandParseResponse(
                kind="error",
                name=head,
                error=f"Argument '{token}' is missing a value. Use the form key=value.",
            )
        key, _, value = token.partition("=")
        params[key] = value.strip().strip('"')
    if head in _BUILTIN_SLASH:
        return CommandParseResponse(
            kind="builtin",
            name=head,
            parameters=params,
            planMode=(head == "plan"),
        )
    return CommandParseResponse(
        kind="error",
        name=head,
        error=f"Unknown command '/{head}'. Define a skill with this slash name first.",
    )


@router.get("/skills", response_model=SkillListResponse)
def list_skills() -> SkillListResponse:
    """Return the configured skill catalog.

    Skill persistence is not wired into this admin service yet.  An empty
    catalog is a valid state, so the read surface must not fall through to
    the legacy agent 503 catch-all.
    """
    return SkillListResponse()


@router.get("/tools", response_model=AgentToolListResponse)
def list_tools() -> AgentToolListResponse:
    """Return agent tools: molexp **builtins** + MCP groups when discovered.

    Builtins (``workspace_ensure``, ``run_land``, ``code_write``, …) are
    always present with ``source="builtin"``. MCP tools attach as
    ``source="mcp:<server>"`` when runtime discovery is connected; until
    then ``mcpGroups`` may be empty without hiding builtins.
    """
    from molexp.agent.ops.builtins import BUILTIN_SOURCE, BUILTIN_TOOLS

    tools = [
        AgentToolResponse(
            name=t.name,
            description=t.description,
            parameters=[
                ToolParameterResponse(name=n, annotation=a, required=req)
                for n, a, req in t.parameters
            ],
            requiresApproval=False,
            source=BUILTIN_SOURCE,
        )
        for t in BUILTIN_TOOLS
    ]
    return AgentToolListResponse(tools=tools, mcpGroups=[])


class TierModelsResponse(BaseModel):
    """Concrete model ids for the router's three semantic cost tiers."""

    cheap: str = ""
    default: str = ""
    heavy: str = ""


class ProviderConfigurationResponse(BaseModel):
    """One provider's credentials (+ legacy per-provider tier models)."""

    provider: str
    models: TierModelsResponse = Field(default_factory=TierModelsResponse)
    baseUrl: str = ""
    apiKeyPreview: str = ""
    apiKeySet: bool = False


class ProviderResponse(BaseModel):
    """The Settings page's provider view — never carries a key value."""

    provider: str
    model: str
    baseUrl: str
    apiKeyPreview: str
    apiKeySet: bool
    instructions: str
    supportedProviders: list[str] = Field(default_factory=lambda: list(SUPPORTED_PROVIDERS))
    #: Global cheap/default/heavy table (full ``provider:model`` ids). Tiers may
    #: come from different providers; this is the Router's source of truth.
    models: TierModelsResponse = Field(default_factory=TierModelsResponse)
    configurations: list[ProviderConfigurationResponse] = Field(default_factory=list)


class ProviderUpdateRequest(BaseModel):
    """PUT body — only submitted fields are written.

    ``models`` is the **global** tier table: each value should be a full
    ``provider:model`` id (or a bare id + ``provider`` for same-provider legacy).
    Credential fields still use ``provider`` + ``api_key`` / ``base_url``.
    """

    provider: str | None = None
    model: str | None = None
    models: dict[str, str] | None = None
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


def _qualified_model(provider: str, model: str) -> str:
    """Turn a provider-local model name into pydantic-ai's provider:model id."""
    value = model.strip()
    return value if ":" in value or not value else f"{provider}:{value}"


def _agent_section(config: dict[str, Any]) -> dict[str, Any]:
    section = config.get("agent")
    return section if isinstance(section, dict) else {}


def _provider_response() -> ProviderResponse:
    from molexp.services.operator_config import configured_agent_models, load_operator_config

    agent = _agent_section(load_operator_config())
    raw_model = agent.get("model")
    model = raw_model if isinstance(raw_model, str) else ""
    provider = _provider_of(model, agent.get("provider"))
    raw_providers = agent.get("providers")
    provider_sections = raw_providers if isinstance(raw_providers, dict) else {}
    resolved = configured_agent_models({"agent": agent}) or {}
    global_models = TierModelsResponse(
        cheap=str(resolved.get("cheap") or model or ""),
        default=str(resolved.get("default") or model or ""),
        heavy=str(resolved.get("heavy") or model or ""),
    )
    configurations: list[ProviderConfigurationResponse] = []
    for name in SUPPORTED_PROVIDERS:
        raw_section = provider_sections.get(name)
        section = raw_section if isinstance(raw_section, dict) else {}
        raw_models = section.get("models")
        models = raw_models if isinstance(raw_models, dict) else {}
        key_name = f"{name.replace('-', '_')}_api_key"
        key = agent.get(key_name)
        key_set = isinstance(key, str) and bool(key)
        configurations.append(
            ProviderConfigurationResponse(
                provider=name,
                models=TierModelsResponse(
                    cheap=str(models.get("cheap") or ""),
                    default=str(models.get("default") or ""),
                    heavy=str(models.get("heavy") or ""),
                ),
                baseUrl=str(
                    section.get("base_url")
                    or (agent.get("base_url") if name == provider else "")
                    or ""
                ),
                apiKeyPreview=_mask_key(key) if key_set else "",
                apiKeySet=key_set,
            )
        )
    active = next((item for item in configurations if item.provider == provider), None)
    return ProviderResponse(
        provider=provider,
        model=model or global_models.default or "",
        baseUrl=active.baseUrl if active is not None else "",
        apiKeyPreview=active.apiKeyPreview if active is not None else "",
        apiKeySet=active.apiKeySet if active is not None else False,
        instructions=agent.get("instructions") or "",
        models=global_models,
        configurations=configurations,
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
        AGENT_MODELS_KEY,
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
    if request.models is not None:
        unknown_tiers = set(request.models) - {"cheap", "default", "heavy"}
        if unknown_tiers:
            raise HTTPException(
                status.HTTP_422_UNPROCESSABLE_CONTENT,
                f"unknown model tiers: {sorted(unknown_tiers)}",
            )
        missing_tiers = {"cheap", "default", "heavy"} - set(request.models)
        if missing_tiers:
            raise HTTPException(
                status.HTTP_422_UNPROCESSABLE_CONTENT,
                f"missing model tiers: {sorted(missing_tiers)}",
            )
        # Global tier table: each value may already be ``provider:model``
        # (cross-provider). Bare ids still need a fallback ``provider``.
        qualified: dict[str, str] = {}
        for tier, tier_model in request.models.items():
            text = str(tier_model).strip()
            if ":" in text:
                qualified[tier] = text
            elif provider:
                qualified[tier] = _qualified_model(provider, text)
            else:
                raise HTTPException(
                    status.HTTP_422_UNPROCESSABLE_CONTENT,
                    f"tier {tier!r} model {text!r} needs a provider: prefix "
                    "or a top-level provider field",
                )
            updates[f"agent.models.{tier}"] = qualified[tier]
        default_id = qualified["default"]
        updates["agent.model"] = default_id
        updates["agent.provider"] = _provider_of(default_id, None) or provider or ""
        touched_config_keys.extend((AGENT_MODEL_KEY, AGENT_MODELS_KEY))
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
        if provider:
            updates[f"agent.providers.{provider}.base_url"] = request.base_url
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
    requested_default = request.models.get("default") if request.models is not None else None
    model = (
        requested_default
        or request.model
        or (stored_model if isinstance(stored_model, str) else "")
    )
    provider = _provider_of(model or "", request.provider or agent.get("provider"))
    model = _qualified_model(provider, model)
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
    #: Non-secret env literals for the editor (e.g. ``MOLMCP_SOURCES``).
    env: dict[str, str] = Field(default_factory=dict)
    headerKeys: list[str]
    secretRefs: list[str]
    unresolvedSecrets: list[str]
    shadowed: bool
    valid: bool
    invalidReason: str
    auth: dict[str, Any] | None = None
    #: Parsed ``MOLMCP_SOURCES`` when this is a molmcp-class server.
    knowledgeSources: list[str] = Field(default_factory=list)


class McpServerListResponse(BaseModel):
    workspacePath: str
    userPath: str
    servers: list[McpServerResponse]


class McpServerUpsertRequest(BaseModel):
    name: str
    scope: str = "user"
    spec: dict[str, Any]


def _parse_molmcp_sources(raw: str | None) -> list[str]:
    if not raw or not str(raw).strip():
        return []
    return [p.strip() for p in str(raw).replace(";", ",").split(",") if p.strip()]


def _is_molmcp_server(name: str) -> bool:
    n = name.lower()
    return n == "molmcp" or "molmcp" in n


def _to_mcp_response(store: McpStore, entry: McpServerEntry) -> McpServerResponse:
    from molexp.agent.mcp.store import McpScope

    try:
        scope = McpScope(str(entry.scope))
    except ValueError:
        scope = McpScope.USER
    public_env = store.public_env(scope, entry.name)
    sources = (
        _parse_molmcp_sources(public_env.get("MOLMCP_SOURCES"))
        if _is_molmcp_server(entry.name)
        else []
    )
    return McpServerResponse(
        name=entry.name,
        scope=str(entry.scope),
        transport=entry.transport,
        command=entry.command,
        args=list(entry.args),
        url=entry.url,
        envKeys=list(entry.env_keys),
        env=public_env,
        headerKeys=list(entry.header_keys),
        secretRefs=list(entry.secret_refs),
        unresolvedSecrets=list(entry.unresolved_secrets),
        shadowed=entry.shadowed,
        valid=entry.valid,
        invalidReason=entry.invalid_reason,
        auth=entry.auth.model_dump(mode="json") if entry.auth is not None else None,
        knowledgeSources=sources,
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
        servers=[_to_mcp_response(store, entry) for entry in store.list()],
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
    return _to_mcp_response(store, entry)


@router.put("/mcp/servers/{name}", response_model=McpServerResponse)
def replace_mcp_server(
    name: str,
    request: McpServerUpsertRequest,
    workspace: Workspace = Depends(get_workspace),
) -> McpServerResponse:
    """Replace one MCP server entry (name path must match body)."""
    from molexp.agent.mcp.store import McpScope

    if request.name and request.name != name:
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            f"path name {name!r} does not match body name {request.name!r}",
        )
    store = _mcp_store(workspace)
    try:
        scope = McpScope(request.scope)
        entry = store.upsert(scope, name, request.spec)
    except ValueError as exc:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, str(exc)) from exc
    return _to_mcp_response(store, entry)


@router.delete("/mcp/servers/{name}", status_code=status.HTTP_204_NO_CONTENT)
def delete_mcp_server(
    name: str,
    scope: str = "user",
    workspace: Workspace = Depends(get_workspace),
) -> None:
    """Delete one MCP server entry at the given scope."""
    from molexp.agent.mcp.store import McpScope

    store = _mcp_store(workspace)
    try:
        mcp_scope = McpScope(scope)
    except ValueError as exc:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, str(exc)) from exc
    if not store.delete(mcp_scope, name):
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"MCP server {name!r} not found")


class McpServerTestResponse(BaseModel):
    ok: bool
    name: str
    scope: str
    transport: str
    latencyMs: int = 0
    toolCount: int = 0
    error: str | None = None


@router.post("/mcp/servers/{name}/test", response_model=McpServerTestResponse)
async def test_mcp_server(
    name: str,
    scope: str = "user",
    workspace: Workspace = Depends(get_workspace),
) -> McpServerTestResponse:
    """Best-effort stdio/HTTP reachability probe (list_tools when possible)."""
    from molexp.agent.mcp.store import McpScope

    store = _mcp_store(workspace)
    try:
        mcp_scope = McpScope(scope)
    except ValueError as exc:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, str(exc)) from exc
    entry = store.get(mcp_scope, name)
    if entry is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"MCP server {name!r} not found")
    transport = entry.transport or "?"
    started = time.monotonic()
    tool_count = 0
    err: str | None = None
    try:
        resolved = store.resolve(entry)
        if resolved.transport == "stdio" and resolved.command:
            import os
            from pathlib import Path

            from mcp import ClientSession, StdioServerParameters
            from mcp.client.stdio import stdio_client

            params = StdioServerParameters(
                command=resolved.command,
                args=list(resolved.args or ()),
                env=dict(resolved.env or {}) or None,
            )
            with Path(os.devnull).open("w", encoding="utf-8") as errlog:
                async with (
                    stdio_client(params, errlog=errlog) as (read, write),
                    ClientSession(read, write) as session,
                ):
                    await session.initialize()
                    listed = await session.list_tools()
                    tools = getattr(listed, "tools", None) or []
                    tool_count = len(tools)
        else:
            # HTTP/SSE: config-only smoke (full client needs auth plumbing).
            tool_count = 0
    except Exception as exc:  # probe must never 500 the settings page
        err = f"{type(exc).__name__}: {exc}"
    latency = int((time.monotonic() - started) * 1000)
    return McpServerTestResponse(
        ok=err is None,
        name=name,
        scope=scope,
        transport=str(transport),
        latencyMs=latency,
        toolCount=tool_count,
        error=err,
    )


class McpSecretListResponse(BaseModel):
    scope: str
    keys: list[str]
    path: str


class McpSecretPutRequest(BaseModel):
    scope: str = "user"
    value: str = ""


@router.get("/mcp/secrets", response_model=McpSecretListResponse)
def list_mcp_secrets(
    scope: str = "user",
    workspace: Workspace = Depends(get_workspace),
) -> McpSecretListResponse:
    """List secret *keys* (never values) at the given scope."""
    from molexp.agent.mcp.store import McpScope

    store = _mcp_store(workspace)
    try:
        mcp_scope = McpScope(scope)
    except ValueError as exc:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, str(exc)) from exc
    secrets = store.secrets(mcp_scope)
    return McpSecretListResponse(
        scope=scope,
        keys=list(secrets.list_keys()),
        path=str(secrets.path),
    )


@router.put("/mcp/secrets/{key}")
def put_mcp_secret(
    key: str,
    request: McpSecretPutRequest,
    workspace: Workspace = Depends(get_workspace),
) -> dict[str, Any]:
    """Set or delete a secret value (empty value deletes)."""
    from molexp.agent.mcp.store import McpScope

    store = _mcp_store(workspace)
    try:
        mcp_scope = McpScope(request.scope)
    except ValueError as exc:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, str(exc)) from exc
    secrets = store.secrets(mcp_scope)
    if not request.value:
        secrets.delete(key)
        return {"ok": True, "key": key, "deleted": True}
    secrets.set(key, request.value)
    return {"ok": True, "key": key, "deleted": False}


# ── Knowledge sources — per-MCP (molmcp) package allowlist ──────────────────


#: Packages operators commonly pin for plan grounding (UI chips).
_KNOWN_KNOWLEDGE_SOURCES: tuple[str, ...] = (
    "molpy",
    "molpack",
    "molvis",
    "molplot",
    "molq",
    "molcfg",
    "atomiverse",
    "lammps",
)


class KnowledgeSourcesResponse(BaseModel):
    """Package scope for the **molmcp** MCP server entry (not a global agent key).

    Stored as ``env.MOLMCP_SOURCES`` on that server's config row. Plan sessions
    and capability grounding read the same pin from the MCP store.
    """

    sources: list[str] = Field(default_factory=list)
    knownPackages: list[str] = Field(default_factory=lambda: list(_KNOWN_KNOWLEDGE_SOURCES))
    unrestricted: bool = True
    serverName: str = "molmcp"
    scope: str = "user"
    configured: bool = False


class KnowledgeSourcesUpdateRequest(BaseModel):
    """PUT body — empty list clears the pin (unrestricted)."""

    sources: list[str] = Field(default_factory=list)
    #: Which MCP scope owns the molmcp entry (default: active workspace row).
    scope: str | None = None


def _find_molmcp_entry(store: McpStore) -> McpServerEntry | None:
    """Prefer workspace molmcp, then user; match name ``molmcp`` or *molmcp*."""
    from molexp.agent.mcp.store import McpScope

    listed = store.list()
    for preferred in ("molmcp",):
        for scope in (McpScope.WORKSPACE, McpScope.USER):
            entry = store.get(scope, preferred)
            if entry is not None and not entry.shadowed:
                return entry
    for entry in listed:
        if _is_molmcp_server(entry.name) and not entry.shadowed:
            return entry
    return None


def _knowledge_sources_response(workspace: Workspace) -> KnowledgeSourcesResponse:
    store = _mcp_store(workspace)
    entry = _find_molmcp_entry(store)
    if entry is None:
        return KnowledgeSourcesResponse(
            sources=[],
            knownPackages=list(_KNOWN_KNOWLEDGE_SOURCES),
            unrestricted=True,
            serverName="molmcp",
            scope="user",
            configured=False,
        )
    from molexp.agent.mcp.store import McpScope

    try:
        scope = McpScope(str(entry.scope))
    except ValueError:
        scope = McpScope.USER
    env = store.public_env(scope, entry.name)
    sources = _parse_molmcp_sources(env.get("MOLMCP_SOURCES"))
    return KnowledgeSourcesResponse(
        sources=sources,
        knownPackages=list(_KNOWN_KNOWLEDGE_SOURCES),
        unrestricted=len(sources) == 0,
        serverName=entry.name,
        scope=str(entry.scope),
        configured=True,
    )


@router.get("/knowledge-sources", response_model=KnowledgeSourcesResponse)
def get_knowledge_sources(
    workspace: Workspace = Depends(get_workspace),
) -> KnowledgeSourcesResponse:
    """Read package pin from the molmcp MCP server entry (``MOLMCP_SOURCES``)."""
    return _knowledge_sources_response(workspace)


@router.put("/knowledge-sources", response_model=KnowledgeSourcesResponse)
def update_knowledge_sources(
    request: KnowledgeSourcesUpdateRequest,
    workspace: Workspace = Depends(get_workspace),
) -> KnowledgeSourcesResponse:
    """Write package pin onto the molmcp server's env (per-MCP, not global agent)."""
    from molexp._typing import JSONValue
    from molexp.agent.mcp.store import McpScope

    store = _mcp_store(workspace)
    entry = _find_molmcp_entry(store)
    if entry is None:
        raise HTTPException(
            status.HTTP_404_NOT_FOUND,
            "No molmcp MCP server configured. Add one under Agent Settings → MCP "
            "(name: molmcp), then set knowledge sources.",
        )
    if request.scope is not None:
        try:
            scope = McpScope(request.scope)
        except ValueError as exc:
            raise HTTPException(status.HTTP_400_BAD_REQUEST, str(exc)) from exc
    else:
        try:
            scope = McpScope(str(entry.scope))
        except ValueError:
            scope = McpScope.USER
    # Rebuild stdio spec with updated MOLMCP_SOURCES; preserve other public env.
    public = store.public_env(scope, entry.name)
    cleaned = [str(s).strip().lower() for s in request.sources if str(s).strip()]
    seen: set[str] = set()
    ordered: list[str] = []
    for s in cleaned:
        if s not in seen:
            seen.add(s)
            ordered.append(s)
    env: dict[str, JSONValue] = dict(public)
    if ordered:
        env["MOLMCP_SOURCES"] = ",".join(ordered)
    else:
        env.pop("MOLMCP_SOURCES", None)
    # Keep secret placeholders for secret env keys we don't return publicly.
    for key in entry.env_keys:
        if key not in env and key in entry.secret_refs:
            env[key] = f"${{SECRET:{key}}}"
        elif key not in env and key != "MOLMCP_SOURCES" and key in entry.secret_refs:
            # unknown non-public key — leave as secret ref if it was secret-like
            env[key] = f"${{SECRET:{key}}}"
    if entry.transport != "stdio" or not entry.command:
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            "molmcp must be a stdio MCP server to pin MOLMCP_SOURCES",
        )
    spec: dict[str, JSONValue] = {
        "type": "stdio",
        "command": entry.command,
        "args": list(entry.args),
        "env": env,
    }
    if entry.usage_instructions:
        spec["usage_instructions"] = entry.usage_instructions
    store.upsert(scope, entry.name, spec)
    return _knowledge_sources_response(workspace)


@router.get("/admin/providers")
def list_admin_providers() -> dict[str, Any]:
    """Provider form registry for Settings (bootstrap schema; never 503)."""
    return {
        "providers": [
            {
                "name": name,
                "label": {
                    "anthropic": "Anthropic (Claude)",
                    "openai": "OpenAI",
                    "google": "Google (Gemini)",
                    "deepseek": "DeepSeek",
                    "openai-compatible": "OpenAI-compatible (proxy / Ollama / vLLM)",
                }.get(name, name),
                "modelHint": "provider:model id or bare model name",
                "fields": [
                    {"key": "api_key", "label": "API key", "kind": "secret", "required": True},
                    {"key": "model", "label": "Model", "kind": "text", "required": True},
                    *(
                        [
                            {
                                "key": "base_url",
                                "label": "Base URL",
                                "kind": "url",
                                "required": True,
                                "placeholder": "http://localhost:11434/v1",
                            }
                        ]
                        if name == "openai-compatible"
                        else []
                    ),
                ],
            }
            for name in SUPPORTED_PROVIDERS
        ]
    }


__all__ = ["router"]
