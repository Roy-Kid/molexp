"""Management endpoints for the agent: MCP servers/secrets, tools, skills.

UI surfaces (settings panels, skill libraries) hit these to manage the
workspace's agent config without loading the live agent. Skills and the
MCP server/secret stores support full CRUD; the native tool catalog and
provider health are read-only.
"""

from __future__ import annotations

from typing import Literal

from fastapi import APIRouter, Depends, HTTPException, Query

from molexp.plugins.agent_pydanticai.admin import describe_native_tools
from molexp.plugins.agent_pydanticai.commands import parse as parse_slash
from molexp.plugins.agent_pydanticai.mcp_oauth import (
    OAuthFlowSession,
    START_TIMEOUT_SECONDS,
    build_oauth_provider,
    default_redirect_uri,
    session_registry,
    storage_for,
)
from molexp.plugins.agent_pydanticai.mcp_probe import list_mcp_tools, probe_server
from molexp.plugins.agent_pydanticai.mcp_store import McpScope, McpStore
from molexp.plugins.agent_pydanticai.provider import (
    DEFAULT_MODELS,
    SUPPORTED_PROVIDERS,
    ProviderConfig,
    ProviderStore,
    check_credentials,
    probe_provider,
    to_public,
)
from molexp.plugins.agent_pydanticai.skills import RESERVED_SLASH_NAMES, SkillStore

from ..dependencies import get_workspace
from ..schemas import (
    AgentHealthResponse,
    AgentProviderResponse,
    AgentProviderTestResponse,
    AgentProviderUpdateRequest,
    AgentToolListResponse,
    AgentToolResponse,
    CommandListResponse,
    CommandParameterSpec,
    CommandParseRequest,
    CommandParseResponse,
    CommandSpec,
    McpAuthSummary,
    McpOAuthCallbackRequest,
    McpOAuthStartResponse,
    McpOAuthStatusResponse,
    McpSecretListResponse,
    McpSecretRefRow,
    McpSecretSetRequest,
    McpServerListResponse,
    McpServerResponse,
    McpServerTestResponse,
    McpServerUpsertRequest,
    McpToolGroupResponse,
    MessageResponse,
    SkillCreateRequest,
    SkillListResponse,
    SkillResponse,
    SkillUpdateRequest,
    ToolParameterResponse,
)

router = APIRouter(prefix="/agent", tags=["agent-admin"])


# ── MCP servers ─────────────────────────────────────────────────────────────


def _mcp_store(workspace) -> McpStore:
    root = getattr(workspace, "root", None)
    if root is None:
        raise HTTPException(status_code=500, detail="Workspace has no root path")
    return McpStore(root)


def _entry_to_response(entry, *, store: McpStore | None = None) -> McpServerResponse:
    auth_summary: McpAuthSummary | None = None
    if entry.auth is not None and entry.auth.type == "oauth2":
        connected = False
        if store is not None:
            connected = storage_for(store, entry.scope, entry.name).path.exists()
        auth_summary = McpAuthSummary(
            type="oauth2",
            scopes=list(entry.auth.scopes),
            clientId=entry.auth.client_id,
            connected=connected,
        )
    return McpServerResponse(
        name=entry.name,
        scope=entry.scope.value,
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
        auth=auth_summary,
    )


@router.get("/mcp/servers", response_model=McpServerListResponse)
async def get_mcp_servers(workspace=Depends(get_workspace)) -> McpServerListResponse:
    """Return merged User+Workspace MCP servers, including shadowed entries."""
    store = _mcp_store(workspace)
    return McpServerListResponse(
        workspacePath=str(store.config_path(McpScope.WORKSPACE)),
        userPath=str(store.config_path(McpScope.USER)),
        servers=[_entry_to_response(e, store=store) for e in store.list()],
    )


@router.post("/mcp/servers", response_model=McpServerResponse, status_code=201)
async def create_mcp_server(
    request: McpServerUpsertRequest,
    workspace=Depends(get_workspace),
) -> McpServerResponse:
    """Create an MCP server entry at ``request.scope``."""
    store = _mcp_store(workspace)
    if store.get(McpScope(request.scope), request.name) is not None:
        raise HTTPException(
            status_code=409,
            detail=(
                f"MCP server '{request.name}' already exists at scope "
                f"'{request.scope}'. Use PUT to replace it."
            ),
        )
    try:
        entry = store.upsert(
            McpScope(request.scope),
            request.name,
            request.spec.model_dump(),
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return _entry_to_response(entry, store=store)


@router.put("/mcp/servers/{name}", response_model=McpServerResponse)
async def replace_mcp_server(
    name: str,
    request: McpServerUpsertRequest,
    workspace=Depends(get_workspace),
) -> McpServerResponse:
    """Fully replace an MCP server entry. ``request.name`` must match the path."""
    if request.name != name:
        raise HTTPException(
            status_code=400,
            detail=f"Path name '{name}' does not match body name '{request.name}'.",
        )
    store = _mcp_store(workspace)
    try:
        entry = store.upsert(
            McpScope(request.scope),
            name,
            request.spec.model_dump(),
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return _entry_to_response(entry, store=store)


@router.delete("/mcp/servers/{name}", response_model=MessageResponse)
async def delete_mcp_server(
    name: str,
    scope: Literal["user", "workspace"] = Query(
        "workspace",
        description="Which scope to delete from.",
    ),
    workspace=Depends(get_workspace),
) -> MessageResponse:
    store = _mcp_store(workspace)
    deleted = store.delete(McpScope(scope), name)
    if not deleted:
        raise HTTPException(
            status_code=404,
            detail=f"MCP server '{name}' not found at scope '{scope}'.",
        )
    return MessageResponse(message=f"MCP server '{name}' deleted from {scope}.")


@router.post("/mcp/servers/{name}/test", response_model=McpServerTestResponse)
async def test_mcp_server(
    name: str,
    scope: Literal["user", "workspace"] = Query(
        "workspace",
        description="Which scope's entry to probe.",
    ),
    workspace=Depends(get_workspace),
) -> McpServerTestResponse:
    """Open a real connection to the server, list its tools, then disconnect.

    Bounded by a 10-second hard timeout. The test result never includes
    secret values — only the resolved tool count + connection metrics.
    """
    store = _mcp_store(workspace)
    entry = store.get(McpScope(scope), name)
    if entry is None:
        raise HTTPException(
            status_code=404,
            detail=f"MCP server '{name}' not found at scope '{scope}'.",
        )
    outcome = await probe_server(store, entry)
    return McpServerTestResponse(
        ok=outcome.ok,
        name=name,
        scope=scope,
        transport=entry.transport,
        latencyMs=outcome.latency_ms,
        toolCount=outcome.tool_count,
        error=outcome.error,
    )


# ── MCP OAuth flow ──────────────────────────────────────────────────────────


def _require_oauth_entry(store: McpStore, scope: McpScope, name: str):
    """Look up an entry and verify it's configured for OAuth.

    Raises HTTPException(404) when the entry doesn't exist and 400 when
    found but not OAuth-shaped — the routes only make sense for OAuth specs.
    """
    entry = store.get(scope, name)
    if entry is None:
        raise HTTPException(
            status_code=404,
            detail=f"MCP server '{name}' not found at scope '{scope.value}'.",
        )
    if entry.auth is None or entry.auth.type != "oauth2":
        raise HTTPException(
            status_code=400,
            detail=(
                f"MCP server '{name}' is not configured for OAuth. "
                "Edit the server and pick OAuth 2.0 as the auth method."
            ),
        )
    return entry


@router.post(
    "/mcp/servers/{name}/oauth/start",
    response_model=McpOAuthStartResponse,
)
async def start_mcp_oauth(
    name: str,
    scope: Literal["user", "workspace"] = Query("workspace"),
    workspace=Depends(get_workspace),
) -> McpOAuthStartResponse:
    """Begin the OAuth 2.0 + PKCE flow for an MCP server.

    Spawns a background task that drives the SDK's
    ``OAuthClientProvider`` until it produces an authorize URL, then
    returns that URL. The browser opens it; once the IdP bounces back to
    the SPA, the SPA POSTs the code/state to ``/oauth/callback`` to
    complete token exchange.

    Replaces any in-flight session for this server (clicking Connect
    twice cancels the older flow rather than queueing).
    """
    import asyncio
    from contextlib import suppress

    import httpx

    store = _mcp_store(workspace)
    mcp_scope = McpScope(scope)
    entry = _require_oauth_entry(store, mcp_scope, name)

    # Preflight: confirm the resource server actually advertises OAuth.
    # Many MCP servers are public (no OAuth metadata) — kicking off
    # OAuthClientProvider against one of those produces a confusing 400/404
    # buried inside the SDK's internal discovery flow.
    if not await _has_oauth_metadata(entry.url or ""):
        raise HTTPException(
            status_code=400,
            detail=(
                f"{entry.url} does not advertise OAuth metadata "
                "(.well-known/oauth-protected-resource → 404). "
                "This server is public — change Authentication to None or "
                "Custom Headers, or point at an OAuth-protected MCP."
            ),
        )

    session = session_registry().create(mcp_scope, name)
    storage = storage_for(store, mcp_scope, name)
    provider = build_oauth_provider(
        server_url=entry.url or "",
        redirect_uri=default_redirect_uri(),
        scopes=list(entry.auth.scopes),
        storage=storage,
        session=session,
        client_name=f"molexp:{name}",
    )

    async def drive_flow() -> None:
        """Sign a single dummy request to make ``OAuthClientProvider`` kick
        off the flow. GET + ``Accept: text/event-stream`` matches the MCP
        streamable-HTTP wire format; HEAD gets rejected by HF Space-style
        gateways. The ``finally`` evicts the session whether the user
        completed, abandoned, or hit the 5-minute SDK callback timeout —
        otherwise the registry would leak entries forever.
        """
        try:
            async with httpx.AsyncClient(auth=provider, timeout=30.0) as client:
                with suppress(Exception):
                    await client.get(
                        entry.url or "",
                        headers={"Accept": "text/event-stream"},
                    )
        finally:
            session_registry().discard(mcp_scope, name)

    flow_task = asyncio.create_task(drive_flow(), name=f"mcp-oauth-{name}")

    try:
        authorize_url = await asyncio.wait_for(
            session.authorize_url_future, timeout=START_TIMEOUT_SECONDS
        )
    except TimeoutError as exc:
        flow_task.cancel()
        session_registry().discard(mcp_scope, name)
        raise HTTPException(
            status_code=504,
            detail=(
                f"OAuth issuer at {entry.url} did not respond within "
                f"{START_TIMEOUT_SECONDS:.0f}s — check the URL and network."
            ),
        ) from exc
    except Exception as exc:
        flow_task.cancel()
        session_registry().discard(mcp_scope, name)
        raise HTTPException(
            status_code=502,
            detail=f"Failed to start OAuth flow: {exc}",
        ) from exc

    session.flow_task = flow_task

    return McpOAuthStartResponse(
        name=name,
        scope=scope,
        authorizeUrl=authorize_url,
    )


@router.post(
    "/mcp/servers/{name}/oauth/callback",
    response_model=McpOAuthStatusResponse,
)
async def callback_mcp_oauth(
    name: str,
    request: McpOAuthCallbackRequest,
    scope: Literal["user", "workspace"] = Query("workspace"),
    workspace=Depends(get_workspace),
) -> McpOAuthStatusResponse:
    """Complete an in-flight OAuth flow with the IdP's callback payload.

    Fed by the SPA after the IdP redirects the browser back to
    ``/oauth-callback``. We hand the ``(code, state)`` to the awaiting SDK
    callback_handler, await flow completion, then return the new
    connection status.
    """
    import asyncio

    store = _mcp_store(workspace)
    mcp_scope = McpScope(scope)
    _require_oauth_entry(store, mcp_scope, name)

    session = session_registry().get(mcp_scope, name)
    if session is None:
        raise HTTPException(
            status_code=410,
            detail=(
                "No in-flight OAuth flow for this server. The flow may "
                "have timed out or been superseded — click Connect again."
            ),
        )

    delivered = session.submit_callback(request.code, request.state)
    if not delivered:
        raise HTTPException(
            status_code=410,
            detail="OAuth callback was already submitted for this flow.",
        )

    flow_task = session.flow_task
    if flow_task is not None:
        with_suppress = asyncio.wait_for(flow_task, timeout=60.0)
        try:
            await with_suppress
        except TimeoutError:
            session_registry().discard(mcp_scope, name)
            raise HTTPException(
                status_code=504,
                detail="OAuth token exchange did not complete within 60s.",
            ) from None
        except Exception as exc:
            session_registry().discard(mcp_scope, name)
            raise HTTPException(
                status_code=502,
                detail=f"OAuth token exchange failed: {exc}",
            ) from exc

    session_registry().discard(mcp_scope, name)

    storage = storage_for(store, mcp_scope, name)
    entry = store.get(mcp_scope, name)
    return McpOAuthStatusResponse(
        name=name,
        scope=scope,
        hasTokens=storage.path.exists(),
        scopes=list(entry.auth.scopes) if entry and entry.auth else [],
    )


@router.get(
    "/mcp/servers/{name}/oauth",
    response_model=McpOAuthStatusResponse,
)
async def get_mcp_oauth_status(
    name: str,
    scope: Literal["user", "workspace"] = Query("workspace"),
    workspace=Depends(get_workspace),
) -> McpOAuthStatusResponse:
    """Whether the named server has a usable OAuth token on disk."""
    store = _mcp_store(workspace)
    mcp_scope = McpScope(scope)
    entry = _require_oauth_entry(store, mcp_scope, name)
    storage = storage_for(store, mcp_scope, name)
    return McpOAuthStatusResponse(
        name=name,
        scope=scope,
        hasTokens=storage.path.exists(),
        scopes=list(entry.auth.scopes) if entry.auth else [],
    )


@router.delete(
    "/mcp/servers/{name}/oauth",
    response_model=MessageResponse,
)
async def disconnect_mcp_oauth(
    name: str,
    scope: Literal["user", "workspace"] = Query("workspace"),
    workspace=Depends(get_workspace),
) -> MessageResponse:
    """Drop stored OAuth tokens for a server. Idempotent.

    Equivalent to "log out" — the spec stays in place, so a future Connect
    walks the user through PKCE again. Any in-flight flow is cancelled.
    """
    store = _mcp_store(workspace)
    mcp_scope = McpScope(scope)
    _require_oauth_entry(store, mcp_scope, name)

    in_flight = session_registry().get(mcp_scope, name)
    if in_flight is not None:
        in_flight.cancel("disconnected")
        session_registry().discard(mcp_scope, name)

    cleared = storage_for(store, mcp_scope, name).clear()
    msg = (
        f"OAuth tokens cleared for '{name}'."
        if cleared
        else f"No OAuth tokens were stored for '{name}'."
    )
    return MessageResponse(message=msg)


async def _has_oauth_metadata(server_url: str) -> bool:
    """Whether ``server_url`` advertises RFC 9728 protected-resource metadata.

    Probes ``<origin>/.well-known/oauth-protected-resource`` with a 5-second
    cap. Returns True on 2xx, False on 4xx/network error. Used purely as a
    preflight to give a clear error before kicking off ``OAuthClientProvider``
    against a server that has no OAuth at all.
    """
    if not server_url:
        return False
    import httpx

    # Origin-level discovery per RFC 9728. Some servers may also host the
    # metadata at the resource path, but the origin form is what every
    # production OAuth-protected MCP we've seen (FastMCP, etc.) uses, and
    # what the SDK probes first.
    parsed = httpx.URL(server_url)
    metadata_url = httpx.URL(
        scheme=parsed.scheme,
        host=parsed.host,
        port=parsed.port,
        path="/.well-known/oauth-protected-resource",
    )
    try:
        async with httpx.AsyncClient(timeout=5.0, follow_redirects=True) as client:
            resp = await client.get(str(metadata_url))
    except httpx.RequestError:
        return False
    return 200 <= resp.status_code < 300


# ── MCP secrets ─────────────────────────────────────────────────────────────


@router.get("/mcp/secrets", response_model=McpSecretListResponse)
async def list_mcp_secrets(
    scope: Literal["user", "workspace"] = Query(
        "workspace",
        description="Which secret store to inspect.",
    ),
    workspace=Depends(get_workspace),
) -> McpSecretListResponse:
    """List secret keys at ``scope`` plus which servers reference them.

    Plaintext values are **never** returned. ``isSet`` is the only signal
    of whether the value exists for a referenced key.
    """
    store = _mcp_store(workspace)
    scope_enum = McpScope(scope)
    refs = store.secret_references(scope_enum)
    secret_store = store.secrets(scope_enum)
    set_keys = set(secret_store.list_keys())
    all_keys = sorted(set_keys | refs.keys())
    return McpSecretListResponse(
        scope=scope,
        path=str(secret_store.path),
        secrets=[
            McpSecretRefRow(
                key=key,
                isSet=key in set_keys,
                referencedBy=refs.get(key, []),
            )
            for key in all_keys
        ],
    )


@router.put("/mcp/secrets/{key}", response_model=MessageResponse)
async def set_mcp_secret(
    key: str,
    request: McpSecretSetRequest,
    workspace=Depends(get_workspace),
) -> MessageResponse:
    """Write a secret value at ``request.scope``. Empty value deletes the key."""
    store = _mcp_store(workspace)
    secret_store = store.secrets(McpScope(request.scope))
    secret_store.set(key, request.value)
    if request.value == "":
        return MessageResponse(message=f"Secret '{key}' cleared at {request.scope}.")
    return MessageResponse(message=f"Secret '{key}' saved at {request.scope}.")


# ── Native agent tools ───────────────────────────────────────────────────────


@router.get("/tools", response_model=AgentToolListResponse)
async def get_agent_tools(workspace=Depends(get_workspace)) -> AgentToolListResponse:
    """List tools the agent will see — native + each configured MCP server.

    Native tools always come back instantly. MCP servers are probed in
    parallel; each server contributes (a) its tools (with ``source =
    "mcp:<name>"`` so the UI can group) and (b) a row in ``mcpGroups``
    capturing per-server connection status. A failed probe is not fatal —
    its group surfaces ``ok=False`` + ``error`` so users can fix it
    without losing the rest of the listing.
    """
    rows = describe_native_tools()
    native_tools = [
        AgentToolResponse(
            name=row["name"],
            description=row["description"],
            parameters=[
                ToolParameterResponse(
                    name=p["name"],
                    annotation=p["annotation"],
                    required=p["required"],
                )
                for p in row["parameters"]
            ],
            requiresApproval=row["requires_approval"],
            source=row["source"],
        )
        for row in rows
    ]

    store = _mcp_store(workspace)
    mcp_results = await list_mcp_tools(store)
    mcp_tools: list[AgentToolResponse] = []
    mcp_groups: list[McpToolGroupResponse] = []
    for grp in mcp_results:
        mcp_groups.append(
            McpToolGroupResponse(
                server=grp.server,
                scope=grp.scope,  # type: ignore[arg-type]
                ok=grp.ok,
                toolCount=len(grp.tools),
                error=grp.error,
            )
        )
        for t in grp.tools:
            mcp_tools.append(
                AgentToolResponse(
                    name=t.name,
                    description=t.description,
                    parameters=[],
                    requiresApproval=False,
                    source=f"mcp:{grp.server}",
                )
            )

    return AgentToolListResponse(
        tools=native_tools + mcp_tools,
        mcpGroups=mcp_groups,
    )


# ── Skills ───────────────────────────────────────────────────────────────────


def _skill_store(workspace) -> SkillStore:
    root = getattr(workspace, "root", None)
    if root is None:
        raise HTTPException(status_code=500, detail="Workspace has no root path")
    return SkillStore(root)


def _to_response(skill) -> SkillResponse:
    return SkillResponse(
        id=skill.id,
        name=skill.name,
        description=skill.description,
        goalTemplate=skill.goal_template,
        slashName=skill.slash_name,
        instructions=skill.instructions,
        defaultPlanMode=skill.default_plan_mode,
        constraints=list(skill.constraints),
        successCriteria=list(skill.success_criteria),
        tags=list(skill.tags),
        createdAt=skill.created_at,
        updatedAt=skill.updated_at,
    )


@router.get("/skills", response_model=SkillListResponse)
async def list_skills(workspace=Depends(get_workspace)) -> SkillListResponse:
    store = _skill_store(workspace)
    return SkillListResponse(skills=[_to_response(s) for s in store.list_all()])


@router.post("/skills", response_model=SkillResponse, status_code=201)
async def create_skill(
    request: SkillCreateRequest,
    workspace=Depends(get_workspace),
) -> SkillResponse:
    store = _skill_store(workspace)
    try:
        skill = store.create(
            name=request.name,
            goal_template=request.goal_template,
            description=request.description,
            slash_name=request.slash_name,
            instructions=request.instructions,
            default_plan_mode=request.default_plan_mode,
            constraints=request.constraints,
            success_criteria=request.success_criteria,
            tags=request.tags,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return _to_response(skill)


@router.get("/skills/{skill_id}", response_model=SkillResponse)
async def get_skill(
    skill_id: str,
    workspace=Depends(get_workspace),
) -> SkillResponse:
    skill = _skill_store(workspace).get(skill_id)
    if skill is None:
        raise HTTPException(status_code=404, detail=f"Skill '{skill_id}' not found")
    return _to_response(skill)


@router.patch("/skills/{skill_id}", response_model=SkillResponse)
async def update_skill(
    skill_id: str,
    request: SkillUpdateRequest,
    workspace=Depends(get_workspace),
) -> SkillResponse:
    store = _skill_store(workspace)
    changes = {k: v for k, v in request.model_dump().items() if v is not None}
    try:
        skill = store.update(skill_id, **changes)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return _to_response(skill)


@router.delete("/skills/{skill_id}", response_model=MessageResponse)
async def delete_skill(
    skill_id: str,
    workspace=Depends(get_workspace),
) -> MessageResponse:
    deleted = _skill_store(workspace).delete(skill_id)
    if not deleted:
        raise HTTPException(status_code=404, detail=f"Skill '{skill_id}' not found")
    return MessageResponse(message=f"Skill '{skill_id}' deleted")


# ── Slash commands ───────────────────────────────────────────────────────────


_BUILTIN_COMMANDS: tuple[CommandSpec, ...] = (
    CommandSpec(
        slashName="plan",
        name="Plan mode",
        description="Toggle plan mode for the next message (read-only inspection only).",
        defaultPlanMode=True,
        isBuiltin=True,
    ),
    CommandSpec(
        slashName="clear",
        name="Clear conversation",
        description="Discard the current chat transcript and start fresh.",
        isBuiltin=True,
    ),
    CommandSpec(
        slashName="model",
        name="Change model",
        description="Show or change the active model (e.g. /model claude-sonnet-4-6).",
        isBuiltin=True,
    ),
    CommandSpec(
        slashName="help",
        name="Help",
        description="Show available commands and a short usage reminder.",
        isBuiltin=True,
    ),
)


@router.get("/commands", response_model=CommandListResponse)
async def list_commands(workspace=Depends(get_workspace)) -> CommandListResponse:
    """Return all slash commands available to the chat input.

    Includes the four builtins plus every skill with a non-empty
    ``slash_name``. Each entry carries enough metadata for the client to
    render an autocomplete popover and validate arguments before
    submitting.
    """
    store = _skill_store(workspace)
    commands: list[CommandSpec] = list(_BUILTIN_COMMANDS)
    for skill in store.list_all():
        if not skill.slash_name:
            continue
        commands.append(
            CommandSpec(
                slashName=skill.slash_name,
                name=skill.name,
                description=skill.description,
                parameters=[
                    CommandParameterSpec(name=p, required=True)
                    for p in skill.required_parameters()
                ],
                defaultPlanMode=skill.default_plan_mode,
                isBuiltin=False,
                skillId=skill.id,
            )
        )
    return CommandListResponse(commands=commands)


@router.post("/commands/parse", response_model=CommandParseResponse)
async def parse_command(
    request: CommandParseRequest,
    workspace=Depends(get_workspace),
) -> CommandParseResponse:
    """Parse a raw chat input into a structured ``CommandParseResponse``.

    Mirrors :func:`molexp.plugins.agent_pydanticai.commands.parse`. Errors
    surface as ``kind="error"`` with a UI-ready message — the route never
    raises a 4xx for parser-level issues so the client can render the
    message inline.
    """
    parsed = parse_slash(request.raw, _skill_store(workspace))
    return CommandParseResponse(
        kind=parsed.kind,
        name=parsed.name,
        skillId=parsed.skill_id,
        parameters=dict(parsed.parameters),
        planMode=parsed.plan_mode,
        error=parsed.error,
    )


# Re-export so the router-level / chat code can verify reservation policy
# without re-reaching into the plugin module — used by the UI client too
# via ``GET /agent/commands`` (each entry's ``isBuiltin`` flag).
_ = RESERVED_SLASH_NAMES  # noqa: F841 — kept for IDE-discoverability


# ── Provider config ─────────────────────────────────────────────────────────


def _provider_store(workspace) -> ProviderStore:
    root = getattr(workspace, "root", None)
    if root is None:
        raise HTTPException(status_code=500, detail="Workspace has no root path")
    return ProviderStore(root)


def _to_provider_response(public) -> AgentProviderResponse:
    return AgentProviderResponse(
        provider=public.provider,
        model=public.model,
        baseUrl=public.base_url,
        apiKeyPreview=public.api_key_preview,
        apiKeySet=public.api_key_set,
        instructions=public.instructions,
        supportedProviders=list(SUPPORTED_PROVIDERS),
    )


@router.get("/provider", response_model=AgentProviderResponse)
async def get_provider(workspace=Depends(get_workspace)) -> AgentProviderResponse:
    """Return the workspace's LLM provider config (key redacted)."""
    config = _provider_store(workspace).load()
    return _to_provider_response(to_public(config))


@router.get("/health", response_model=AgentHealthResponse)
async def get_agent_health(workspace=Depends(get_workspace)) -> AgentHealthResponse:
    """Report whether the agent runtime is ready to start a new session.

    UI uses this to render a "configure provider" banner before the user
    even tries to launch — much better UX than letting POST /sessions
    return a structured 400 only after the goal is typed.
    """
    config = _provider_store(workspace).load()
    status = check_credentials(config)
    return AgentHealthResponse(
        ready=status.ready,
        provider=status.provider,
        model=status.model,
        source=status.source,
        reason=status.reason,
        envVar=status.env_var,
    )


@router.put("/provider", response_model=AgentProviderResponse)
async def update_provider(
    request: AgentProviderUpdateRequest,
    workspace=Depends(get_workspace),
) -> AgentProviderResponse:
    """Patch provider/model/api_key/base_url. Empty ``api_key`` clears the key."""
    if request.provider is not None and request.provider not in SUPPORTED_PROVIDERS:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Unsupported provider '{request.provider}'. "
                f"Expected one of: {', '.join(SUPPORTED_PROVIDERS)}"
            ),
        )
    store = _provider_store(workspace)
    updated = store.update(
        provider=request.provider,  # type: ignore[arg-type]
        model=request.model,
        api_key=request.api_key,
        base_url=request.base_url,
        instructions=request.instructions,
    )
    return _to_provider_response(to_public(updated))


def _resolve_probe_config(
    request: AgentProviderUpdateRequest,
    stored: ProviderConfig,
) -> ProviderConfig:
    """Merge unsaved draft fields with the stored config for the probe.

    Any field present in the request overrides the stored value; missing
    fields fall back to what's persisted. This lets the UI test a freshly
    typed key without requiring the user to save first. Switching the
    provider without an explicit model resets to the provider default
    so we don't probe with a mismatched model name (e.g. an Anthropic
    model name against the OpenAI endpoint).
    """
    target_provider = request.provider if request.provider is not None else stored.provider
    if request.model is not None:
        target_model = request.model
    elif request.provider is not None and request.provider != stored.provider:
        target_model = DEFAULT_MODELS[request.provider]  # type: ignore[index]
    else:
        target_model = stored.model
    target_key = request.api_key if request.api_key not in (None, "") else stored.api_key
    target_base = request.base_url if request.base_url is not None else stored.base_url
    return ProviderConfig(
        provider=target_provider,  # type: ignore[arg-type]
        model=target_model,
        api_key=target_key,
        base_url=target_base,
    )


@router.post("/provider/test", response_model=AgentProviderTestResponse)
async def test_provider(
    request: AgentProviderUpdateRequest,
    workspace=Depends(get_workspace),
) -> AgentProviderTestResponse:
    """Send a minimal probe to the configured provider — never persists.

    The request body is treated as an optional override over the stored
    config so the UI can validate user input before saving.
    """
    if request.provider is not None and request.provider not in SUPPORTED_PROVIDERS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported provider '{request.provider}'",
        )
    stored = _provider_store(workspace).load()
    target = _resolve_probe_config(request, stored)
    result = await probe_provider(target)
    return AgentProviderTestResponse(
        ok=result.ok,
        provider=target.provider,
        model=target.model,
        latencyMs=result.latency_ms,
        reply=result.reply,
        error=result.error,
    )
