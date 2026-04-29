"""PydanticAIRuntime: concrete AgentRuntime implementation.

Wraps pydantic-ai Agent with:
- MolexpToolCatalog (built-in + user tools)
- ApprovalPolicy via pydantic-ai's approval_required()
- Session persistence in workspace sessions/ directory
- Message history for cross-process resumption
"""

from __future__ import annotations

import json
import os
import uuid
from pathlib import Path
from typing import Any

from mollog import get_logger
from pydantic_ai import Agent
from pydantic_ai.models import Model

from ..policy import ApprovalPolicy
from ..provider import DEEPSEEK_DEFAULT_BASE_URL, ProviderConfig, ProviderStore
from ..runtime import AgentRuntime
from ..sessions_store import write_session_metadata
from ..tools import Tool
from ..types import (
    AgentSession,
    Goal,
    ToolContext,  # noqa: F401 - re-exported for tool wrappers
)
from .catalog import MolexpToolCatalog
from .deps import MolexpDeps
from .session import PydanticAISession
from .system_prompt import BASE_SYSTEM_PROMPT, compose_system_prompt

logger = get_logger(__name__)

# Re-exported for backward compatibility — the canonical location is
# :mod:`._pydantic_ai.system_prompt`. Older imports of ``_SYSTEM_PROMPT``
# continue to work without change.
_SYSTEM_PROMPT = BASE_SYSTEM_PROMPT


class PydanticAIRuntime(AgentRuntime):
    """AgentRuntime backed by pydantic-ai.

    Creates a pydantic-ai Agent configured with the MolexpToolCatalog
    and manages the session lifecycle including persistence.

    Args:
        model: pydantic-ai model name or instance (default: claude-sonnet-4-6)
        sessions_dir_name: Workspace subdirectory for session persistence
    """

    def __init__(
        self,
        model: str = "anthropic:claude-sonnet-4-6",
        sessions_dir_name: str = "sessions",
    ) -> None:
        self._model = model
        self._sessions_dir_name = sessions_dir_name
        self._active_sessions: dict[str, PydanticAISession] = {}

    def _get_sessions_dir(self, workspace: Any) -> Path:
        sessions_dir = Path(workspace.root) / self._sessions_dir_name
        sessions_dir.mkdir(parents=True, exist_ok=True)
        return sessions_dir

    def _resolve_model(self, workspace: Any | None) -> str | Model:
        """Pick the live model: workspace provider config wins over the runtime default.

        Falls back to the runtime ``self._model`` string when no workspace is
        attached or no key has been configured — pydantic-ai will then look
        for credentials in the process environment, preserving prior behavior.
        """
        if workspace is None:
            return self._model
        root = getattr(workspace, "root", None)
        if root is None:
            return self._model
        config = ProviderStore(root).load()
        if not config.api_key:
            return self._model
        try:
            return _build_model_from_config(config)
        except Exception:
            logger.exception("Failed to build model from workspace provider config")
            return self._model

    def _build_agent(
        self,
        extra_tools: list[Tool],
        approval_policy: ApprovalPolicy,
        workspace: Any | None = None,
        goal: Goal | None = None,
    ) -> Agent[MolexpDeps, str]:
        plan_mode = bool(goal and goal.plan_mode)
        catalog = MolexpToolCatalog(
            extra_tools=extra_tools,
            approval_policy=approval_policy,
            read_only=plan_mode,
        )
        toolset = catalog.build()

        toolsets: list[Any] = [toolset]
        if workspace is not None:
            toolsets.extend(_load_mcp_toolsets(workspace))

        system_prompt = self._compose_system_prompt(workspace, goal)

        return Agent(
            model=self._resolve_model(workspace),
            system_prompt=system_prompt,
            deps_type=MolexpDeps,
            toolsets=toolsets,
        )

    def _compose_system_prompt(self, workspace: Any | None, goal: Goal | None) -> str:
        """Layer the workspace, skill, and session prompts into a single string.

        Reads the workspace-default ``instructions`` from
        :class:`ProviderStore` (returns ``""`` for unconfigured workspaces),
        then layers on the goal-supplied skill addendum and override.
        """
        workspace_instructions = ""
        root = getattr(workspace, "root", None) if workspace is not None else None
        if root is not None:
            try:
                workspace_instructions = ProviderStore(root).load().instructions
            except Exception:
                logger.exception("Failed to load workspace instructions")
                workspace_instructions = ""
        skill_instructions = goal.skill_instructions if goal else ""
        session_override = goal.instructions_override if goal else None
        plan_mode = bool(goal and goal.plan_mode)
        return compose_system_prompt(
            base=BASE_SYSTEM_PROMPT,
            workspace_instructions=workspace_instructions,
            skill_instructions=skill_instructions,
            session_override=session_override,
            plan_mode=plan_mode,
        )

    def _goal_to_prompt(self, goal: Goal) -> str:
        lines = [f"Goal: {goal.description}"]
        if goal.constraints:
            lines.append(f"Constraints: {goal.constraints}")
        if goal.success_criteria:
            lines.append("Success criteria:")
            for criterion in goal.success_criteria:
                lines.append(f"  - {criterion}")
        return "\n".join(lines)

    async def start_session(
        self,
        goal: Goal,
        workspace: Any,
        extra_tools: list[Tool],
        approval_policy: ApprovalPolicy,
    ) -> AgentSession:
        session_id = f"sess-{uuid.uuid4().hex[:12]}"

        session = PydanticAISession(
            session_id=session_id,
            goal=goal,
            workspace=workspace,
        )

        deps = MolexpDeps(
            workspace=workspace,
            session_id=session_id,
            session=session,
        )

        agent = self._build_agent(
            extra_tools, approval_policy, workspace=workspace, goal=goal
        )
        session.set_system_prompt(self._compose_system_prompt(workspace, goal))
        prompt = self._goal_to_prompt(goal)

        # Register before launching so persistence can find it
        self._active_sessions[session_id] = session

        # Persist session metadata at start, and again on terminal transition
        # so disk listings reflect the final status (completed / failed).
        self._save_session_metadata(session, workspace)
        session._on_terminal = lambda s, ws=workspace: self._save_session_metadata(s, ws)

        # Launch agent run as background task
        session._launch(agent=agent, prompt=prompt, deps=deps)

        logger.info(f"Started agent session {session_id}")
        return session

    async def resume_session(
        self,
        session_id: str,
        workspace: Any,
    ) -> AgentSession:
        sessions_dir = self._get_sessions_dir(workspace)
        session_dir = sessions_dir / session_id

        if not session_dir.exists():
            raise ValueError(f"Session '{session_id}' not found in workspace")

        # Load metadata
        meta_path = session_dir / "metadata.json"
        with meta_path.open() as f:
            meta = json.load(f)

        goal_meta = meta.get("goal", {})
        goal = Goal(
            description=goal_meta.get("description", ""),
            constraints=goal_meta.get("constraints", {}),
            success_criteria=goal_meta.get("success_criteria", []),
            plan_mode=bool(goal_meta.get("plan_mode", False)),
            instructions_override=goal_meta.get("instructions_override"),
            skill_id=goal_meta.get("skill_id"),
            skill_instructions=goal_meta.get("skill_instructions", ""),
        )

        session = PydanticAISession(
            session_id=session_id,
            goal=goal,
            workspace=workspace,
        )

        # Restore message history if available
        history_path = session_dir / "history.json"
        if history_path.exists():
            from pydantic_ai.messages import ModelMessagesTypeAdapter

            with history_path.open("rb") as f:
                history = ModelMessagesTypeAdapter.validate_json(f.read())
            session.restore_message_history(history)

        # Re-launch with empty extra_tools and default approval policy
        # (user can pass updated tools via AgentService.resume)
        deps = MolexpDeps(
            workspace=workspace,
            session_id=session_id,
            session=session,
        )
        agent = self._build_agent([], ApprovalPolicy(), workspace=workspace, goal=goal)
        session.set_system_prompt(self._compose_system_prompt(workspace, goal))
        prompt = "Resume from where we left off and continue towards the original goal."

        self._active_sessions[session_id] = session
        session._launch(agent=agent, prompt=prompt, deps=deps)

        logger.info(f"Resumed agent session {session_id}")
        return session

    async def get_session_history(self, session_id: str) -> Any:
        session = self._active_sessions.get(session_id)
        if session is not None:
            return {"session_id": session_id, "messages": session.get_message_history()}
        return {"session_id": session_id, "messages": []}

    def _save_session_metadata(self, session: PydanticAISession, workspace: Any) -> None:
        """Persist current session metadata atomically (delegates to sessions_store).

        Called once at start (status = ``running``) and again at termination
        with the final status so historical listings reflect the real outcome.
        """
        root = getattr(workspace, "root", None)
        if root is None:
            return
        completed_at = (
            session.stats.completed_at.isoformat() if session.stats.completed_at else None
        )
        created_at = (
            session.stats.started_at.isoformat() if session.stats.started_at else None
        )
        write_session_metadata(
            root,
            session.session_id,
            status=session.status,
            goal_description=session.goal.description,
            constraints=session.goal.constraints,
            success_criteria=list(session.goal.success_criteria),
            created_at=created_at,
            completed_at=completed_at,
            plan_mode=session.goal.plan_mode,
            instructions_override=session.goal.instructions_override,
            skill_id=session.goal.skill_id,
            skill_instructions=session.goal.skill_instructions,
        )

    def save_session_history(self, session: PydanticAISession, workspace: Any) -> None:
        """Persist message history for resumption (call after session completes)."""
        try:
            from pydantic_ai.messages import ModelMessagesTypeAdapter

            history = session.get_message_history()
            if not history:
                return
            sessions_dir = self._get_sessions_dir(workspace)
            session_dir = sessions_dir / session.session_id
            session_dir.mkdir(parents=True, exist_ok=True)
            history_path = session_dir / "history.json"
            tmp_path = history_path.with_suffix(".tmp")
            tmp_path.write_bytes(ModelMessagesTypeAdapter.dump_json(history, indent=2))
            tmp_path.rename(history_path)
        except Exception:
            logger.exception(f"Failed to save session history for {session.session_id}")


# ── MCP toolset loading ─────────────────────────────────────────────────────


def _expand_args(args: list[str], workspace_root: Path) -> list[str]:
    """Expand ``${workspaceRoot}`` and environment variables in MCP args."""
    expanded: list[str] = []
    for arg in args:
        substituted = arg.replace("${workspaceRoot}", str(workspace_root))
        substituted = os.path.expandvars(substituted)
        expanded.append(substituted)
    return expanded


def _load_mcp_toolsets(workspace: Any) -> list[Any]:
    """Build pydantic-ai toolsets from the multi-scope MCP store.

    Reads merged User+Workspace entries via :class:`McpStore`, dispatches
    each one to the matching pydantic-ai class (Stdio / SSE / streamable
    HTTP). Entries with unresolved ``${SECRET:K}`` references are skipped
    with a clear warning — there is no env-var fallback. Skipping bad
    entries is per-server: a single broken spec cannot brick the agent.
    """
    root = getattr(workspace, "root", None)
    if root is None:
        return []

    try:
        from pydantic_ai.mcp import (
            MCPServerSSE,
            MCPServerStdio,
            MCPServerStreamableHTTP,
        )
    except ImportError:
        logger.warning(
            "pydantic-ai MCP support unavailable; "
            "skipping MCP servers. Install: pip install 'pydantic-ai[mcp]'."
        )
        return []

    from ..mcp_store import McpStore, UnresolvedSecretError

    store = McpStore(root)
    workspace_root = Path(root).resolve()
    toolsets: list[Any] = []

    for entry in store.list():
        if entry.shadowed:
            # Workspace beats User; the workspace twin will be processed
            # separately. Don't double-load the same logical name.
            continue
        if not entry.valid:
            logger.warning(
                f"Skipping MCP server '{entry.name}' ({entry.scope.value}): "
                f"{entry.invalid_reason}"
            )
            continue
        if entry.unresolved_secrets:
            logger.warning(
                f"Skipping MCP server '{entry.name}' ({entry.scope.value}): "
                f"missing secrets {list(entry.unresolved_secrets)}. "
                f"Set them via the agent settings UI."
            )
            continue

        try:
            resolved = store.resolve(entry)
        except UnresolvedSecretError as exc:
            logger.warning(
                f"Skipping MCP server '{entry.name}': {exc}"
            )
            continue

        try:
            if resolved.transport == "stdio":
                toolsets.append(
                    MCPServerStdio(
                        command=resolved.command or "",
                        args=_expand_args(list(resolved.args), workspace_root),
                        env=resolved.env or None,
                        tool_prefix=entry.name,
                    )
                )
            elif resolved.transport in ("http", "sse"):
                http_client = _maybe_oauth_http_client_runtime(
                    resolved, entry, store
                )
                kwargs: dict[str, Any] = (
                    {"http_client": http_client}
                    if http_client is not None
                    else {"headers": resolved.headers or None}
                )
                # ``http`` = streamable HTTP (Claude Code convention);
                # ``sse`` = legacy long-poll transport.
                cls = (
                    MCPServerSSE
                    if resolved.transport == "sse"
                    else MCPServerStreamableHTTP
                )
                toolsets.append(
                    cls(url=resolved.url or "", tool_prefix=entry.name, **kwargs)
                )
            else:
                logger.warning(
                    f"Skipping MCP server '{entry.name}': "
                    f"unknown transport {resolved.transport!r}"
                )
        except Exception:
            logger.exception(f"Failed to build MCP toolset '{entry.name}'")
    return toolsets


def _maybe_oauth_http_client_runtime(resolved: Any, entry: Any, store: Any) -> Any:
    """Return an OAuth-equipped httpx client when the entry uses OAuth.

    Skips servers whose token file does not exist (proxy for "user never
    clicked Connect"). The check is synchronous because toolset loading
    happens in a sync code path that may already be inside an event loop;
    running ``asyncio.run`` would deadlock. File-existence misses corrupt
    tokens on disk, but those degrade to a 401 on the next MCP request —
    surfaced via the same "reconnect via UI" message.
    """
    if resolved.auth is None or resolved.auth.type != "oauth2":
        return None
    import httpx

    from ..mcp_oauth import build_oauth_provider, default_redirect_uri, storage_for

    storage = storage_for(store, entry.scope, entry.name)
    if not storage.path.exists():
        logger.warning(
            "Skipping MCP server '%s': OAuth not connected. "
            "Click Connect in the agent settings.",
            entry.name,
        )
        return None
    provider = build_oauth_provider(
        server_url=resolved.url or "",
        redirect_uri=default_redirect_uri(),
        scopes=list(resolved.auth.scopes),
        storage=storage,
    )
    return httpx.AsyncClient(auth=provider)


# ── Provider-config-driven model construction ──────────────────────────────


def _build_model_from_config(config: ProviderConfig) -> Model:
    """Build a pydantic-ai :class:`Model` from a workspace :class:`ProviderConfig`.

    Each branch instantiates the provider-specific Provider with the user's
    API key (and optional ``base_url``) so we never mutate the process env.
    Unknown ``provider`` values raise — callers should swallow and fall back.
    """
    if config.provider == "anthropic":
        from pydantic_ai.models.anthropic import AnthropicModel
        from pydantic_ai.providers.anthropic import AnthropicProvider

        provider = AnthropicProvider(
            api_key=config.api_key,
            base_url=config.base_url or None,
        )
        return AnthropicModel(config.model, provider=provider)

    if config.provider in ("openai", "openai-compatible"):
        from pydantic_ai.models.openai import OpenAIChatModel
        from pydantic_ai.providers.openai import OpenAIProvider

        provider = OpenAIProvider(
            api_key=config.api_key,
            base_url=config.base_url or None,
        )
        return OpenAIChatModel(config.model, provider=provider)

    if config.provider == "deepseek":
        # DeepSeek speaks the OpenAI chat-completions wire format; we just
        # default the base_url so users only have to paste the API key.
        # An explicit ``base_url`` (e.g. a regional mirror) overrides.
        from pydantic_ai.models.openai import OpenAIChatModel
        from pydantic_ai.providers.openai import OpenAIProvider

        provider = OpenAIProvider(
            api_key=config.api_key,
            base_url=config.base_url or DEEPSEEK_DEFAULT_BASE_URL,
        )
        return OpenAIChatModel(config.model, provider=provider)

    if config.provider == "google":
        from pydantic_ai.models.google import GoogleModel
        from pydantic_ai.providers.google import GoogleProvider

        provider = GoogleProvider(
            api_key=config.api_key,
            base_url=config.base_url or None,
        )
        return GoogleModel(config.model, provider=provider)

    raise ValueError(f"Unsupported provider '{config.provider}'")
