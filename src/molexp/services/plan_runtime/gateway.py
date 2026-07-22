"""Production gateway builder + agent-stack preflight for PlanMode pipelines.

The ONE gateway construction path both entry points share: ``molexp plan``
(:mod:`molexp.cli.plan_cmd`) and the server's ``plan-tasks`` background task.
A module-level factory seam lets tests inject a ``StubAgentGateway`` instead
of constructing a real ``PydanticAIRouter``.

Zero-residue preflight (``molexp plan`` P0): :func:`preflight_plan_router`
imports the agent stack, constructs the router, and forces credential /
provider resolution — all **without touching the network or the disk** — so
the CLI can fail fast *before* it materializes anything into the workspace.
A missing ``molexp[agent]`` extra, an unknown model id, and a missing API
key all surface here as a short :class:`PlanPreflightError`; missing-key
failures additionally carry one ``molexp:`` guidance line naming the key
path that actually works (see :func:`_credential_guidance`).
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import TYPE_CHECKING

from mollog import get_logger

if TYPE_CHECKING:
    from collections.abc import Callable

    from molexp.agent.router import Router
    from molexp.harness.gateways.gateway import AgentGateway
    from molexp.workspace.run import Run

    PlanGatewayFactory = Callable[[Run, str], AgentGateway]

__all__ = [
    "PlanPreflightError",
    "build_plan_gateway",
    "preflight_plan_router",
    "reset_plan_gateway_factory",
    "set_plan_gateway_factory",
]

logger = get_logger(__name__)

# Test seam (mirrors routes/agent.py's _runner_factory): a factory(run, model).
_gateway_factory: PlanGatewayFactory | None = None

#: Known base model → HEAVY counterpart used only when ``agent.models`` is
#: absent. Stages still must declare tiers explicitly; this is operator-side
#: model-id wiring, not a gateway tier fallback.
_HEAVY_COUNTERPART: dict[str, str] = {
    "deepseek:deepseek-v4-flash": "deepseek:deepseek-v4-pro",
}


def _resolve_tier_models(*, model: str, models: dict[str, str] | None) -> dict:
    """Resolve cheap/default/heavy model ids for the plan router.

    Preference: explicit ``models=`` arg → bridged ``agent.models`` → disk
    ``agent.models`` → derive from ``agent.model`` (flash→pro when known).
    Stage→tier mapping is separate and never falls back (see plan_agent_tiers).
    """
    from molexp.agent.router import ModelTier

    if models is not None:
        missing = [t.value for t in ModelTier if t.value not in models or not models[t.value]]
        if missing:
            raise PlanPreflightError(
                "plan models map is incomplete — configure agent.models for "
                f"every tier; missing: {missing}"
            )
        return {tier: models[tier.value] for tier in ModelTier}

    import molexp
    from molexp.services.operator_config import (
        AGENT_MODELS_KEY,
        configured_agent_models,
        load_operator_config,
    )

    bridged = molexp.config.get(AGENT_MODELS_KEY)
    if (
        bridged is not None
        and callable(getattr(bridged, "get", None))
        and all(
            isinstance(bridged.get(t), str) and bridged.get(t)
            for t in ("cheap", "default", "heavy")
        )
    ):
        return {tier: str(bridged.get(tier.value)) for tier in ModelTier}

    from_disk = configured_agent_models(load_operator_config())
    if from_disk is not None and all(from_disk.get(t.value) for t in ModelTier):
        return {tier: from_disk[tier.value] for tier in ModelTier}

    # Derive from single agent.model. Known bases (flash) upgrade HEAVY to
    # pro; others use the same id for all three slots until agent.models is
    # set. Stage→tier assignment remains fully explicit (plan_agent_tiers).
    heavy = _HEAVY_COUNTERPART.get(model, model)
    return {
        ModelTier.CHEAP: model,
        ModelTier.DEFAULT: model,
        ModelTier.HEAVY: heavy,
    }


class PlanPreflightError(RuntimeError):
    """The agent stack cannot run this plan — a short, human-readable reason.

    Raised by :func:`preflight_plan_router` *before* any workspace write, so
    callers can print ``str(exc)`` verbatim and exit non-zero with nothing
    left on disk. A missing-credential failure appends one extra ``molexp:``
    guidance line naming the key path that actually works (see
    :func:`_credential_guidance`).
    """


def set_plan_gateway_factory(factory: PlanGatewayFactory) -> None:
    """Install a test gateway factory called as ``factory(run, model)``."""
    global _gateway_factory
    _gateway_factory = factory


def reset_plan_gateway_factory() -> None:
    """Drop any installed test gateway factory."""
    global _gateway_factory
    _gateway_factory = None


def preflight_plan_router(*, model: str, models: dict[str, str] | None = None) -> Router:
    """Import, construct, and credential-prime the production router.

    No disk writes and no network traffic — safe to call before the Run is
    materialized. Three failure classes translate into a one-line
    :class:`PlanPreflightError`:

    - the ``molexp[agent]`` extra is not installed (``ModuleNotFoundError``);
    - the model id is unknown / malformed (pydantic-ai rejects it);
    - the provider credential is missing (pydantic-ai's provider raises at
      construction, e.g. ``ANTHROPIC_API_KEY``; DeepSeek keys come from
      ``molexp.config``);
    - the installed pydantic-ai major rejects an ``Agent(...)`` constructor
      kwarg (version incompatibility) — the router's ``preflight()`` primes
      the text *and* structured agent paths so this fails here, not
      mid-pipeline.

    Bridges the operator config first, so keys persisted via
    ``molexp config set agent.<provider>_api_key`` reach ``molexp.config``
    before the router is constructed (in-code registrations keep precedence).
    """
    from molexp.services.operator_config import bridge_operator_config

    bridge_operator_config()
    try:
        from molexp.agent import PydanticAIRouter
    except ModuleNotFoundError as exc:
        raise PlanPreflightError(
            f"PlanMode needs the LLM agent stack, but {exc.name!r} is not installed — "
            'install it with: pip install "molexp[agent]"'
        ) from exc
    try:
        tier_models = _resolve_tier_models(model=model, models=models)
        router = PydanticAIRouter(models=tier_models)
        _prime_credentials(router)
    except Exception as exc:
        message = f"model {model!r} failed its preflight check: {exc}"
        guidance = _credential_guidance(model, exc)
        if guidance is not None:
            message = f"{message}\n{guidance}"
        raise PlanPreflightError(message) from exc
    return router


#: Provider env-var spelling inside upstream credential errors
#: (``ANTHROPIC_API_KEY``, ``OPENAI_API_KEY``, …).
_API_KEY_ENV_RE = re.compile(r"[A-Z][A-Z0-9]*(?:_[A-Z0-9]+)*_API_KEY")


def _credential_guidance(model: str, exc: Exception) -> str | None:
    """molexp's own guidance line for a credential-shaped preflight failure.

    Appended below the upstream (pydantic-ai) reason. It names ONLY key paths
    that actually work today — verified against the loaders:

    - ``deepseek:*`` models read their key from the in-code
      ``molexp.config["deepseek_api_key"]`` — registered in Python, or
      persisted once via ``molexp config set agent.deepseek_api_key sk-…``
      (bridged into ``molexp.config`` at preflight; never read from the
      environment).
    - Every other provider is resolved by pydantic-ai from the process
      environment (``ANTHROPIC_API_KEY``, …) — for those providers the
      environment is currently the only key path.

    Returns ``None`` for non-credential failures (unknown model id, …) so
    irrelevant hints never appear.
    """
    text = str(exc)
    lowered = text.lower()
    if "api_key" not in lowered and "api key" not in lowered:
        return None
    if model.partition(":")[0] == "deepseek":
        return (
            "molexp: register the DeepSeek key once with "
            "`molexp config set agent.deepseek_api_key sk-…`, or set "
            'molexp.config["deepseek_api_key"] = "sk-…" in Python (in-code wins; '
            "molexp never reads DEEPSEEK_API_KEY from the environment)."
        )
    match = _API_KEY_ENV_RE.search(text)
    env_var = match.group(0) if match else "the provider's *_API_KEY variable"
    return (
        f"molexp: set {env_var} in the environment of the process running "
        "`molexp plan` — for this provider the environment is currently the only "
        "key path (`molexp config` stores the model id `agent.model` only, never "
        "API keys)."
    )


def _prime_credentials(router: object) -> None:
    """Force provider/credential resolution now instead of at the first LLM call.

    pydantic-ai validates credentials *and* constructor kwargs when the
    underlying ``Agent`` is constructed; ``PydanticAIRouter`` defers that to
    first use — far too late for the zero-residue preflight. Its public
    :meth:`~molexp.agent._pydanticai.router.PydanticAIRouter.preflight` hook
    constructs one text and one structured agent per tier (no network I/O;
    the primed agents stay cached for real use), so a missing API key AND a
    pydantic-ai version incompatibility (an ``Agent(...)`` kwarg the
    installed major no longer accepts) both surface here rather than
    mid-pipeline at the first structured call.
    """
    primer = getattr(router, "preflight", None)
    if callable(primer):
        primer()
    else:  # pragma: no cover — only a router refactor can take this branch
        logger.warning(
            "router exposes no preflight() hook; credential preflight skipped "
            "(a missing API key will surface at the first LLM call instead)"
        )


def _resolve_agent_mcp_tools(
    servers_by_agent: dict[str, tuple[str, ...]],
    workspace_root: str | Path,
) -> dict[str, tuple[object, ...]]:
    """Resolve per-agent MCP server *names* into concrete ``McpToolSpec``s.

    Reads the same MCP config (`~/.molexp/mcp.json` + workspace `.mcp.json`)
    the capability prefetch uses. Only stdio servers are attachable as codegen
    tools. A missing server is omitted from the return map; the caller
    (:func:`build_plan_gateway`) fail-closes for agents that *require* molmcp.
    Never raises — resolution errors become absent tools.
    """
    from molexp.agent.mcp import McpScope, McpStore
    from molexp.agent.router import McpToolSpec

    resolved: dict[str, tuple[object, ...]] = {}
    store: McpStore | None = None
    cache: dict[str, McpToolSpec | None] = {}

    def spec_for(name: str) -> McpToolSpec | None:
        nonlocal store
        if name in cache:
            return cache[name]
        try:
            if store is None:
                store = McpStore(workspace_root)
            entry = store.get(McpScope.WORKSPACE, name) or store.get(McpScope.USER, name)
            if entry is None:
                cache[name] = None
                return None
            spec = store.resolve(entry)
            if spec.transport != "stdio" or not spec.command:
                cache[name] = None
                return None
            tool = McpToolSpec(
                name=name,
                command=spec.command,
                args=tuple(spec.args),
                env=tuple(spec.env.items()),
            )
        except Exception as exc:  # never let MCP config break gateway build
            logger.warning(f"MCP tool {name!r} not attached to codegen agents: {exc!r}")
            cache[name] = None
            return None
        cache[name] = tool
        return tool

    for agent_name, names in servers_by_agent.items():
        tools = tuple(t for t in (spec_for(n) for n in names) if t is not None)
        if tools:
            resolved[agent_name] = tools
    return resolved


def build_plan_gateway(
    *,
    model: str,
    run: Run,
    router: Router | None = None,
    models: dict[str, str] | None = None,
    workspace_root: str | Path | None = None,
    task_id: str | None = None,
    draft: str | None = None,
    turn_id: str | None = None,
) -> AgentGateway:
    """Build the production ``RouterBackedAgentGateway`` (or the test stub).

    The gateway's artifact store shares the run's ``artifacts`` directory with
    the Mode-built context, so stage outputs land in one place. Pass the
    ``router`` returned by :func:`preflight_plan_router` to reuse the already
    validated (and agent-primed) instance; without one, the router is built —
    and preflighted — here.

    When ``workspace_root`` + ``task_id`` are set, each LLM call is projected
    into that agent-task session as a cacheable ``llm_call`` event (full bodies
    under ``llm_cache/``, pruned by age/size). ``draft`` seeds a single
    ``loop_started`` so the Agents chat groups mid-plan calls under one turn.
    """
    if _gateway_factory is not None:
        return _gateway_factory(run, model)

    from molexp.harness import RouterBackedAgentGateway
    from molexp.harness.gateways import (
        plan_agent_mcp_servers,
        plan_agent_responses,
        plan_agent_tiers,
        plan_output_kinds,
        plan_system_prompts,
    )
    from molexp.harness.store.file_artifact_store import FileArtifactStore

    if router is None:
        router = preflight_plan_router(model=model, models=models)
    store = FileArtifactStore(root=Path(run.run_dir / "artifacts"))
    required_mcp = plan_agent_mcp_servers()
    mcp_tools_by_agent = _resolve_agent_mcp_tools(required_mcp, workspace_root or run.run_dir)
    # Fail-closed: codegen agents that declare molmcp must actually get tools.
    # Silent catalog-only fallback invents APIs; raise before the plan starts.
    missing = [
        agent
        for agent, names in required_mcp.items()
        if names and not mcp_tools_by_agent.get(agent)
    ]
    if missing:
        raise PlanPreflightError(
            "Plan codegen requires the molmcp MCP server, but it is not configured "
            f"for agent(s): {sorted(missing)}. Install/configure molmcp "
            "(e.g. `molexp` MCP settings or `~/.molexp/mcp.json`) so writers can "
            "look up real molcrafts APIs — do not proceed with invented symbols."
        )

    on_llm_call = None
    if workspace_root is not None and task_id:
        from molexp.services.agent_task_llm_cache import (
            ensure_plan_session_started,
            make_llm_call_observer,
            prune_agent_task_llm_cache,
        )

        # Retention before a new plan starts — drop stale cache from prior runs.
        try:
            prune_agent_task_llm_cache(workspace_root)
        except Exception as exc:  # never block gateway construction
            logger.warning(f"llm cache prune at plan start failed: {exc!r}")
        if draft is not None:
            ensure_plan_session_started(workspace_root, task_id, draft=draft, turn_id=turn_id)
        on_llm_call = make_llm_call_observer(workspace_root, task_id, turn_id=turn_id)

    responses = plan_agent_responses()
    tiers = plan_agent_tiers()
    missing_tiers = set(responses) - set(tiers)
    if missing_tiers:
        raise PlanPreflightError(
            "plan_agent_tiers() must list every plan agent (no default-tier "
            f"fallback); missing: {sorted(missing_tiers)!r}"
        )
    return RouterBackedAgentGateway(
        router=router,
        artifact_store=store,
        agent_responses=responses,
        output_kind_by_agent=plan_output_kinds(),
        system_prompt_by_agent=plan_system_prompts(),
        tier_by_agent=tiers,
        mcp_tools_by_agent=mcp_tools_by_agent,
        model=model,
        on_llm_call=on_llm_call,
    )
