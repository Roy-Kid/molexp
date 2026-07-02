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


def preflight_plan_router(*, model: str) -> Router:
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
        from molexp.agent.router import ModelTier
    except ModuleNotFoundError as exc:
        raise PlanPreflightError(
            f"PlanMode needs the LLM agent stack, but {exc.name!r} is not installed — "
            'install it with: pip install "molexp[agent]"'
        ) from exc
    try:
        router = PydanticAIRouter(models=dict.fromkeys(ModelTier, model))
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


def build_plan_gateway(*, model: str, run: Run, router: Router | None = None) -> AgentGateway:
    """Build the production ``RouterBackedAgentGateway`` (or the test stub).

    The gateway's artifact store shares the run's ``artifacts`` directory with
    the Mode-built context, so stage outputs land in one place. Pass the
    ``router`` returned by :func:`preflight_plan_router` to reuse the already
    validated (and agent-primed) instance; without one, the router is built —
    and preflighted — here.
    """
    if _gateway_factory is not None:
        return _gateway_factory(run, model)

    from molexp.harness import RouterBackedAgentGateway
    from molexp.harness.gateways import (
        plan_agent_responses,
        plan_output_kinds,
        plan_system_prompts,
    )
    from molexp.harness.store.file_artifact_store import FileArtifactStore

    if router is None:
        router = preflight_plan_router(model=model)
    store = FileArtifactStore(root=Path(run.run_dir / "artifacts"))
    return RouterBackedAgentGateway(
        router=router,
        artifact_store=store,
        agent_responses=plan_agent_responses(),
        output_kind_by_agent=plan_output_kinds(),
        system_prompt_by_agent=plan_system_prompts(),
        model=model,
    )
