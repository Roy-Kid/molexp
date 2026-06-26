"""Production gateway builder for server-run PlanMode pipelines.

Mirrors ``cli/plan_cmd.py``'s ``PlanRuntime.build_gateway`` for the server entry
point: the two application shells each build their own production
``AgentGateway`` (as ``routes/agent.py`` builds its own ``AgentRunner``), so
neither imports the other. A module-level factory seam lets tests inject a
``StubAgentGateway`` instead of constructing a real ``PydanticAIRouter``.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable

    from molexp.harness.gateways.gateway import AgentGateway
    from molexp.workspace.run import Run

    PlanGatewayFactory = Callable[[Run, str], AgentGateway]

__all__ = ["build_plan_gateway", "reset_plan_gateway_factory", "set_plan_gateway_factory"]

# Test seam (mirrors routes/agent.py's _runner_factory): a factory(run, model).
_gateway_factory: PlanGatewayFactory | None = None


def set_plan_gateway_factory(factory: PlanGatewayFactory) -> None:
    """Install a test gateway factory called as ``factory(run, model)``."""
    global _gateway_factory
    _gateway_factory = factory


def reset_plan_gateway_factory() -> None:
    """Drop any installed test gateway factory."""
    global _gateway_factory
    _gateway_factory = None


def build_plan_gateway(*, model: str, run: Run) -> AgentGateway:
    """Build the production ``RouterBackedAgentGateway`` (or the test stub).

    The gateway's artifact store shares the run's ``artifacts`` directory with
    the Mode-built context, so stage outputs land in one place.
    """
    if _gateway_factory is not None:
        return _gateway_factory(run, model)

    from molexp.agent import PydanticAIRouter
    from molexp.agent.router import ModelTier
    from molexp.harness import RouterBackedAgentGateway
    from molexp.harness.gateways import (
        plan_agent_responses,
        plan_output_kinds,
        plan_system_prompts,
    )
    from molexp.harness.store.file_artifact_store import FileArtifactStore

    store = FileArtifactStore(root=Path(run.run_dir / "artifacts"))
    router = PydanticAIRouter(models=dict.fromkeys(ModelTier, model))
    return RouterBackedAgentGateway(
        router=router,
        artifact_store=store,
        agent_responses=plan_agent_responses(),
        output_kind_by_agent=plan_output_kinds(),
        system_prompt_by_agent=plan_system_prompts(),
        model=model,
    )
