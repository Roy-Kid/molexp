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
    from molexp.harness.prompts import prompts_by_agent
    from molexp.harness.prompts.workflow_source import (
        SYSTEM_PROMPT as WORKFLOW_SOURCE_SYSTEM_PROMPT,
    )
    from molexp.harness.schemas import (
        BoundWorkflow,
        ExperimentReport,
        FinalReport,
        TestSource,
        TestSpecBundle,
        WorkflowIR,
        WorkflowSource,
    )
    from molexp.harness.store.file_artifact_store import FileArtifactStore

    store = FileArtifactStore(root=Path(run.run_dir / "artifacts"))
    router = PydanticAIRouter(models=dict.fromkeys(ModelTier, model))
    return RouterBackedAgentGateway(
        router=router,
        artifact_store=store,
        agent_responses={
            "experiment_report_writer": ExperimentReport,
            "workflow_ir_extractor": WorkflowIR,
            "bound_workflow_binder": BoundWorkflow,
            "workflow_source_writer": WorkflowSource,
            "test_spec_writer": TestSpecBundle,
            "test_code_writer": TestSource,
            "final_report_writer": FinalReport,
        },
        output_kind_by_agent={
            "experiment_report_writer": "experiment_report",
            "workflow_ir_extractor": "workflow_ir",
            "bound_workflow_binder": "bound_workflow",
            "workflow_source_writer": "workflow_source",
            "test_spec_writer": "test_spec",
            "test_code_writer": "test_source",
            "final_report_writer": "final_report",
        },
        system_prompt_by_agent={
            **prompts_by_agent(),
            "workflow_source_writer": WORKFLOW_SOURCE_SYSTEM_PROMPT,
        },
        model=model,
    )
