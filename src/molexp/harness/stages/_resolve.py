"""Upstream-input resolution for Mode-driven pipeline stages.

Stages are self-contained pipeline steps: they read their input(s) by
artifact *kind* from the run's store and write one output. No upstream
artifact ids are threaded through constructors — the pipeline order
guarantees each kind is produced before its consumer runs, so a stage
resolves the latest artifact of the kind it needs at ``run()`` time.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from molexp.harness.errors import StageExecutionError

if TYPE_CHECKING:
    from molexp.harness.core.run_context import HarnessRunContext
    from molexp.harness.gateways.gateway import AgentGateway
    from molexp.harness.schemas import PlanArtifactRef


def require_latest(ctx: HarnessRunContext, kind: str, *, stage: str) -> PlanArtifactRef:
    """Return the most recent artifact of ``kind``, or raise.

    Args:
        ctx: The run context whose ``artifact_store`` is queried.
        kind: The artifact kind the stage consumes (e.g. ``"workflow_ir"``).
        stage: The requesting stage's name, for a clear error message.

    Returns:
        The latest :class:`PlanArtifactRef` of ``kind``.

    Raises:
        StageExecutionError: If no artifact of ``kind`` exists yet — the
            upstream stage that produces it has not run.
    """
    ref = ctx.artifact_store.latest_by_kind(kind)
    if ref is None:
        raise StageExecutionError(
            f"stage {stage!r} requires an upstream {kind!r} artifact, but none exists in the run"
        )
    return ref


def feedback_inputs(ctx: HarnessRunContext, feedback_kind: str) -> list[str]:
    """Return ``[feedback_ref.id]`` if a repair-feedback artifact exists, else ``[]``.

    A generator includes this so that, on a :class:`RepairLoop` retry, it sees the
    previous attempt's validation violations (persisted under ``feedback_kind`` by
    the loop) and can fix them. Absent on the first attempt — the list is empty.
    """
    ref = ctx.artifact_store.latest_by_kind(feedback_kind)
    return [ref.id] if ref is not None else []


def require_agent_gateway(ctx: HarnessRunContext, *, stage: str) -> AgentGateway:
    """Return the run's agent gateway, or raise.

    ``HarnessRunContext.agent_gateway`` is optional because non-LLM stages do
    not need one. A stage that *does* dispatch an agent call must say so and
    fail with its own name rather than dereference ``None``.

    Raises:
        StageExecutionError: If no gateway is wired on ``ctx``.
    """
    if ctx.agent_gateway is None:
        raise StageExecutionError(f"{stage} requires ctx.agent_gateway")
    return ctx.agent_gateway
