"""``drive_plan_mode`` — the ONE way CLI and server run the plan bundle.

Wraps ``run_plan`` in the run's own lifecycle (``run.start()``), so a
plan Run's workspace status is honest: ``running`` while the pipeline
executes, ``succeeded`` when it completes, ``failed`` when it raises.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from molexp.harness import CapabilityRegistry, ModeResult
    from molexp.harness.gateways.gateway import AgentGateway
    from molexp.workspace.run import Run

__all__ = ["drive_plan_mode"]

PlanRunner = Callable[..., Awaitable[Any]]


async def drive_plan_mode(
    *,
    run: Run,
    user_input: str,
    gateway: AgentGateway,
    capability_registry: CapabilityRegistry | None = None,
    plan: PlanRunner | None = None,
    **plan_kwargs: Any,  # noqa: ANN401 — forwarded to run_plan
) -> ModeResult:
    """Run the plan bundle against *run* inside the run lifecycle.

    Tests inject a fake via ``plan=``. Production uses
    :func:`molexp.harness.run_plan`.
    """
    from molexp.harness.modes.plan import run_plan

    runner = plan if plan is not None else run_plan
    with run.start() as run_ctx:
        result = await runner(
            run=run,
            user_input=user_input,
            gateway=gateway,
            capability_registry=capability_registry,
            **plan_kwargs,
        )
        run_ctx.mark_succeeded()
    return result
