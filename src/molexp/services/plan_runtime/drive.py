"""``drive_plan_mode`` — CLI and server share this Run-lifecycle wrapper.

The plan *bundle* is :class:`molexp.harness.Plan` (harness). This module only
starts the workspace Run so both faces see the same status machine.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    from molexp.harness import CapabilityRegistry, ModeResult
    from molexp.harness.gateways.gateway import AgentGateway
    from molexp.workspace.run import Run

__all__ = ["drive_plan_mode"]


class _PlanLike(Protocol):
    async def run(
        self,
        *,
        run: Any,  # noqa: ANN401
        user_input: str,
        gateway: Any,  # noqa: ANN401
        capability_registry: Any = None,  # noqa: ANN401
    ) -> Any: ...  # noqa: ANN401


async def drive_plan_mode(
    plan: _PlanLike,
    *,
    run: Run,
    user_input: str,
    gateway: AgentGateway,
    capability_registry: CapabilityRegistry | None = None,
) -> ModeResult:
    """Run *plan* against *run* inside ``run.start()``."""
    with run.start() as run_ctx:
        result = await plan.run(
            run=run,
            user_input=user_input,
            gateway=gateway,
            capability_registry=capability_registry,
        )
        run_ctx.mark_succeeded()
    return result
