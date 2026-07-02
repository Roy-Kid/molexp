"""``drive_plan_mode`` — the ONE way CLI and server run a PlanMode pipeline.

Wraps ``mode.run(...)`` in the run's own lifecycle (``run.start()``), so a
plan Run's workspace status is honest: ``running`` while the pipeline
executes (with the ownership stamp + heartbeat every workflow run gets),
``succeeded`` when all stages complete, ``failed`` when a stage raises.
Without this, a plan run stayed ``pending`` forever and every status
surface in the UI contradicted the 13 green pipeline steps.

Plan re-entry stays governed by the stage ledger, NOT the run verbs: a
re-run against a ``succeeded`` plan run enters the lifecycle again (ledger
hits make it cheap), unlike ``molexp run`` which refuses succeeded runs.
A concurrently *running* plan on the same Run is still refused by the
lifecycle's ownership claim — one Run, one live execution.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    from molexp.harness import CapabilityRegistry, ModeResult
    from molexp.harness.gateways.gateway import AgentGateway
    from molexp.workspace.run import Run

__all__ = ["drive_plan_mode"]


class _ModeLike(Protocol):
    async def run(
        self,
        *,
        run: Any,  # noqa: ANN401 — workspace Run, duck-typed for tests
        user_input: str,
        gateway: Any,  # noqa: ANN401
        capability_registry: Any = None,  # noqa: ANN401
    ) -> Any: ...  # noqa: ANN401


async def drive_plan_mode(
    mode: _ModeLike,
    *,
    run: Run,
    user_input: str,
    gateway: AgentGateway,
    capability_registry: CapabilityRegistry | None = None,
) -> ModeResult:
    """Run *mode* against *run* inside the run lifecycle; return its result.

    Success marks the run ``succeeded`` explicitly (the lifecycle never
    defaults to success); a raising stage propagates after the lifecycle
    records ``failed``.
    """
    with run.start() as run_ctx:
        result = await mode.run(
            run=run,
            user_input=user_input,
            gateway=gateway,
            capability_registry=capability_registry,
        )
        run_ctx.mark_succeeded()
    return result
