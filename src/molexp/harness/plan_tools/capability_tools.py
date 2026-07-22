"""The side-effecting plan tools — ``run_capability`` / ``run_acceptance_test``.

Thin, agent-facing tools that dispatch a registered capability by routing through
:func:`~molexp.harness.stages.invoke_capability.dispatch_capability` (imported as
a module global so tests can monkeypatch the delegation target). They never
re-implement the runner/subprocess materialization path — the whole
validate → resolve → side-effect gate → materialize → execute → promote flow
lives once, in ``dispatch_capability``. A denied side-effect gate raises
:class:`StageExecutionError` out of ``dispatch_capability`` and propagates
unchanged.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from molexp.harness.plan_tools.tool import PlanToolResult
from molexp.harness.stages.invoke_capability import dispatch_capability

if TYPE_CHECKING:
    from molexp.harness.core.run_context import HarnessRunContext
    from molexp.harness.executors import Executor
    from molexp.harness.stages.approval_gate import Approver

__all__ = ["ACCEPTANCE_TEST_CAPABILITY_ID", "run_acceptance_test", "run_capability"]

# The fixed capability id ``run_acceptance_test`` dispatches. Its concrete value
# is the tool's own concern; callers pass only the parameters.
ACCEPTANCE_TEST_CAPABILITY_ID = "molexp.acceptance_test"


async def run_capability(
    *,
    ctx: HarnessRunContext,
    capability_id: str,
    parameters: dict[str, object],
    executor: Executor | None = None,
    approve: Approver | None = None,
    timeout_s: int = 3600,
) -> PlanToolResult:
    """Dispatch one registered capability and wrap its persisted result.

    Routes through :func:`dispatch_capability` — no second runner-materialization
    path. The returned :class:`PlanToolResult` carries the dispatched
    ``capability_invocation_result`` id as its ``artifact_ref``.

    Args:
        ctx: The harness run context (must carry a ``capability_registry``).
        capability_id: The id of the capability to invoke.
        parameters: The call parameters.
        executor: The executor to run the invocation (``None`` ⇒ the dispatch
            default; inject :class:`DryRunExecutor` to no-op).
        approve: The approver consulted by the pre-dispatch side-effect gate.
        timeout_s: The runner subprocess timeout in seconds.

    Returns:
        A successful :class:`PlanToolResult` whose ``artifact_ref`` is the
        persisted ``capability_invocation_result`` id.

    Raises:
        StageExecutionError: If the side-effect gate denies the invocation.
    """
    ref = await dispatch_capability(
        ctx,
        capability_id,
        parameters,
        executor=executor,
        approve=approve,
        timeout_s=timeout_s,
    )
    return PlanToolResult(
        ok=True,
        summary=f"invoked capability {capability_id!r}",
        data={"capability_id": capability_id},
        artifact_ref=ref.id,
    )


async def run_acceptance_test(
    *,
    ctx: HarnessRunContext,
    parameters: dict[str, object],
    executor: Executor | None = None,
    approve: Approver | None = None,
    timeout_s: int = 3600,
) -> PlanToolResult:
    """Dispatch the fixed acceptance-test capability and wrap its result.

    A specialization of :func:`run_capability` pinned to
    :data:`ACCEPTANCE_TEST_CAPABILITY_ID`; it routes through
    :func:`dispatch_capability` just the same.

    Args:
        ctx: The harness run context (must carry a ``capability_registry``).
        parameters: The acceptance-test parameters.
        executor: The executor to run the invocation (``None`` ⇒ the dispatch
            default; inject :class:`DryRunExecutor` to no-op).
        approve: The approver consulted by the pre-dispatch side-effect gate.
        timeout_s: The runner subprocess timeout in seconds.

    Returns:
        A successful :class:`PlanToolResult` whose ``artifact_ref`` is the
        persisted ``capability_invocation_result`` id.

    Raises:
        StageExecutionError: If the side-effect gate denies the invocation.
    """
    ref = await dispatch_capability(
        ctx,
        ACCEPTANCE_TEST_CAPABILITY_ID,
        parameters,
        executor=executor,
        approve=approve,
        timeout_s=timeout_s,
    )
    return PlanToolResult(
        ok=True,
        summary="ran acceptance test",
        data={"capability_id": ACCEPTANCE_TEST_CAPABILITY_ID},
        artifact_ref=ref.id,
    )
