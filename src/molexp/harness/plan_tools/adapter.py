"""``as_loop_tool`` — adapt a :class:`PlanTool` into an agent-loop callable.

Wraps one :class:`PlanTool` into an opaque ``async`` callable an agent loop can
accept: it copies the tool's ``name``/``description`` onto the returned function's
``__name__``/``__doc__`` (so pydantic-ai can derive a schema), gates the tool's
declared ``side_effects`` **before** running the body (via
:func:`~molexp.harness.policy.side_effect_gate.enforce_side_effect_approvals`,
referenced as a module global so tests can spy on it — an empty ``side_effects``
list bypasses the gate and writes no approval artifact), and records the outcome
as a ``tool_*`` event on ``ctx.event_log`` **after** the body runs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from molexp.harness.plan_tools.tool import PlanTool, PlanToolFn, PlanToolResult
from molexp.harness.policy import enforce_side_effect_approvals

if TYPE_CHECKING:
    from molexp.harness.core.run_context import HarnessRunContext
    from molexp.harness.stages.approval_gate import Approver

__all__ = ["as_loop_tool"]


@dataclass(frozen=True)
class _ToolGateItem:
    """A minimal ``(id, side_effects)`` view of a :class:`PlanTool` for the gate.

    The side-effect gate screens items exposing ``id`` + ``side_effects``; a
    :class:`PlanTool` names its id ``name``, so this adapts it to the gate's
    expected shape without introducing a new gate contract.
    """

    id: str
    side_effects: list[str]


def as_loop_tool(
    tool: PlanTool,
    *,
    ctx: HarnessRunContext,
    approve: Approver | None = None,
) -> PlanToolFn:
    """Adapt one :class:`PlanTool` into an opaque async agent-loop callable.

    The returned coroutine carries ``tool.name`` as ``__name__`` and
    ``tool.description`` as ``__doc__`` so a loop can derive a schema. On each
    call it gates the tool's ``side_effects`` before running ``tool.fn`` (a
    read-only tool bypasses the gate — no approval artifact) and records a
    ``tool_completed``/``tool_failed`` event on ``ctx.event_log`` afterward.

    Args:
        tool: The plan tool to adapt.
        ctx: The harness run context (its event log records the after-hook and
            its stores back any side-effect gate).
        approve: The approver consulted by the pre-body side-effect gate for a
            tool that declares ``side_effects``.

    Returns:
        An ``async`` callable that runs the tool and returns its
        :class:`PlanToolResult`.
    """
    gate_item = _ToolGateItem(id=tool.name, side_effects=list(tool.side_effects))

    async def _adapted(**kwargs: object) -> PlanToolResult:
        # Before-hook: gate declared side effects. Empty ⇒ read-only bypass
        # (no gate runs, no approval artifact is written).
        await enforce_side_effect_approvals([gate_item], ctx=ctx, approve=approve)
        result = await tool.fn(**kwargs)
        # After-hook: record the tool outcome on the event log.
        ctx.event_log.append(
            run_id=ctx.run_id,
            type="tool_completed" if result.ok else "tool_failed",
            actor=tool.name,
            payload={"tool": tool.name, "summary": result.summary},
        )
        return result

    _adapted.__name__ = tool.name
    _adapted.__doc__ = tool.description
    return _adapted
