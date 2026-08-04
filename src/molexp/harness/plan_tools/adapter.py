"""``as_loop_tool`` — adapt a :class:`PlanTool` into an agent-loop callable.

Wraps one :class:`PlanTool` into an opaque ``async`` callable an agent loop can
accept: it copies the tool's ``name``/``description`` onto the returned function's
``__name__``/``__doc__`` (so pydantic-ai can derive a schema), **injects** the
harness ``ctx`` and board handle (the LLM never supplies them), gates the tool's
declared ``side_effects`` **before** running the body, and records the outcome
as a ``tool_*`` event on ``ctx.event_log`` **after** the body runs.
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from molexp.harness.plan_tools.tool import PlanTool, PlanToolFn, PlanToolResult
from molexp.harness.policy import enforce_side_effect_approvals

if TYPE_CHECKING:
    from molexp.harness.core.run_context import HarnessRunContext
    from molexp.harness.stages.approval_gate import Approver

__all__ = ["as_loop_tool"]


@dataclass(frozen=True)
class _ToolGateItem:
    """A minimal ``(id, side_effects)`` view of a :class:`PlanTool` for the gate."""

    id: str
    side_effects: list[str]


def _public_parameters(fn: PlanToolFn) -> list[str]:
    """Parameter names the LLM may supply (excludes injected ``ctx`` / ``board``)."""
    names: list[str] = []
    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError):
        return names
    for name, param in sig.parameters.items():
        if name in {"ctx", "board", "self", "cls"}:
            continue
        if param.kind in (
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        ):
            continue
        names.append(name)
    return names


def as_loop_tool(
    tool: PlanTool,
    *,
    ctx: HarnessRunContext,
    board: object | None = None,
    approve: Approver | None = None,
) -> PlanToolFn:
    """Adapt one :class:`PlanTool` into an opaque async agent-loop callable.

    The returned coroutine carries ``tool.name`` as ``__name__`` and
    ``tool.description`` as ``__doc__``. On each call it:

    1. Gates declared ``side_effects`` (empty ⇒ read-only bypass).
    2. Injects ``ctx`` / ``board`` into the tool body (stripping any
       LLM-supplied values for those names).
    3. Records ``tool_completed`` / ``tool_failed`` on ``ctx.event_log``.

    Args:
        tool: The plan tool to adapt.
        ctx: Harness run context (event log + stores).
        board: Production :class:`~molexp.harness.plan_tools.TaskBoardHandle`
            (typically :class:`~molexp.harness.plan.disk_board.DiskTaskBoard`).
            Required for board tools; optional for pure capability tools.
        approve: Approver for tools that declare ``side_effects``.
    """
    gate_item = _ToolGateItem(id=tool.name, side_effects=list(tool.side_effects))
    public_params = _public_parameters(tool.fn)

    async def _adapted(**kwargs: object) -> PlanToolResult:
        await enforce_side_effect_approvals([gate_item], ctx=ctx, approve=approve)
        # Never let the model invent harness internals.
        clean = {k: v for k, v in kwargs.items() if k not in {"ctx", "board"}}
        call: dict[str, Any] = dict(clean)
        try:
            sig = inspect.signature(tool.fn)
            params = sig.parameters
        except (TypeError, ValueError):
            params = {}
        if "ctx" in params:
            call["ctx"] = ctx
        if "board" in params:
            if board is None:
                return PlanToolResult(
                    ok=False,
                    summary=f"tool {tool.name!r} requires a task board handle",
                    data={"error": "board_missing"},
                )
            call["board"] = board
        try:
            result = await tool.fn(**call)
        except TypeError as exc:
            return PlanToolResult(
                ok=False,
                summary=f"tool {tool.name!r} call failed: {exc}",
                data={"error": "type_error", "detail": str(exc)},
            )
        except Exception as exc:  # tool bodies surface as failed results, not crashes
            ctx.event_log.append(
                run_id=ctx.run_id,
                type="tool_failed",
                actor=tool.name,
                payload={"tool": tool.name, "summary": str(exc)},
            )
            return PlanToolResult(
                ok=False,
                summary=f"tool {tool.name!r} raised {type(exc).__name__}: {exc}",
                data={"error": type(exc).__name__, "detail": str(exc)},
            )
        ctx.event_log.append(
            run_id=ctx.run_id,
            type="tool_completed" if result.ok else "tool_failed",
            actor=tool.name,
            payload={"tool": tool.name, "summary": result.summary},
        )
        return result

    _adapted.__name__ = tool.name
    # Keep tool.description as the docstring (pydantic-ai / tests pin it).
    # Public parameters are exposed via annotations for schema derivation.
    _adapted.__doc__ = tool.description
    if public_params:
        ann: dict[str, Any] = dict.fromkeys(public_params, object)
        ann["return"] = PlanToolResult
        _adapted.__annotations__ = ann
    return _adapted
