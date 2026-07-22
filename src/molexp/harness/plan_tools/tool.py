"""``PlanTool`` + ``PlanToolResult`` — the plan-tool descriptor and its payload.

The two value types shared by every process-internal plan tool:

* :class:`PlanTool` — a frozen dataclass (not pydantic: it holds a live async
  callable) describing one tool an agent loop can invoke — its name, human
  description, JSON ``input_schema``, declared ``side_effects`` (empty ⇒
  read-only, gate-bypassed), and the coroutine ``fn`` that runs it.
* :class:`PlanToolResult` — a frozen pydantic model returned by every tool: an
  ``ok`` flag, a human ``summary``, a free-form ``data`` payload, and an optional
  ``artifact_ref`` pointing at a persisted harness artifact.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field

from pydantic import BaseModel, ConfigDict, Field

__all__ = ["PlanTool", "PlanToolFn", "PlanToolResult"]


class PlanToolResult(BaseModel):
    """The uniform result payload every plan tool returns.

    Attributes:
        ok: Whether the tool succeeded.
        summary: A short human-readable outcome line.
        data: A free-form structured payload (JSON-serializable).
        artifact_ref: The id of a persisted harness artifact the tool produced,
            or ``None`` when the tool wrote no artifact.
    """

    model_config = ConfigDict(frozen=True)

    ok: bool
    summary: str = ""
    data: dict[str, object] = Field(default_factory=dict)
    artifact_ref: str | None = None


# A plan tool body: a coroutine invoked with keyword arguments that resolves to
# a :class:`PlanToolResult`.
PlanToolFn = Callable[..., Awaitable[PlanToolResult]]


@dataclass(frozen=True)
class PlanTool:
    """An immutable descriptor for one process-internal plan tool.

    A frozen dataclass rather than a pydantic model because it holds a live
    async callable (``fn``) — a runtime container, exactly the case CLAUDE.md
    reserves for plain/dataclass types over pydantic.

    Attributes:
        name: The tool's stable identifier (also its adapter ``__name__``).
        description: A human description (also its adapter ``__doc__``).
        fn: The coroutine that runs the tool and returns a
            :class:`PlanToolResult`.
        input_schema: The JSON schema describing the tool's keyword arguments.
        side_effects: The declared destructive side effects; an empty list marks
            the tool read-only and bypasses the approval gate.
    """

    name: str
    description: str
    fn: PlanToolFn
    input_schema: dict[str, object] = field(default_factory=dict)
    side_effects: list[str] = field(default_factory=list)
