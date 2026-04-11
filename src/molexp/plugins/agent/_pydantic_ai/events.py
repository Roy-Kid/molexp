"""Event mapping: pydantic-ai AgentStreamEvent → molexp SessionEvent.

The pydantic-ai event stream contains low-level model events
(text deltas, tool calls, tool results). We map the relevant
ones to molexp's higher-level session event types.

Mapping table (§6.6 of proposal):
    FunctionToolCallEvent  → ToolCallEvent
    FunctionToolResultEvent → ToolResultEvent
    FinalResultEvent       → (ignored — SessionCompletedEvent is emitted separately)
    PartStartEvent (text)  → ObservationEvent (when agent produces text)
"""

from __future__ import annotations

from typing import Any

from pydantic_ai.messages import (
    AgentStreamEvent,
    FunctionToolCallEvent,
    FunctionToolResultEvent,
    PartStartEvent,
)

from ..types import (
    ObservationEvent,
    SessionEvent,
    ToolCallEvent,
    ToolResultEvent,
)


def map_stream_event(event: AgentStreamEvent) -> SessionEvent | None:
    """Map a pydantic-ai stream event to a molexp SessionEvent.

    Returns None for events that don't have a molexp equivalent
    (e.g., streaming text deltas, thinking parts).
    """
    if isinstance(event, FunctionToolCallEvent):
        args = event.part.args
        if isinstance(args, str):
            try:
                import json
                args = json.loads(args)
            except Exception:
                args = {"raw": args}
        return ToolCallEvent(
            tool_name=event.part.tool_name,
            args=args if isinstance(args, dict) else {"value": args},
        )

    if isinstance(event, FunctionToolResultEvent):
        result_part = event.result
        result_value: Any = getattr(result_part, "content", str(result_part))
        return ToolResultEvent(
            tool_name=getattr(result_part, "tool_name", "unknown"),
            result=result_value,
        )

    if isinstance(event, PartStartEvent):
        # Emit text parts as observations (agent's reasoning/response text)
        part = event.part
        if hasattr(part, "content") and isinstance(part.content, str) and part.content.strip():
            return ObservationEvent(content=part.content[:500])

    # FinalResultEvent, PartDeltaEvent, PartEndEvent → not mapped
    return None
