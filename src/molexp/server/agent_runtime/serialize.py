"""``AgentEvent`` → Server-Sent-Events frame serialization (spec 00c).

Pure ``str``-returning helpers — runtime objects never cross into
``server.schemas`` and this module imports only the wire event type plus
stdlib ``json``. Every frame is the SSE ``data:`` form ``data: {json}\\n\\n``.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from molexp.agent.events import AgentEvent


def event_to_sse_frame(event: AgentEvent) -> str:
    """Render one :data:`AgentEvent` as an SSE ``data:`` frame.

    The payload is the event's ``model_dump(mode="json")`` (so ``kind`` and the
    ISO ``timestamp`` survive); the consumer dispatches on ``kind``.
    """
    return f"data: {json.dumps(event.model_dump(mode='json'))}\n\n"


def done_frame() -> str:
    """Render the terminal ``done`` control frame (after ``mode_completed``)."""
    return f"data: {json.dumps({'type': 'done'})}\n\n"


def error_frame(message: str) -> str:
    """Render the single ``error`` control frame emitted on a turn failure."""
    return f"data: {json.dumps({'type': 'error', 'message': message})}\n\n"
