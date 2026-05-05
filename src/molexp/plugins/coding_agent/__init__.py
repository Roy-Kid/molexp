"""Provider-neutral coding-agent plugin contract.

Concrete coding-agent providers (``agent_claude``, ``agent_codex``, …) live as
peer plugins under :mod:`molexp.plugins`. They each spawn / drive a coding-agent
runtime — Claude CLI subprocess, Codex app-server JSON-RPC, etc. — and translate
provider-specific protocol messages into a common stream of normalized events
plus a single :class:`TurnResult` per turn.

This module owns only the **shared contract** every provider implements:

- :class:`CodingAgentClient` — runtime-checkable Protocol.
- :class:`TurnResult` — frozen dataclass, the per-turn outcome.
- :class:`AgentError` (and :class:`AgentTurnInputRequiredError`) — error
  hierarchy raised by providers and surfaced to callers.
- :data:`AgentEventCallback` — type alias for the on_event sink.

Concrete provider implementations must not import from each other; they only
import from this module.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Protocol, runtime_checkable

__all__ = [
    "AgentError",
    "AgentEventCallback",
    "AgentTurnInputRequiredError",
    "CodingAgentClient",
    "TurnResult",
]


AgentEventCallback = Callable[[dict[str, Any]], Awaitable[None] | None]
"""Sink for normalized provider events.

Providers call ``await on_event(payload)`` (or ``on_event(payload)`` if the
return value is non-awaitable). Payloads are plain ``dict`` records with a
``"event"`` key plus arbitrary provider-specific fields. The orchestrator
consumes these to drive workflow state and dashboards.
"""


@dataclass(frozen=True)
class TurnResult:
    """Outcome of one coding-agent turn, normalized across providers.

    Attributes:
        thread_id: Stable identifier of the agent thread the turn ran on.
            Reused for every continuation turn within a session.
        turn_id: Unique identifier of this single turn within the thread.
        status: ``"completed"`` on success, otherwise a provider-specific
            failure label (e.g. ``"failed"``).
    """

    thread_id: str
    turn_id: str
    status: str


class AgentError(RuntimeError):
    """Generic provider-side error raised by a :class:`CodingAgentClient`.

    Concrete subclasses may carry structured fields; for now the message
    string and ``code`` class attribute are the public contract.
    """

    code: str = "agent_error"


class AgentTurnInputRequiredError(AgentError):
    """The provider asked the user to supply input mid-turn.

    The orchestrator does not run interactive turns; it surfaces this as a
    permanent failure for the caller to handle (typically by escalating to
    a human or aborting the workflow).
    """

    code: str = "turn_input_required"


@runtime_checkable
class CodingAgentClient(Protocol):
    """Provider-neutral coding-agent client interface.

    Concrete implementations own their own subprocess / connection lifecycle
    and translate provider-specific protocol messages into the normalized
    event shape consumed by the molexp workflow.

    Lifecycle:
        1. ``await client.start()`` — acquire provider-level resources
           (spawn long-lived subprocess, run handshake, etc.). Idempotent
           on providers whose subprocesses are per-turn.
        2. ``thread_id = await client.start_thread()`` — open a new agent
           thread; the returned id MUST stay stable across continuation
           turns in the same session.
        3. Repeat: ``result = await client.run_turn(thread_id, prompt)``.
        4. ``await client.close()`` — release any provider resources.

    Attributes:
        pid: PID of the underlying subprocess if there is exactly one
            long-lived process, ``None`` otherwise. Used for log / dashboard
            display only.
    """

    pid: int | None

    async def start(self) -> None:
        """Acquire any provider-level resources."""

    async def start_thread(self) -> str:
        """Open a new agent thread and return its identifier."""

    async def run_turn(self, thread_id: str, prompt: str) -> TurnResult:
        """Run one turn on ``thread_id`` with ``prompt`` and return the outcome."""

    async def close(self) -> None:
        """Release provider-level resources."""
