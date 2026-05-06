"""Provider-neutral coding-agent plugin contract.

Concrete coding-agent providers (``agent_claude``, ``agent_codex``, …) live as
peer plugins under :mod:`molexp.plugins`. They each spawn / drive a coding-agent
runtime — Claude CLI subprocess, Codex app-server JSON-RPC, etc. — and translate
provider-specific protocol messages into a common stream of normalized events
plus a single :class:`TurnResult` per turn.

This module owns the **shared contract** every provider implements:

- :class:`CodingAgentClient` — runtime-checkable Protocol.
- :class:`TurnResult` — frozen dataclass, the per-turn outcome.
- :class:`AgentError` (and :class:`AgentTurnInputRequiredError`) — error
  hierarchy raised by providers and surfaced to callers.
- :data:`AgentEventCallback` — type alias for the on_event sink.

Plus a few **shared subprocess helpers** the providers reuse so the lifecycle
contract (event emission, stderr drain, graceful termination) lives in one
place instead of being copy-pasted into each provider:

- :func:`emit_event` — call ``on_event(payload)`` and await if it's a coroutine
- :func:`drain_stderr` — readline loop emitting ``{"event": "stderr", ...}``
- :func:`terminate_subprocess` — graceful terminate → wait → kill on timeout

Concrete provider implementations must not import from each other; they only
import from this module.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any, Awaitable, Callable, Literal, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict

__all__ = [
    "AgentError",
    "AgentEventCallback",
    "AgentTurnInputRequiredError",
    "CodingAgentClient",
    "TurnResult",
    "drain_stderr",
    "emit_event",
    "terminate_subprocess",
]


AgentEventCallback = Callable[[dict[str, Any]], Awaitable[None] | None]
"""Sink for normalized provider events. Sync or async; payloads are dicts
with an ``"event"`` key plus provider-specific fields."""


class TurnResult(BaseModel):
    """Outcome of one coding-agent turn, normalized across providers.

    Attributes:
        thread_id: Stable identifier of the agent thread the turn ran on.
            Reused for every continuation turn within a session.
        turn_id: Unique identifier of this single turn within the thread.
        status: ``"completed"`` on success, ``"failed"`` on provider-side
            failure.
    """

    model_config = ConfigDict(frozen=True)

    thread_id: str
    turn_id: str
    status: Literal["completed", "failed"]


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
        pid: PID of the underlying subprocess while one is live, ``None``
            otherwise.
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


# ── shared subprocess helpers ──────────────────────────────────────────────


async def emit_event(callback: AgentEventCallback, payload: dict[str, Any]) -> None:
    """Stamp ``payload`` with ``timestamp`` and dispatch to ``callback``.

    Awaits the callback if it returns a coroutine; otherwise treats the
    return value as a fire-and-forget result. Mutates ``payload`` in place
    (adds ``timestamp`` if not already present).
    """
    payload.setdefault("timestamp", time.time())
    result = callback(payload)
    if asyncio.iscoroutine(result):
        await result


async def drain_stderr(
    proc: asyncio.subprocess.Process,
    on_event: AgentEventCallback,
    *,
    chunk_limit: int = 1000,
) -> None:
    """Forward subprocess stderr to ``on_event`` as ``"stderr"`` events.

    Reads ``proc.stderr`` line by line until EOF. Each chunk is decoded
    with replacement, truncated to ``chunk_limit`` bytes, and emitted.
    Cancellation exits cleanly (returns instead of raising).
    """
    if proc.stderr is None:
        return
    try:
        while True:
            chunk = await proc.stderr.readline()
            if not chunk:
                return
            await emit_event(
                on_event,
                {"event": "stderr", "message": chunk.decode("utf-8", "replace")[:chunk_limit]},
            )
    except asyncio.CancelledError:
        return


async def terminate_subprocess(
    proc: asyncio.subprocess.Process | None,
    *,
    timeout: float = 5.0,
) -> None:
    """Gracefully terminate ``proc`` with a kill fallback.

    Sends SIGTERM (``proc.terminate()``) and waits up to ``timeout`` seconds
    for exit; on timeout sends SIGKILL (``proc.kill()``) and waits
    indefinitely. No-op when ``proc`` is ``None`` or already exited.
    """
    if proc is None or proc.returncode is not None:
        return
    proc.terminate()
    try:
        await asyncio.wait_for(proc.wait(), timeout=timeout)
    except asyncio.TimeoutError:
        proc.kill()
        await proc.wait()
