"""Process-wide server shutdown signal.

Long-lived SSE generators (approvals inbox, agent/plan task tails) hold open
HTTP connections. Uvicorn will not exit while those connections are open
unless either the generators stop or ``timeout_graceful_shutdown`` fires.

This module is the cooperative stop flag those generators poll, and the
wake-up path for in-process pub/sub waiters, so graceful shutdown finishes
in seconds instead of hanging until the browser tab dies.
"""

from __future__ import annotations

import asyncio
import threading

__all__ = [
    "is_shutting_down",
    "mark_shutting_down",
    "reset_shutdown_flag",
    "shutdown_event",
    "wait_or_shutdown",
]

_lock = threading.Lock()
_shutting_down = False
_event: asyncio.Event | None = None


def is_shutting_down() -> bool:
    """True once the FastAPI lifespan has entered its shutdown branch."""
    return _shutting_down


def mark_shutting_down() -> None:
    """Flip the flag and wake every waiter on the current event loop's Event."""
    global _shutting_down, _event
    with _lock:
        _shutting_down = True
        event = _event
    if event is not None and not event.is_set():
        event.set()


def reset_shutdown_flag() -> None:
    """Test/helper: clear the flag so a new serve cycle can start clean."""
    global _shutting_down, _event
    with _lock:
        _shutting_down = False
        _event = None


def shutdown_event() -> asyncio.Event:
    """Lazily create (or return) the loop-bound shutdown :class:`asyncio.Event`."""
    global _event
    with _lock:
        if _event is None:
            _event = asyncio.Event()
            if _shutting_down:
                _event.set()
        return _event


async def wait_or_shutdown(timeout: float) -> bool:
    """Sleep up to *timeout* seconds, or return early on shutdown.

    Returns ``True`` when shutdown was signalled, ``False`` when the full
    timeout elapsed.
    """
    if _shutting_down:
        return True
    event = shutdown_event()
    try:
        await asyncio.wait_for(event.wait(), timeout=timeout)
        return True
    except TimeoutError:
        return is_shutting_down()
