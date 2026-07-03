"""In-process approval-change broadcast — the SSE signal source.

A minimal asyncio pub/sub: task drivers call :func:`notify_approvals_changed`
when a task suspends on a pending approval or a decision lands; the server's
``GET /api/approvals/events`` SSE route holds a subscription and re-emits a
``changed`` ping so the UI refetches the inbox.

Deliberately payload-free (a ping, not a data channel): the inbox list is the
single source of truth and the UI always refetches it — no state to keep
consistent here. Subscribers that fall behind are never blocked on; the queue
is bounded at 1 and a pending ping coalesces with the next (an SSE consumer
only needs to know "something changed since you last looked").

Lives in ``services`` (not ``server``) because the *emitters* are the
services-layer task drivers — the server route is just one subscriber, and
services must never import server.
"""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import AsyncIterator

__all__ = ["notify_approvals_changed", "subscribe_approvals_changed"]


_subscribers: set[asyncio.Queue[None]] = set()


def notify_approvals_changed() -> None:
    """Ping every live subscriber (never blocks; coalesces backlogged pings)."""
    for queue in list(_subscribers):
        with contextlib.suppress(asyncio.QueueFull):
            queue.put_nowait(None)


async def subscribe_approvals_changed() -> AsyncIterator[None]:
    """Yield once per approval change until the consumer disconnects."""
    queue: asyncio.Queue[None] = asyncio.Queue(maxsize=1)
    _subscribers.add(queue)
    try:
        while True:
            yield await queue.get()
    finally:
        _subscribers.discard(queue)
