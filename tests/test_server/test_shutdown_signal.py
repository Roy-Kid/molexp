"""Server shutdown signal + approval SSE cooperative stop."""

from __future__ import annotations

import asyncio

import pytest

from molexp.server.shutdown import (
    is_shutting_down,
    mark_shutting_down,
    reset_shutdown_flag,
    wait_or_shutdown,
)
from molexp.services.approval_notify import (
    close_approval_subscribers,
    notify_approvals_changed,
    reset_approval_subscribers,
    subscribe_approvals_changed,
)


@pytest.fixture(autouse=True)
def _clean_flags() -> None:
    reset_shutdown_flag()
    reset_approval_subscribers()
    yield
    reset_shutdown_flag()
    reset_approval_subscribers()


@pytest.mark.asyncio
async def test_wait_or_shutdown_returns_true_when_flagged() -> None:
    mark_shutting_down()
    assert is_shutting_down()
    assert await wait_or_shutdown(5.0) is True


@pytest.mark.asyncio
async def test_wait_or_shutdown_times_out_when_idle() -> None:
    assert await wait_or_shutdown(0.05) is False


@pytest.mark.asyncio
async def test_approval_subscribe_ends_on_close() -> None:
    async def _consumer() -> list[None]:
        out: list[None] = []
        async for _ in subscribe_approvals_changed():
            out.append(None)
            if len(out) >= 1:
                break
        return out

    task = asyncio.create_task(_consumer())
    await asyncio.sleep(0.02)
    notify_approvals_changed()
    got = await asyncio.wait_for(task, timeout=1.0)
    assert len(got) == 1

    # After close, new subscribers must not hang forever.
    close_approval_subscribers()

    async def _after_close() -> int:
        n = 0
        async for _ in subscribe_approvals_changed():
            n += 1
        return n

    assert await asyncio.wait_for(_after_close(), timeout=1.0) == 0
