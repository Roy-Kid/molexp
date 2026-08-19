"""Function-level unit tests for the approvals inbox routes (routes/approvals.py).

Two route-module units are exercised directly, without booting the app:
- ``ApprovalDecisionRequest`` — the decision payload schema requires ``action``.
- ``stream_approval_events`` — the SSE route coroutine: its
  ``StreamingResponse`` body emits one ``changed`` frame per broadcast ping,
  the same ping a task suspension and a landed decision notify.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest


def _text(chunk: Any) -> str:
    return chunk if isinstance(chunk, str) else chunk.decode()


class TestApprovalDecisionRequest:
    def test_action_field_is_required(self) -> None:
        from pydantic import ValidationError

        from molexp.server.routes.approvals import ApprovalDecisionRequest

        ok = ApprovalDecisionRequest(requestId="r", action="approve")
        assert ok.action == "approve"
        ok2 = ApprovalDecisionRequest(requestId="r", action="revise", fieldValues={"a": 1})
        assert ok2.action == "revise"
        with pytest.raises(ValidationError):
            ApprovalDecisionRequest(requestId="r")  # type: ignore[call-arg]


class TestStreamApprovalEvents:
    # Driven by invoking the route coroutine and stepping its StreamingResponse
    # body against the broadcast primitive both the suspend path and the decide
    # route notify. (An endless SSE endpoint can't be consumed through
    # TestClient, which buffers a streaming response to completion on `stream()`
    # entry — hence the direct-coroutine unit here.)

    async def test_route_emits_a_changed_event_per_notification(self) -> None:
        """One ``changed`` SSE frame per broadcast ping — the same ping a task
        suspension and a landed decision emit."""
        from molexp.server.routes.approvals import stream_approval_events
        from molexp.services.approval_notify import notify_approvals_changed

        resp = await stream_approval_events()
        assert resp.media_type == "text/event-stream"
        assert resp.headers["cache-control"] == "no-cache"

        iterator = resp.body_iterator
        connected = await asyncio.wait_for(anext(iterator), timeout=1.0)
        assert "changed" in _text(connected)  # the connect frame

        ping = asyncio.ensure_future(anext(iterator))
        await asyncio.sleep(0.01)  # let the generator reach its subscription await
        notify_approvals_changed()

        frame = await asyncio.wait_for(ping, timeout=1.0)
        assert "changed" in _text(frame)
        await iterator.aclose()
