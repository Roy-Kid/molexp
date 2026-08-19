"""Public-API regression for host-approval-01-request."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime

from molexp.harness.errors import ApprovalPendingError
from molexp.harness.host import Host, Keys
from molexp.harness.host.plugins.approval import ApprovalPlugin
from molexp.harness.schemas import ApprovalRequest


class _Store:
    def __init__(self) -> None:
        self.pending: list[object] = []

    def granted_decision_for(self, request_id: str) -> None:
        del request_id
        return

    def record_pending(self, run_id: str, request: object) -> None:
        del run_id
        self.pending.append(request)

    def record_decision(self, decision: object) -> None:
        del decision

    def pending(self, run_id: str) -> list[object]:
        del run_id
        return []


async def _main() -> None:
    host = Host()
    store = _Store()
    host.ctx.provide(Keys.APPROVAL, store)
    host.ctx.provide(Keys.RUN_ID, "r")
    host.mount(ApprovalPlugin())
    request = ApprovalRequest(
        id="req-1",
        intent="overwrite",
        reason="reg",
        triggered_by_policy="side_effects_present",
        metadata={},
        created_at=datetime.now(tz=UTC),
    )
    try:
        await host.ctx.waterfall("approval/request", request)
    except ApprovalPendingError:
        assert store.pending
        print("host-approval-01-request: ok")
        return
    raise AssertionError("expected ApprovalPendingError")


if __name__ == "__main__":
    asyncio.run(_main())
