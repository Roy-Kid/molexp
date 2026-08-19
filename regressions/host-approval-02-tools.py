"""Public-API regression for host-approval-02-tools."""

from __future__ import annotations

import asyncio

from molexp.harness.errors import ApprovalPendingError
from molexp.harness.host import Context, Host, Keys
from molexp.harness.host.plugins.approval import ApprovalPlugin
from molexp.harness.host.plugins.tools import ToolBelt


class _Store:
    def granted_decision_for(self, request_id: str) -> None:
        del request_id
        return

    def record_pending(self, run_id: str, request: object) -> None:
        del run_id, request

    def record_decision(self, decision: object) -> None:
        del decision

    def pending(self, run_id: str) -> list[object]:
        del run_id
        return []


async def _main() -> None:
    ctx = Context()
    ctx.provide(Keys.APPROVAL, _Store())
    ctx.provide(Keys.RUN_ID, "r")
    belt = ToolBelt()
    belt.bind(ctx)
    ctx.provide(Keys.TOOLS, belt)
    host = Host()
    host.ctx = ctx
    host.mount(ApprovalPlugin())
    ran = {"n": 0}

    async def boom() -> str:
        ran["n"] += 1
        return "x"

    boom.__name__ = "boom"
    boom.side_effects = ["overwrite"]  # type: ignore[attr-defined]
    belt.register(boom, ctx)
    try:
        await belt.execute("boom", {})
    except ApprovalPendingError:
        assert ran["n"] == 0
        print("host-approval-02-tools: ok")
        return
    raise AssertionError("expected ApprovalPendingError")


if __name__ == "__main__":
    asyncio.run(_main())
