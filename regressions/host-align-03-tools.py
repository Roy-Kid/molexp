"""Public-API regression for host-align-03-tools."""

from __future__ import annotations

import asyncio

from molexp.harness.host.context import Context
from molexp.harness.host.plugins.tools import ToolBelt


async def _main() -> None:
    ctx = Context()
    belt = ToolBelt()
    belt.bind(ctx)
    ran = {"n": 0}

    async def probe() -> str:
        ran["n"] += 1
        return "hi"

    probe.__name__ = "probe"
    belt.register(probe, ctx)

    async def block(payload: object, nxt: object) -> str:
        del payload, nxt
        return "skipped"

    ctx.on("tools/pre-execute", block, mode="waterfall")
    assert await belt.execute("probe", {}) == "skipped"
    assert ran["n"] == 0
    print("host-align-03-tools: ok")


if __name__ == "__main__":
    asyncio.run(_main())
