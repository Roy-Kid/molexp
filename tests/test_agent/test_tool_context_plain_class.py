"""ToolContext + RegisteredTool plain-class tests.

Both types carry live runtime references (workspace, chat gateway,
memory store, tool callable) and are plain Python classes — not
BaseModel, not @dataclass, no ``arbitrary_types_allowed=True``.
"""

from __future__ import annotations

import asyncio
import dataclasses

from molexp.agent.tools.spec import (
    RegisteredTool,
    ToolContext,
    ToolResult,
    ToolSpec,
)


class TestToolContextPlainClass:
    def test_construct_minimal(self):
        ctx = ToolContext(workspace=object(), session_id="s1", turn_id="t1")
        assert ctx.session_id == "s1"
        assert ctx.turn_id == "t1"
        assert ctx.run is None
        assert ctx.memory is None
        assert ctx.chat is None

    def test_chat_is_assignable_post_construction(self):
        """ToolContext is mutable so the runner can attach a chat gateway."""
        ctx = ToolContext(workspace=object(), session_id="s", turn_id="t")
        sentinel = object()
        ctx.chat = sentinel
        assert ctx.chat is sentinel

    def test_carries_live_runtime_refs_without_escape_hatch(self):
        """A live `asyncio.Lock` flows through ToolContext fine."""
        lock = asyncio.Lock()
        ctx = ToolContext(workspace=lock, session_id="s", turn_id="t")
        assert ctx.workspace is lock

    def test_not_dataclass_not_basemodel(self):
        assert not dataclasses.is_dataclass(ToolContext)
        from pydantic import BaseModel

        assert not issubclass(ToolContext, BaseModel)


class TestRegisteredToolPlainClass:
    def test_construct(self):
        async def _fn(args, ctx):  # pragma: no cover - shape only
            return ToolResult(ok=True)

        spec = ToolSpec(name="t", description="x", input_schema={})
        rt = RegisteredTool(spec=spec, fn=_fn)
        assert rt.spec is spec
        assert rt.fn is _fn

    def test_not_dataclass_not_basemodel(self):
        assert not dataclasses.is_dataclass(RegisteredTool)
        from pydantic import BaseModel

        assert not issubclass(RegisteredTool, BaseModel)
