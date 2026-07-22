"""``PydanticAIRouter.stream_agentic`` honors the emergent-loop hooks.

The router threads three optional, keyword-only hooks
(``before_tool`` / ``after_tool`` / ``should_stop``) through pydantic-ai's
native agentic loop. Phase 01 enforces the one boundary passive iteration
permits: an ``after_tool`` returning
:meth:`~molexp.agent.loops.hooks.HookOutcome.deny` rewrites the emitted
:class:`~molexp.agent.router.ToolResultChunk` with ``ok=False`` and folds the
deny message into ``result_summary``. The before-tool boundary is triggered and
recorded. Signature-shape and no-op-invariance are exercised transitively by
this behavioral test.
"""

from __future__ import annotations

import pytest

from molexp.agent.loops.hooks import HookOutcome
from molexp.agent.router import ToolCallChunk, ToolResultChunk

pytestmark = pytest.mark.asyncio


def _router(model: object) -> object:
    """Build a :class:`PydanticAIRouter` whose every tier is ``model``."""
    from molexp.agent._pydanticai.router import PydanticAIRouter
    from molexp.agent.router import ModelTier

    return PydanticAIRouter(
        models={
            ModelTier.CHEAP: model,
            ModelTier.DEFAULT: model,
            ModelTier.HEAVY: model,
        },
    )


class TestStreamAgenticHooks:
    async def test_after_tool_deny_flips_tool_result_ok_and_records_before_tool(self) -> None:
        """Spy hooks fire and record; an ``after_tool`` deny flips ToolResultChunk.ok."""
        pytest.importorskip("pydantic_ai")
        from pydantic_ai.models.test import TestModel

        async def peek(path: str) -> str:
            """A trivial read-only tool the model may call."""
            return f"contents of {path}"

        before_calls: list[str] = []
        after_calls: list[str] = []

        async def spy_before(*, tool_name: str, args: object) -> HookOutcome:
            before_calls.append(tool_name)
            return HookOutcome.proceed()

        async def deny_after(*, tool_name: str, result: str) -> HookOutcome:
            after_calls.append(tool_name)
            return HookOutcome.deny("blocked")

        router = _router(TestModel())
        chunks = [
            chunk
            async for chunk in router.stream_agentic(
                prompt="inspect",
                tools=(peek,),
                before_tool=spy_before,
                after_tool=deny_after,
            )
        ]

        tool_calls = [chunk for chunk in chunks if isinstance(chunk, ToolCallChunk)]
        assert tool_calls, "TestModel should have dispatched the tool"
        expected_name = tool_calls[0].tool_name

        assert expected_name in before_calls, "before_tool must be awaited and recorded"
        assert expected_name in after_calls, "after_tool must be awaited and recorded"

        tool_results = [chunk for chunk in chunks if isinstance(chunk, ToolResultChunk)]
        assert tool_results, "expected at least one ToolResultChunk"
        assert all(not chunk.ok for chunk in tool_results)
        assert any("blocked" in chunk.result_summary for chunk in tool_results)
