"""``PydanticAIRouter.stream_agentic`` — agentic-loop streaming (ac-003).

Drives the router's emergent loop with pydantic-ai's offline
``TestModel`` / ``FunctionModel`` plus one tool, and asserts the
SDK-free :data:`~molexp.agent.router.AgenticChunk` translation:
text deltas, tool-call + tool-result chunks, terminal ``FinalChunk``.
"""

from __future__ import annotations

import pytest

from molexp.agent.router import (
    FinalChunk,
    ModelTier,
    TextDeltaChunk,
    ToolCallChunk,
    ToolResultChunk,
)

pytestmark = pytest.mark.asyncio


def _router(model: object) -> object:
    """Build a :class:`PydanticAIRouter` whose every tier is ``model``."""
    from molexp.agent._pydanticai.router import PydanticAIRouter

    return PydanticAIRouter(
        models={
            ModelTier.CHEAP: model,
            ModelTier.DEFAULT: model,
            ModelTier.HEAVY: model,
        },
    )


async def test_stream_agentic_dispatches_tool_and_ends_with_final() -> None:
    """TestModel calls the supplied tool, then the loop ends in a FinalChunk."""
    pytest.importorskip("pydantic_ai")
    from pydantic_ai.models.test import TestModel

    seen: list[str] = []

    async def peek(path: str) -> str:
        """A trivial read-only tool the model may call."""
        seen.append(path)
        return f"contents of {path}"

    router = _router(TestModel())
    chunks = [
        chunk async for chunk in router.stream_agentic(prompt="inspect the project", tools=(peek,))
    ]

    assert seen, "TestModel should have dispatched the tool"
    kinds = {type(chunk) for chunk in chunks}
    assert ToolCallChunk in kinds
    assert ToolResultChunk in kinds
    assert any(isinstance(chunk, TextDeltaChunk) for chunk in chunks)
    assert isinstance(chunks[-1], FinalChunk)
    assert sum(isinstance(chunk, FinalChunk) for chunk in chunks) == 1


async def test_stream_agentic_chunk_order_is_call_then_result_then_final() -> None:
    """The chunk stream is ordered: tool-call → tool-result → … → FinalChunk."""
    pytest.importorskip("pydantic_ai")
    from pydantic_ai.models.test import TestModel

    async def peek(path: str) -> str:
        return f"contents of {path}"

    router = _router(TestModel())
    chunks = [chunk async for chunk in router.stream_agentic(prompt="inspect", tools=(peek,))]

    call_chunks = [c for c in chunks if isinstance(c, ToolCallChunk)]
    result_chunks = [c for c in chunks if isinstance(c, ToolResultChunk)]
    assert len(call_chunks) == 1
    assert call_chunks[0].tool_name == "peek"
    assert len(result_chunks) == 1
    assert result_chunks[0].tool_name == "peek"
    assert result_chunks[0].ok is True

    # tool-call chunk precedes its result chunk precedes the terminal FinalChunk
    call_idx = chunks.index(call_chunks[0])
    result_idx = chunks.index(result_chunks[0])
    assert call_idx < result_idx < len(chunks) - 1
    assert isinstance(chunks[-1], FinalChunk)


async def test_stream_agentic_streams_text_without_tools() -> None:
    """With no tools, the loop still streams text deltas and a FinalChunk."""
    pytest.importorskip("pydantic_ai")
    from pydantic_ai.models.test import TestModel

    router = _router(TestModel())
    chunks = [chunk async for chunk in router.stream_agentic(prompt="hello")]

    assert any(isinstance(chunk, TextDeltaChunk) for chunk in chunks)
    assert isinstance(chunks[-1], FinalChunk)
