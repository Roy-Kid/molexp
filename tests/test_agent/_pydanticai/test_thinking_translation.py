"""Reasoning-part → ``ThinkingDeltaChunk`` translation.

The router's ``_request_stream_chunk`` is the seam that used to drop a
reasoning model's chain-of-thought on the floor (it handled only ``TextPart`` /
``TextPartDelta``). These pure-sync unit tests exercise that translation
directly, since no offline pydantic-ai model emits a ``ThinkingPart``.
"""

from __future__ import annotations

import pytest

from molexp.agent.router import TextDeltaChunk, ThinkingDeltaChunk


def test_thinking_part_start_translates_to_thinking_chunk() -> None:
    """A ``ThinkingPart`` part-start surfaces as a ``ThinkingDeltaChunk``."""
    pytest.importorskip("pydantic_ai")
    from pydantic_ai.messages import PartStartEvent, ThinkingPart

    from molexp.agent._pydanticai.router import _request_stream_chunk

    chunk = _request_stream_chunk(
        PartStartEvent(index=0, part=ThinkingPart(content="let me reason"))
    )
    assert isinstance(chunk, ThinkingDeltaChunk)
    assert chunk.text == "let me reason"


def test_thinking_part_delta_translates_to_thinking_chunk() -> None:
    """A ``ThinkingPartDelta`` surfaces as a ``ThinkingDeltaChunk``."""
    pytest.importorskip("pydantic_ai")
    from pydantic_ai.messages import PartDeltaEvent, ThinkingPartDelta

    from molexp.agent._pydanticai.router import _request_stream_chunk

    chunk = _request_stream_chunk(
        PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=" further"))
    )
    assert isinstance(chunk, ThinkingDeltaChunk)
    assert chunk.text == " further"


def test_answer_text_still_routes_to_text_chunk() -> None:
    """Reasoning is checked first, but answer text still maps to a text chunk."""
    pytest.importorskip("pydantic_ai")
    from pydantic_ai.messages import PartStartEvent, TextPart

    from molexp.agent._pydanticai.router import _request_stream_chunk

    chunk = _request_stream_chunk(PartStartEvent(index=1, part=TextPart(content="the answer")))
    assert isinstance(chunk, TextDeltaChunk)


def test_empty_thinking_content_yields_no_chunk() -> None:
    """A reasoning event with no content is dropped (no no-op chunk)."""
    pytest.importorskip("pydantic_ai")
    from pydantic_ai.messages import PartStartEvent, ThinkingPart

    from molexp.agent._pydanticai.router import _request_stream_chunk

    assert _request_stream_chunk(PartStartEvent(index=0, part=ThinkingPart(content=""))) is None
