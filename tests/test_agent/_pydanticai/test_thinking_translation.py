"""Reasoning-part → ``ThinkingDeltaChunk`` translation.

The router's ``_request_stream_chunk`` is the seam that used to drop a
reasoning model's chain-of-thought on the floor (it handled only ``TextPart`` /
``TextPartDelta``). These pure-sync unit tests exercise that translation
directly, since no offline pydantic-ai model emits a ``ThinkingPart``.
"""

from __future__ import annotations

import pytest

from molexp.agent.router import ThinkingDeltaChunk

pytest.importorskip("pydantic_ai")

from pydantic_ai.messages import (  # noqa: E402 — gated on the importorskip above
    PartDeltaEvent,
    PartStartEvent,
    ThinkingPart,
    ThinkingPartDelta,
)

from molexp.agent._pydanticai.router import (  # noqa: E402 — gated on the importorskip above
    _request_stream_chunk,
)


class TestRequestStreamChunk:
    """Reasoning stream events surface as ``ThinkingDeltaChunk`` (regression:
    ``_request_stream_chunk`` once handled only text parts and dropped a
    reasoning model's chain-of-thought)."""

    def test_thinking_part_start_surfaces_as_thinking_chunk(self) -> None:
        chunk = _request_stream_chunk(
            PartStartEvent(index=0, part=ThinkingPart(content="let me reason"))
        )
        assert isinstance(chunk, ThinkingDeltaChunk)
        assert chunk.text == "let me reason"

    def test_thinking_part_delta_surfaces_as_thinking_chunk(self) -> None:
        chunk = _request_stream_chunk(
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=" further"))
        )
        assert isinstance(chunk, ThinkingDeltaChunk)
        assert chunk.text == " further"

    def test_empty_thinking_content_yields_no_chunk(self) -> None:
        """Boundary: a reasoning event with no content is dropped (no no-op chunk)."""
        assert (
            _request_stream_chunk(PartStartEvent(index=0, part=ThinkingPart(content="")))
            is None
        )
