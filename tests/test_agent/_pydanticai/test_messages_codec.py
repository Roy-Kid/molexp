"""``_pydanticai.messages_codec`` round-trips pydantic-ai ``ModelMessage`` history.

The codec is the sole serialization site for the pydantic-ai-native
conversation context that :class:`AgentSession` carries between turns.
This test pins the round-trip contract using real pydantic-ai message
types so any drift in the SDK shape gets caught at the boundary, not
deep inside the session catalog.
"""

from __future__ import annotations

import pytest


def test_codec_round_trips_pydantic_ai_messages() -> None:
    """Dump → load preserves the message list element-for-element."""
    pytest.importorskip("pydantic_ai")
    from pydantic_ai.messages import (
        ModelRequest,
        ModelResponse,
        TextPart,
        UserPromptPart,
    )

    from molexp.agent._pydanticai.messages_codec import (
        dump_model_messages,
        load_model_messages,
    )

    original = [
        ModelRequest(parts=[UserPromptPart(content="hello")]),
        ModelResponse(parts=[TextPart(content="hi back")]),
    ]
    data = dump_model_messages(original)
    assert isinstance(data, bytes)
    restored = load_model_messages(data)
    assert isinstance(restored, tuple)
    assert len(restored) == 2
    # pydantic-ai messages are pydantic models — equality compares fields.
    assert list(restored) == original


def test_codec_round_trips_empty_list() -> None:
    """Empty history dumps to ``b'[]'`` and reloads as ``()``."""
    pytest.importorskip("pydantic_ai")
    from molexp.agent._pydanticai.messages_codec import (
        dump_model_messages,
        load_model_messages,
    )

    data = dump_model_messages([])
    assert load_model_messages(data) == ()


def test_codec_rejects_malformed_bytes() -> None:
    """Garbage in → ``ValidationError`` out (caller's job to handle)."""
    pytest.importorskip("pydantic_ai")
    from pydantic import ValidationError

    from molexp.agent._pydanticai.messages_codec import load_model_messages

    with pytest.raises(ValidationError):
        load_model_messages(b'{"not": "a-message-list"}')
