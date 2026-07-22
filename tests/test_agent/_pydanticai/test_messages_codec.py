"""``_pydanticai.messages_codec`` round-trips pydantic-ai ``ModelMessage`` history.

The codec is the sole serialization site for the pydantic-ai-native conversation
context :class:`AgentSession` carries between turns. This pins the wrapper's
own contract — dump returns ``bytes``, load coerces back to a ``tuple`` — using
real SDK message types so any drift in the SDK shape is caught at the boundary.
"""

from __future__ import annotations

import pytest


class TestModelMessagesCodec:
    def test_dump_load_round_trips_messages_and_preserves_tool_identity(self) -> None:
        """dump→load is element-for-element identity; tool-call name + args
        (load-bearing for harvest/export feedstock) survive intact. Uses the
        strongest message graph — text, tool-call, and tool-return parts."""
        pytest.importorskip("pydantic_ai")
        from pydantic_ai.messages import (
            ModelRequest,
            ModelResponse,
            TextPart,
            ToolCallPart,
            ToolReturnPart,
            UserPromptPart,
        )

        from molexp.agent._pydanticai.messages_codec import (
            dump_model_messages,
            load_model_messages,
        )

        original = [
            ModelRequest(parts=[UserPromptPart(content="peek")]),
            ModelResponse(
                parts=[
                    ToolCallPart(tool_name="read_file", args={"path": "a.py"}, tool_call_id="t1"),
                ]
            ),
            ModelRequest(
                parts=[
                    ToolReturnPart(tool_name="read_file", content="print(1)", tool_call_id="t1"),
                ]
            ),
            ModelResponse(parts=[TextPart(content="done")]),
        ]

        data = dump_model_messages(original)
        assert isinstance(data, bytes)
        restored = load_model_messages(data)
        assert isinstance(restored, tuple)
        assert list(restored) == original

        tool_part = restored[1].parts[0]
        assert tool_part.tool_name == "read_file"
        assert tool_part.args == {"path": "a.py"}
