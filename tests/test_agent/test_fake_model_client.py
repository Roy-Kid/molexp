"""Phase 1a: FakeModelClient determinism + io_log shape (spec §14 R2)."""

from __future__ import annotations

import json

import pytest

from molexp.agent import (
    Message,
    ModelClient,
    ModelRequest,
    ModelResponse,
    Usage,
)
from molexp.agent.testing import FakeModelClient, ScriptExhausted


def _request(turn: str = "t1") -> ModelRequest:
    return ModelRequest(
        session_id="s",
        turn_id=turn,
        system="rules",
        messages=(Message(role="user", content="hi"),),
    )


def test_satisfies_modelclient_protocol() -> None:
    client = FakeModelClient()
    assert isinstance(client, ModelClient)


@pytest.mark.asyncio
async def test_complete_pops_scripted_responses_in_order() -> None:
    client = FakeModelClient(
        responses=[
            ModelResponse(text="one", finish_reason="stop"),
            ModelResponse(text="two", finish_reason="stop"),
        ]
    )
    a = await client.complete(_request("t1"))
    b = await client.complete(_request("t2"))
    assert a.text == "one"
    assert b.text == "two"
    assert [c.turn_id for c in client.calls] == ["t1", "t2"]


@pytest.mark.asyncio
async def test_complete_raises_when_script_exhausted() -> None:
    client = FakeModelClient()
    with pytest.raises(ScriptExhausted):
        await client.complete(_request())


@pytest.mark.asyncio
async def test_queue_helpers_build_responses() -> None:
    client = FakeModelClient()
    client.queue_text("hello")
    client.queue_tool_call("native:read", {"path": "foo.txt"})
    first = await client.complete(_request("t1"))
    second = await client.complete(_request("t2"))
    assert first.text == "hello"
    assert len(second.tool_calls) == 1
    assert second.tool_calls[0].name == "native:read"
    assert second.tool_calls[0].arguments == {"path": "foo.txt"}


@pytest.mark.asyncio
async def test_io_log_is_json_serializable() -> None:
    client = FakeModelClient(
        responses=[
            ModelResponse(
                text="bye",
                usage=Usage(input_tokens=1, output_tokens=2, total_tokens=3),
                finish_reason="stop",
            )
        ]
    )
    await client.complete(_request())
    # Decision M1 + §14 R2: model plugins must be able to serialize
    # their io_log into model_io.jsonl. Verify round-trip.
    payload = json.dumps(client.io_log)
    revived = json.loads(payload)
    assert revived[0]["response"]["text"] == "bye"
    assert revived[0]["request"]["turn_id"] == "t1"


@pytest.mark.asyncio
async def test_streaming_replay_is_deterministic() -> None:
    from molexp.agent.model import ModelEvent

    events = [
        ModelEvent(kind="text-delta", text="he"),
        ModelEvent(kind="text-delta", text="llo"),
        ModelEvent(kind="finish", finish_reason="stop"),
    ]
    client = FakeModelClient(streams=[events])
    received = []
    async for ev in client.stream(_request()):
        received.append(ev)
    assert [e.kind for e in received] == ["text-delta", "text-delta", "finish"]
    assert received[0].text == "he"
