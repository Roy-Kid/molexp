"""Contract tests for the PydanticAI model plugin.

Covers ``PydanticAIModelClient.complete`` end-to-end against an
in-process stub of pydantic-ai's :class:`Model` so the test never
makes a network call.
"""

from __future__ import annotations

from typing import Any

import pytest
from pydantic_ai.messages import (
    ModelRequest as PaiModelRequest,
)
from pydantic_ai.messages import (
    ModelResponse as PaiModelResponse,
)
from pydantic_ai.messages import (
    SystemPromptPart,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.usage import RequestUsage

from molexp.agent import (
    ModelConfig,
    ModelRequest,
    ToolSchema,
    create_model_client,
    list_providers,
)
from molexp.agent.types import Message
from molexp.plugins.model_pydanticai import (  # noqa: F401 — registers the plugin
    PydanticAIModelClient,
    PydanticAIModelClientFactory,
    PydanticAIProviderValidator,
)


class _StubModel:
    """Minimal pydantic-ai :class:`Model` stand-in."""

    def __init__(self, response: PaiModelResponse) -> None:
        self._response = response
        self.last_messages: list[Any] = []
        self.last_params: Any = None
        self.model_name = "stub"

    async def request(self, messages, settings, params):
        self.last_messages = messages
        self.last_params = params
        return self._response


def _harness_client(response: PaiModelResponse) -> tuple[PydanticAIModelClient, _StubModel]:
    stub = _StubModel(response)
    return PydanticAIModelClient(stub, model_name="stub"), stub


def _request(messages: tuple[Message, ...], tools=()) -> ModelRequest:
    return ModelRequest(
        session_id="s",
        turn_id="t1",
        system="you are agent",
        messages=messages,
        tools=tools,
    )


@pytest.mark.asyncio
async def test_complete_text_only_round_trip() -> None:
    response = PaiModelResponse(
        parts=[TextPart(content="hello")],
        usage=RequestUsage(input_tokens=4, output_tokens=2),
        model_name="stub",
        finish_reason="end_turn",
    )
    client, _ = _harness_client(response)
    out = await client.complete(_request((Message(role="user", content="hi"),)))
    assert out.text == "hello"
    assert out.tool_calls == ()
    assert out.usage.input_tokens == 4
    assert out.usage.output_tokens == 2
    assert out.usage.total_tokens == 6
    assert out.finish_reason == "end_turn"


@pytest.mark.asyncio
async def test_complete_emits_tool_calls() -> None:
    response = PaiModelResponse(
        parts=[
            TextPart(content="ok"),
            ToolCallPart(tool_name="native:list_projects", args={}, tool_call_id="c1"),
        ],
        usage=RequestUsage(input_tokens=4, output_tokens=2),
        model_name="stub",
        finish_reason="tool_use",
    )
    client, _ = _harness_client(response)
    out = await client.complete(
        _request(
            (Message(role="user", content="hi"),),
            tools=(
                ToolSchema(
                    name="native:list_projects", description="list", input_schema={"type": "object"}
                ),
            ),
        )
    )
    assert out.text == "ok"
    assert len(out.tool_calls) == 1
    assert out.tool_calls[0].id == "c1"
    assert out.tool_calls[0].name == "native:list_projects"


@pytest.mark.asyncio
async def test_request_translation_groups_alternation() -> None:
    response = PaiModelResponse(parts=[TextPart(content="ok")], model_name="stub")
    client, stub = _harness_client(response)
    history = (
        Message(role="user", content="step 1"),
        Message(role="assistant", content="thinking"),
        Message(role="tool", name="native:list_projects", content="[]", metadata={"call_id": "c1"}),
        Message(role="user", content="follow up"),
    )
    await client.complete(_request(history))

    msgs = stub.last_messages
    # First request packs system prompt + initial user turn.
    assert isinstance(msgs[0], PaiModelRequest)
    parts0 = msgs[0].parts
    assert isinstance(parts0[0], SystemPromptPart)
    assert isinstance(parts0[1], UserPromptPart)
    # Second is the assistant text.
    assert isinstance(msgs[1], PaiModelResponse)
    assert isinstance(msgs[1].parts[0], TextPart)
    # Third coalesces the tool return + the next user message.
    assert isinstance(msgs[2], PaiModelRequest)
    assert isinstance(msgs[2].parts[0], ToolReturnPart)
    assert msgs[2].parts[0].tool_call_id == "c1"
    assert isinstance(msgs[2].parts[1], UserPromptPart)


def test_validator_flags_provider_mismatch() -> None:
    validator = PydanticAIProviderValidator("anthropic")
    errors = validator.validate(ModelConfig(provider_name="openai", model="m", api_key="k"))
    assert errors and "validator bound to" in errors[0]


def test_validator_requires_base_url_for_openai_compatible() -> None:
    validator = PydanticAIProviderValidator("openai-compatible")
    errors = validator.validate(
        ModelConfig(provider_name="openai-compatible", model="m", api_key="k")
    )
    assert errors == ("base_url is required for openai-compatible",)


def test_registry_exposes_each_supported_provider() -> None:
    providers = set(list_providers())
    for name in ("anthropic", "openai", "openai-compatible", "deepseek", "google"):
        assert name in providers, f"missing provider {name}"


def test_factory_builds_anthropic_model() -> None:
    config = ModelConfig(
        provider_name="anthropic",
        model="claude-sonnet-4-6",
        api_key="dummy",
    )
    client = create_model_client(config)
    assert isinstance(client, PydanticAIModelClient)
    assert client.name == "claude-sonnet-4-6"


@pytest.mark.asyncio
async def test_model_io_sink_records_request_response_pair() -> None:
    """Per Decision M1 the plugin owns ``model_io.jsonl`` writes."""

    response = PaiModelResponse(
        parts=[
            TextPart(content="hi"),
            ToolCallPart(tool_name="t", args={"x": 1}, tool_call_id="c1"),
        ],
        usage=RequestUsage(input_tokens=4, output_tokens=2),
        model_name="stub",
        finish_reason="end_turn",
    )
    captured: list[tuple[str, dict]] = []

    def sink(session_id: str, payload: dict) -> None:
        captured.append((session_id, payload))

    stub = _StubModel(response)
    client = PydanticAIModelClient(stub, model_name="stub", model_io_sink=sink)
    await client.complete(_request((Message(role="user", content="go"),)))

    assert len(captured) == 1
    sid, record = captured[0]
    assert sid == "s"
    assert record["session_id"] == "s"
    assert record["turn_id"] == "t1"
    assert record["provider"] == "pydantic-ai"
    assert record["request"]["system"] == "you are agent"
    assert record["request"]["messages"][0]["role"] == "user"
    assert record["response"]["text"] == "hi"
    assert record["response"]["tool_calls"][0]["name"] == "t"
    assert record["response"]["finish_reason"] == "end_turn"
