"""PydanticAI-backed :class:`molexp.agent.ModelClient` implementation.

Uses the SDK's *low-level* :meth:`pydantic_ai.models.Model.request`
method directly, so the harness keeps full ownership of the tool loop
and session lifecycle (per §7.1: "the plugin must not execute tools").
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any, AsyncIterator

from molexp.agent.model import (
    ModelClient,
    ModelEvent,
    ModelRequest,
    ModelResponse,
    ModelToolCall,
)
from molexp.agent.types import Message, Usage

if TYPE_CHECKING:
    from pydantic_ai.messages import ModelMessage as PaiModelMessage
    from pydantic_ai.models import Model


class PydanticAIModelClient:
    """Adapt a pydantic-ai :class:`Model` to the harness contract."""

    def __init__(self, model: "Model", model_name: str) -> None:
        self._model = model
        self.name = model_name

    async def complete(self, request: ModelRequest) -> ModelResponse:
        from pydantic_ai.models import ModelRequestParameters
        from pydantic_ai.tools import ToolDefinition

        messages = _harness_to_pai(request.system, request.messages)
        params = ModelRequestParameters(
            function_tools=[
                ToolDefinition(
                    name=tool.name,
                    description=tool.description,
                    parameters_json_schema=tool.input_schema,
                )
                for tool in request.tools
            ],
            output_mode="text",
            allow_text_output=True,
        )
        settings = _build_settings(request)
        raw = await self._model.request(messages, settings, params)
        return _pai_to_harness(raw)

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelEvent]:
        # Streaming is not yet wired through the harness runner; complete()
        # is the only path Phase 3 needs. Surface a clear error rather than
        # silently degrading so callers know to use complete().
        raise NotImplementedError(
            "PydanticAIModelClient.stream is not implemented yet"
        )
        if False:  # pragma: no cover — keeps mypy happy on the AsyncIterator return
            yield  # type: ignore[unreachable]


def _build_settings(request: ModelRequest) -> dict[str, Any] | None:
    if request.budget is None:
        return None
    settings: dict[str, Any] = {}
    if request.budget.max_output_tokens is not None:
        settings["max_tokens"] = request.budget.max_output_tokens
    if request.budget.timeout_seconds is not None:
        settings["timeout"] = request.budget.timeout_seconds
    return settings or None


def _harness_to_pai(system: str, messages: tuple[Message, ...]) -> list["PaiModelMessage"]:
    """Translate ``(system, history)`` into a pydantic-ai message list.

    Groups consecutive tool/user parts into one ``ModelRequest`` and
    consecutive assistant parts into one ``ModelResponse``, matching
    the alternation providers like Anthropic require.
    """

    from pydantic_ai.messages import (
        ModelRequest as PaiModelRequest,
        ModelResponse as PaiModelResponse,
        SystemPromptPart,
        TextPart,
        ToolCallPart,
        ToolReturnPart,
        UserPromptPart,
    )

    out: list[PaiModelMessage] = []
    pending_request_parts: list[Any] = []
    pending_response_parts: list[Any] = []

    if system:
        pending_request_parts.append(SystemPromptPart(content=system))

    def flush_request() -> None:
        nonlocal pending_request_parts
        if pending_request_parts:
            out.append(PaiModelRequest(parts=pending_request_parts))
            pending_request_parts = []

    def flush_response() -> None:
        nonlocal pending_response_parts
        if pending_response_parts:
            out.append(PaiModelResponse(parts=pending_response_parts))
            pending_response_parts = []

    for msg in messages:
        if msg.role == "user":
            flush_response()
            pending_request_parts.append(UserPromptPart(content=msg.content))
        elif msg.role == "tool":
            flush_response()
            pending_request_parts.append(
                ToolReturnPart(
                    tool_name=msg.name or "",
                    content=msg.content,
                    tool_call_id=str(msg.metadata.get("call_id", "")),
                )
            )
        elif msg.role == "assistant":
            flush_request()
            tool_calls = msg.metadata.get("tool_calls") or ()
            for raw_call in tool_calls:
                pending_response_parts.append(
                    ToolCallPart(
                        tool_name=raw_call["name"],
                        args=raw_call.get("arguments") or {},
                        tool_call_id=raw_call.get("id") or "",
                    )
                )
            if msg.content:
                pending_response_parts.append(TextPart(content=msg.content))
        else:
            # role == "system" never lands in history (it sits in
            # ``request.system``); skip defensively rather than fail-fast
            # to keep the message list robust against future role additions.
            continue

    flush_request()
    flush_response()
    return out


def _pai_to_harness(response: Any) -> ModelResponse:
    """Translate a pydantic-ai :class:`ModelResponse` back into ours."""

    from pydantic_ai.messages import TextPart, ToolCallPart

    text_chunks: list[str] = []
    tool_calls: list[ModelToolCall] = []
    for part in response.parts:
        if isinstance(part, TextPart):
            if part.content:
                text_chunks.append(part.content)
        elif isinstance(part, ToolCallPart):
            tool_calls.append(
                ModelToolCall(
                    id=part.tool_call_id or "",
                    name=part.tool_name,
                    arguments=_coerce_args(part.args),
                )
            )

    usage = Usage()
    raw_usage = getattr(response, "usage", None)
    if raw_usage is not None:
        usage = Usage(
            input_tokens=getattr(raw_usage, "input_tokens", 0) or 0,
            output_tokens=getattr(raw_usage, "output_tokens", 0) or 0,
            total_tokens=getattr(raw_usage, "total_tokens", 0) or 0,
        )

    finish_reason = getattr(response, "finish_reason", "") or ""
    return ModelResponse(
        text="".join(text_chunks),
        tool_calls=tuple(tool_calls),
        usage=usage,
        finish_reason=str(finish_reason),
        raw=response,
    )


def _coerce_args(args: Any) -> dict[str, Any]:
    if isinstance(args, dict):
        return args
    if isinstance(args, str):
        if not args:
            return {}
        parsed = json.loads(args)
        return parsed if isinstance(parsed, dict) else {"_": parsed}
    if args is None:
        return {}
    return {"_": args}


__all__ = ["PydanticAIModelClient"]
