"""ModelClient protocol + request/response types.

The model boundary is narrow: a model plugin accepts a
``ModelRequest`` and returns a ``ModelResponse`` (or streams
``ModelEvent`` items). Tool execution and session orchestration live
outside the plugin.
"""

from __future__ import annotations

from typing import Any, AsyncIterator, Literal, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field

from molexp.agent._serialize import JsonValue
from molexp.agent.types import Message, Usage

_FROZEN = ConfigDict(frozen=True)


class ToolSchema(BaseModel):
    """Tool description as the model sees it.

    Distinct from ``ToolSpec`` (which carries harness-internal policy
    flags). The plugin marshals ``ToolSchema`` into provider-native
    tool declarations; the harness never edits a plugin's tool format.
    """

    model_config = _FROZEN

    name: str
    description: str
    input_schema: dict[str, Any]


class ModelBudget(BaseModel):
    """Per-request budget caps applied by the model plugin if it can.

    All fields are advisory. The harness enforces hard limits via the
    constraints layer (§6.6); plugins may surface SDK-side caps when
    cheap to do so.
    """

    model_config = _FROZEN

    max_output_tokens: int | None = None
    max_input_tokens: int | None = None
    timeout_seconds: float | None = None


class ModelToolCall(BaseModel):
    """A single tool invocation requested by the model.

    ``id`` correlates request/result rounds in providers that require
    it (OpenAI ``tool_call_id``, Anthropic ``tool_use_id``); the plugin
    must preserve that correlation in ``model_io.jsonl``.
    """

    model_config = _FROZEN

    id: str
    name: str
    arguments: dict[str, Any]


class ModelRequest(BaseModel):
    """Inputs to one model turn."""

    model_config = _FROZEN

    session_id: str
    turn_id: str
    system: str
    messages: tuple[Message, ...]
    tools: tuple[ToolSchema, ...] = ()
    response_format: dict[str, Any] | None = None
    budget: ModelBudget | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class ModelResponse(BaseModel):
    """Outputs of one (non-streaming) model turn."""

    model_config = _FROZEN

    text: str = ""
    tool_calls: tuple[ModelToolCall, ...] = ()
    usage: Usage = Field(default_factory=Usage)
    finish_reason: str = ""
    raw: JsonValue | None = None


class ModelEvent(BaseModel):
    """Streaming model event.

    ``kind`` is intentionally narrow; new providers should map their
    SDK events into one of these tags rather than introduce variants.
    """

    model_config = _FROZEN

    kind: Literal["text-delta", "tool-call", "usage", "finish", "error"]
    text: str = ""
    tool_call: ModelToolCall | None = None
    usage: Usage | None = None
    finish_reason: str = ""
    error: str = ""
    raw: JsonValue | None = None


@runtime_checkable
class ModelClient(Protocol):
    """The single contract a model plugin must implement.

    The plugin must not execute tools, must not own session lifecycle,
    and must not parse harness ``messages.jsonl``.
    """

    name: str

    async def complete(self, request: ModelRequest) -> ModelResponse: ...

    def stream(self, request: ModelRequest) -> AsyncIterator[ModelEvent]: ...


@runtime_checkable
class ModelClientFactory(Protocol):
    """Provider-side construction surface."""

    provider_name: str

    def create(self, config: "ModelConfig") -> ModelClient: ...


class ModelConfig(BaseModel):
    """Generic provider config.

    Carried as core state so the UI and admin routes can render or
    edit settings without a model plugin loaded. Per-provider
    validation is delegated to ``ProviderConfigValidator``.
    """

    model_config = _FROZEN

    provider_name: str
    model: str
    api_key: str | None = None
    base_url: str | None = None
    instructions: str = ""
    extras: dict[str, Any] = Field(default_factory=dict)


@runtime_checkable
class ProviderConfigValidator(Protocol):
    """Per-provider field rules."""

    provider_name: str

    def validate(self, config: ModelConfig) -> tuple[str, ...]:
        """Return a tuple of human-readable errors; empty on success."""
