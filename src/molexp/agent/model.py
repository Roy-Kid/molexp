"""ModelClient protocol + request/response types.

The model boundary is narrow: a model plugin accepts a
``ModelRequest`` and returns a ``ModelResponse`` (or streams
``ModelEvent`` items). Tool execution and session orchestration live
outside the plugin.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Literal, Protocol, runtime_checkable

from molexp.agent.types import Message, Usage


@dataclass(frozen=True)
class ToolSchema:
    """Tool description as the model sees it.

    Distinct from ``ToolSpec`` (which carries harness-internal policy
    flags). The plugin marshals ``ToolSchema`` into provider-native
    tool declarations; the harness never edits a plugin's tool format.
    """

    name: str
    description: str
    input_schema: dict[str, Any]


@dataclass(frozen=True)
class ModelBudget:
    """Per-request budget caps applied by the model plugin if it can.

    All fields are advisory. The harness enforces hard limits via the
    constraints layer (§6.6); plugins may surface SDK-side caps when
    cheap to do so.
    """

    max_output_tokens: int | None = None
    max_input_tokens: int | None = None
    timeout_seconds: float | None = None


@dataclass(frozen=True)
class ModelToolCall:
    """A single tool invocation requested by the model.

    ``id`` correlates request/result rounds in providers that require
    it (OpenAI ``tool_call_id``, Anthropic ``tool_use_id``); the plugin
    must preserve that correlation in ``model_io.jsonl``.
    """

    id: str
    name: str
    arguments: dict[str, Any]


@dataclass(frozen=True)
class ModelRequest:
    """Inputs to one model turn."""

    session_id: str
    turn_id: str
    system: str
    messages: tuple[Message, ...]
    tools: tuple[ToolSchema, ...] = ()
    response_format: dict[str, Any] | None = None
    budget: ModelBudget | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ModelResponse:
    """Outputs of one (non-streaming) model turn."""

    text: str = ""
    tool_calls: tuple[ModelToolCall, ...] = ()
    usage: Usage = field(default_factory=Usage)
    finish_reason: str = ""
    raw: Any = None


@dataclass(frozen=True)
class ModelEvent:
    """Streaming model event.

    ``kind`` is intentionally narrow; new providers should map their
    SDK events into one of these tags rather than introduce variants.
    """

    kind: Literal["text-delta", "tool-call", "usage", "finish", "error"]
    text: str = ""
    tool_call: ModelToolCall | None = None
    usage: Usage | None = None
    finish_reason: str = ""
    error: str = ""
    raw: Any = None


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


@dataclass(frozen=True)
class ModelConfig:
    """Generic provider config.

    Carried as core state so the UI and admin routes can render or
    edit settings without a model plugin loaded. Per-provider
    validation is delegated to ``ProviderConfigValidator``.
    """

    provider_name: str
    model: str
    api_key: str | None = None
    base_url: str | None = None
    instructions: str = ""
    extras: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class ProviderConfigValidator(Protocol):
    """Per-provider field rules."""

    provider_name: str

    def validate(self, config: ModelConfig) -> tuple[str, ...]:
        """Return a tuple of human-readable errors; empty on success."""
