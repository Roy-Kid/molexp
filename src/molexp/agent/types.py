"""Core agent types — pydantic-validated, plugin-agnostic.

Defines the harness data contracts every layer (context, tools,
orchestration, state, observability, recovery) and every plugin
(model, tool source) shares.

Aligned with the rest of the project's modeling convention
(``workspace/*``, ``workflow/cache``, ``server/schemas`` are all
pydantic). Import-guard contract (see
``tests/test_agent/test_import_guard.py``): nothing here may import
LLM SDKs (``pydantic_ai``, ``openai``, ``anthropic``,
``google.genai``), MCP, or HTTP clients (``httpx`` / ``aiohttp`` /
``requests``). Pydantic itself is **allowed** — it is a pure
validation library and is already a baseline dependency across
workspace / workflow / server.

Artifact references and asset metadata live in
``molexp.workspace.assets`` (``Asset`` / ``ArtifactAsset``); this
module deliberately does **not** define a parallel artifact type.
"""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class GoalMode(StrEnum):
    """Legacy enum tagging a :class:`Goal`'s intended runtime mode.

    Retained on :class:`Goal` for in-flight session metadata; the runtime
    ``AgentMode`` ABC (in :mod:`molexp.agent.mode`) is the post-refactor
    successor for *strategy* selection at the runner layer.
    """

    CHAT = "chat"
    PLAN = "plan"
    REVIEW = "review"


class SessionStatus(StrEnum):
    """Terminal and live session states.

    On server restart any non-terminal session is flipped to
    ``interrupted``; full rehydration to ``resumable`` is a follow-up.
    """

    PENDING = "pending"
    RUNNING = "running"
    AWAITING_APPROVAL = "awaiting_approval"
    AWAITING_PLAN_DECISION = "awaiting_plan_decision"
    AWAITING_USER = "awaiting_user"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    INTERRUPTED = "interrupted"
    RESUMABLE = "resumable"
    LEGACY = "legacy"


class FailureKind(StrEnum):
    """Typed failure taxonomy."""

    MODEL_ERROR = "model_error"
    TOOL_ERROR = "tool_error"
    TOOL_NOT_FOUND = "tool_not_found"
    POLICY_DENIED = "policy_denied"
    APPROVAL_DENIED = "approval_denied"
    CONTEXT_OVERFLOW = "context_overflow"
    INVALID_PLAN = "invalid_plan"
    USER_CANCELLED = "user_cancelled"
    WORKSPACE_CONFLICT = "workspace_conflict"
    INTERNAL_ERROR = "internal_error"


_FROZEN = ConfigDict(frozen=True)


class Goal(BaseModel):
    """A user-specified objective for one agent session.

    The skill is referenced by id (``skill_id``) so the
    ``ContextManager`` can re-resolve the addendum from
    :class:`SkillStore` on every turn. The full skill body is *never*
    inlined here — the harness keeps semantic state only.
    """

    model_config = _FROZEN

    description: str
    constraints: dict[str, Any] = Field(default_factory=dict)
    success_criteria: list[str] = Field(default_factory=list)
    mode: GoalMode = GoalMode.CHAT
    instructions_override: str | None = None
    skill_id: str | None = None


class Message(BaseModel):
    """Harness-level semantic message.

    The only on-disk message shape the harness persists or reads.
    ``content`` is a flat string sufficient for prompt assembly and
    replay; provider-native shapes live in the parallel
    ``model_io.jsonl`` owned by the model plugin.
    """

    model_config = _FROZEN

    role: Literal["system", "user", "assistant", "tool"]
    content: str
    name: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class Usage(BaseModel):
    """Normalized token / request usage counters."""

    model_config = _FROZEN

    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    total_tokens: int = 0
    requests: int = 0


class AgentFailure(BaseModel):
    """Typed failure record threaded through tool results and events."""

    model_config = _FROZEN

    kind: FailureKind
    message: str
    detail: dict[str, Any] = Field(default_factory=dict)


# Helpers --------------------------------------------------------------


def utc_now() -> datetime:
    """Return a timezone-aware UTC ``datetime``.

    Centralized so events keep a single time source and tests can
    monkeypatch a single import site.
    """

    return datetime.now(UTC)
