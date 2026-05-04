"""Core agent types — stdlib-only, plugin-agnostic.

Per spec §3 and §5, this module defines the harness data contracts that
every layer (context, tools, orchestration, state, observability,
recovery) and every plugin (model, tool source) shares. Nothing here may
import third-party SDKs, HTTP clients, or optional dependencies.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Literal


class AgentMode(str, Enum):
    """High-level mode the harness operates in for one session.

    ``REVIEW`` is reserved per spec §5.1 and has no semantics yet.
    """

    CHAT = "chat"
    PLAN = "plan"
    REVIEW = "review"


class SessionStatus(str, Enum):
    """Terminal and live session states.

    ``interrupted`` and ``resumable`` are introduced for Decision O3
    (cross-process resume). Phases 1-4 only emit ``interrupted`` on
    server restart; Phase 5+ promotes to ``resumable``.
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


class FailureKind(str, Enum):
    """Typed failure taxonomy per spec §6.6."""

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


@dataclass(frozen=True)
class Goal:
    """A user-specified objective for one agent session.

    Per Decision M1 the harness keeps only *semantic* state; per spec
    §5.1 the skill is referenced by id (``skill_id``) so that the
    ``ContextManager`` can re-resolve the addendum from ``SkillStore``
    on every turn. The full skill body is **not** inlined.
    """

    description: str
    constraints: dict[str, Any] = field(default_factory=dict)
    success_criteria: list[str] = field(default_factory=list)
    mode: AgentMode = AgentMode.CHAT
    instructions_override: str | None = None
    skill_id: str | None = None


@dataclass(frozen=True)
class Message:
    """Harness-level semantic message.

    Per Decision M1 this is the only on-disk message shape the harness
    persists or reads. ``content`` is a flat string sufficient for
    prompt assembly and replay; provider-native shapes live in the
    parallel ``model_io.jsonl`` owned by the model plugin.
    """

    role: Literal["system", "user", "assistant", "tool"]
    content: str
    name: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Usage:
    """Normalized token / request usage counters per spec §6.5."""

    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    total_tokens: int = 0
    requests: int = 0


@dataclass(frozen=True)
class ArtifactRef:
    """Reference to an artifact captured during a tool call.

    Inline artifacts (small JSON payloads such as a Plotly chart spec)
    live in ``payload``; large artifacts are stored under
    ``.molexp-agent/sessions/<id>/artifacts/`` and reach the harness
    via ``path`` only.
    """

    kind: Literal["plot", "table", "text", "file"]
    title: str = ""
    payload: dict[str, Any] = field(default_factory=dict)
    path: str | None = None


@dataclass(frozen=True)
class WorkflowPreview:
    """Structured preview of the workflow a plan would bind.

    Survives unchanged from the previous event surface (see spec §6.5
    migration table). Every plan corresponds to a workflow IR; the
    Mermaid + Python script + intervention points are derived views.
    """

    workflow_ir: dict[str, Any]
    python_script: str = ""
    mermaid: str = ""
    intervention_points: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class AgentFailure:
    """Typed failure record threaded through tool results and events."""

    kind: FailureKind
    message: str
    detail: dict[str, Any] = field(default_factory=dict)


# Helpers --------------------------------------------------------------

def utc_now() -> datetime:
    """Return a timezone-aware UTC ``datetime``.

    Centralized so events keep a single time source and tests can
    monkeypatch a single import site.
    """

    return datetime.now(timezone.utc)
