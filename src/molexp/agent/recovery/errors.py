"""Typed failure exceptions (spec §6.6).

The taxonomy lives on :class:`molexp.agent.types.FailureKind`. Each
exception class wraps one tag so callers can ``except`` selectively
without string-matching.
"""

from __future__ import annotations

from molexp.agent.types import AgentFailure, FailureKind


class AgentError(Exception):
    """Base class for typed harness failures."""

    kind: FailureKind = FailureKind.INTERNAL_ERROR

    def __init__(self, message: str, detail: dict | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.detail = detail or {}

    def to_failure(self) -> AgentFailure:
        return AgentFailure(kind=self.kind, message=self.message, detail=self.detail)


class ModelFailure(AgentError):
    kind = FailureKind.MODEL_ERROR


class ToolFailure(AgentError):
    kind = FailureKind.TOOL_ERROR


class ToolNotFound(AgentError):
    kind = FailureKind.TOOL_NOT_FOUND


class PolicyDenied(AgentError):
    kind = FailureKind.POLICY_DENIED


class ApprovalDenied(AgentError):
    kind = FailureKind.APPROVAL_DENIED


class ContextOverflow(AgentError):
    kind = FailureKind.CONTEXT_OVERFLOW


class InvalidPlan(AgentError):
    kind = FailureKind.INVALID_PLAN


class UserCancelled(AgentError):
    kind = FailureKind.USER_CANCELLED


class WorkspaceConflict(AgentError):
    kind = FailureKind.WORKSPACE_CONFLICT
