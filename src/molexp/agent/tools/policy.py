"""Approval policy and skill allow/deny gating

Pure data + pure functions: no I/O, no logging, no model imports.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from fnmatch import fnmatchcase

from molexp.agent.tools.spec import ToolSpec


@dataclass(frozen=True)
class ToolPolicy:
    """Filter applied when listing tools or before dispatch.

    ``allow`` and ``deny`` are tuples of fnmatch-compatible patterns
    matched against the canonical tool name (including source prefix).
    ``deny`` wins on conflict. ``approval_overrides`` lets a session
    or skill mark tools that would otherwise auto-approve as gated, or
    vice versa.
    """

    allow: tuple[str, ...] = ()
    deny: tuple[str, ...] = ()
    approval_overrides: dict[str, bool] = field(default_factory=dict)

    def visible(self, spec: ToolSpec) -> bool:
        """Return True if ``spec`` should be exposed under this policy."""

        if any(fnmatchcase(spec.name, p) for p in self.deny):
            return False
        if not self.allow:
            return True
        return any(fnmatchcase(spec.name, p) for p in self.allow)

    def needs_approval(self, spec: ToolSpec) -> bool:
        """Return True if a call to ``spec`` must be human-approved."""

        if spec.name in self.approval_overrides:
            return self.approval_overrides[spec.name]
        return spec.requires_approval or spec.mutates


@dataclass(frozen=True)
class ApprovalDecision:
    """Result of a human-in-the-loop approval prompt."""

    request_id: str
    approved: bool
    reason: str = ""


PERMISSIVE_POLICY = ToolPolicy()
"""Default policy: every visible tool, mutating tools require approval."""


READ_ONLY_POLICY = ToolPolicy(
    deny=("*:write_*", "*:delete_*", "*:run_*", "*:execute_*"),
)
"""Policy used during plan mode: mutators are hidden."""
