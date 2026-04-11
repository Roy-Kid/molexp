"""ApprovalPolicy: controls which tools require human approval."""

from __future__ import annotations

import fnmatch


class ApprovalPolicy:
    """Defines which tool calls require human approval.

    Patterns are glob-style. auto_approve takes precedence.

    Example::

        policy = ApprovalPolicy(
            require_approval_for=["product.*", "workflow.execute"],
            auto_approve=["workspace.read_*", "workflow.inspect"],
        )
    """

    def __init__(
        self,
        require_approval_for: list[str] | None = None,
        auto_approve: list[str] | None = None,
    ) -> None:
        self.require_approval_for: list[str] = require_approval_for or []
        self.auto_approve: list[str] = auto_approve or []

    def needs_approval(self, tool_name: str) -> bool:
        """Check whether a tool invocation requires approval."""
        for pattern in self.auto_approve:
            if fnmatch.fnmatch(tool_name, pattern):
                return False
        for pattern in self.require_approval_for:
            if fnmatch.fnmatch(tool_name, pattern):
                return True
        return False
