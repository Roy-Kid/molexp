"""Skill: a reusable behaviour bundle.

A *skill* combines four things into one named, slash-invokable bundle:

1. A parameterized **goal template** with ``{{name}}`` placeholders
2. **System-prompt addendum** (``instructions``) applied for the session
3. **Tool scoping** via ``allowed_tools`` / ``denied_tools`` glob lists
4. A **session mode** opt-in (``default_plan_mode``, ``requires_exit_tool``)

Skills are *not* tools: they produce :class:`~molexp.agent.types.Goal`
payloads consumed by the agent service; the agent then uses its tool
surface filtered by the skill's allow-list to fulfill the goal.

Storage is the standard three-tier
:class:`~molexp.agent.persistence.TieredResourceStore` shape — native
(code-shipped), user-home (``~/.molexp/skills.json``), workspace
(``<root>/.skills.json``). See :mod:`molexp.agent.skills.store`.
"""

from __future__ import annotations

import fnmatch
import re
from typing import Any

from pydantic import Field

from molexp.agent.persistence import ResourceSpec

_PARAM_RE = re.compile(r"\{\{\s*([A-Za-z_][A-Za-z0-9_]*)\s*\}\}")
SLASH_NAME_RE = re.compile(r"^[a-z0-9][a-z0-9-]{0,31}$")

RESERVED_SLASH_NAMES: frozenset[str] = frozenset({"plan", "clear", "model", "help"})
"""Slash names handled by the chat input — skills cannot reuse them."""


class Skill(ResourceSpec):
    """A saved behaviour bundle (see module docstring)."""

    goal_template: str = ""
    slash_name: str = ""
    instructions: str = ""
    default_plan_mode: bool = False
    constraints: list[str] = Field(default_factory=list)
    success_criteria: list[str] = Field(default_factory=list)
    allowed_tools: list[str] = Field(default_factory=list)
    denied_tools: list[str] = Field(default_factory=list)
    requires_exit_tool: str = ""

    def required_parameters(self) -> list[str]:
        """Return the ordered list of distinct ``{{param}}`` placeholders."""

        seen: list[str] = []
        for match in _PARAM_RE.finditer(self.goal_template):
            key = match.group(1)
            if key not in seen:
                seen.append(key)
        return seen

    def materialize(self, params: dict[str, Any] | None = None) -> dict[str, Any]:
        """Render the template into a :class:`Goal`-compatible dict."""

        params = params or {}

        def _sub(match: re.Match[str]) -> str:
            key = match.group(1)
            if key not in params:
                raise KeyError(f"Missing parameter '{key}' for skill '{self.id}'")
            return str(params[key])

        description = _PARAM_RE.sub(_sub, self.goal_template)
        return {
            "description": description,
            "constraints": list(self.constraints),
            "success_criteria": list(self.success_criteria),
        }

    def matches_tool(self, tool_name: str) -> bool:
        """Glob-based allow/deny check (denial wins, empty allow ⇒ all allowed)."""

        for pattern in self.denied_tools:
            if fnmatch.fnmatchcase(tool_name, pattern):
                return False
        if not self.allowed_tools:
            return True
        return any(fnmatch.fnmatchcase(tool_name, pattern) for pattern in self.allowed_tools)


__all__ = [
    "RESERVED_SLASH_NAMES",
    "SLASH_NAME_RE",
    "Skill",
]
