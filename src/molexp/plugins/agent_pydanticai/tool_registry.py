"""Native tool registry — single source of truth for builtin agent tools.

Replaces the old ``READ_ONLY_TOOLS``/``WRITE_TOOLS``/``CHAT_TOOLS`` static
lists in ``workspace_tools.py``. Each native tool function self-registers
at import time via :func:`native_tool`, carrying enough metadata for:

- The catalog to filter by :attr:`NativeToolSpec.mutates` (plan-mode
  default-deny) or by a :class:`Skill`'s ``allowed_tools`` /
  ``denied_tools`` glob lists
- The settings UI (:func:`molexp.plugins.agent_pydanticai.admin.describe_native_tools`)
  to render the catalog without hand-maintaining a parallel list
- The system-prompt composer to enumerate available tools deterministically

User-supplied tools (passed via ``AgentService(extra_tools=[...])``) and
MCP-discovered tools are NOT held by this registry — they are session-
scoped and resolved at agent-build time. The registry is for the durable,
package-shipped surface only.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

ToolCategory = Literal["workspace", "workflow", "chat", "control"]


@dataclass(frozen=True)
class NativeToolSpec:
    """Metadata for one builtin agent tool.

    ``fn`` is the raw async function — pydantic-ai introspects it
    directly (signature + annotations + docstring) when registering it
    into a :class:`pydantic_ai.toolsets.FunctionToolset`. The fields
    captured here are the ones the registry consumers care about
    *outside* pydantic-ai's view (filtering, UI listing, approval
    decisions).
    """

    name: str
    fn: Callable
    description: str
    category: ToolCategory
    mutates: bool
    requires_approval: bool


class ToolRegistry:
    """In-memory dictionary keyed by tool name. Append-only by design.

    Re-registering an existing name raises — duplicates almost always
    indicate two ``@native_tool`` decorators on the same function or a
    typo in a tool name, both of which deserve a loud failure rather
    than silent shadowing.
    """

    def __init__(self) -> None:
        self._by_name: dict[str, NativeToolSpec] = {}

    def register(self, spec: NativeToolSpec) -> None:
        if spec.name in self._by_name:
            raise ValueError(
                f"Native tool '{spec.name}' is already registered. "
                "Each @native_tool decorator must use a unique tool name."
            )
        self._by_name[spec.name] = spec

    def get(self, name: str) -> NativeToolSpec | None:
        return self._by_name.get(name)

    def all(self) -> list[NativeToolSpec]:
        """Return every registered tool, sorted by category then name.

        Sort order is stable so the system-prompt's tool-surface section
        and the settings UI list render in the same order across runs.
        Read-only tools come first within a category by convention.
        """
        return sorted(
            self._by_name.values(),
            key=lambda s: (s.category, s.mutates, s.name),
        )

    def filter(
        self,
        *,
        category: ToolCategory | None = None,
        mutates: bool | None = None,
    ) -> list[NativeToolSpec]:
        out = self.all()
        if category is not None:
            out = [s for s in out if s.category == category]
        if mutates is not None:
            out = [s for s in out if s.mutates is mutates]
        return out


_REGISTRY = ToolRegistry()


def builtin_tool_registry() -> ToolRegistry:
    """Return the process-wide :class:`ToolRegistry` of native tools."""
    return _REGISTRY


def native_tool(
    *,
    category: ToolCategory,
    mutates: bool = False,
    requires_approval: bool = False,
    name: str | None = None,
) -> Callable:
    """Decorator: register an async function as a native agent tool.

    Usage::

        @native_tool(category="workflow", mutates=True)
        async def set_workflow_from_ir(ctx, ...): ...

    The decorator returns the original function unchanged so pydantic-ai
    can register it directly via ``FunctionToolset(tools=[fn])``. The
    metadata captured here is consumed by:

    - :class:`MolexpToolCatalog` for plan-mode and skill-scoped filtering
    - :func:`describe_native_tools` for the settings UI
    - :class:`ApprovalPolicy` defaulting (when ``requires_approval=True``)
    """

    def decorator(fn: Callable) -> Callable:
        tool_name = name or fn.__name__
        description = (inspect.getdoc(fn) or "").strip()
        spec = NativeToolSpec(
            name=tool_name,
            fn=fn,
            description=description,
            category=category,
            mutates=mutates,
            requires_approval=requires_approval,
        )
        _REGISTRY.register(spec)
        return fn

    return decorator
