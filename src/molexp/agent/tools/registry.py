"""Tool registry — explicit registration, no module-level singletons.

Each :class:`AgentService` instance owns its own registry;
``@native_tool`` only tags a callable as a candidate so tests and
concurrent services do not contaminate each other.
"""

from __future__ import annotations

import importlib
import pkgutil
from collections.abc import Iterator

from molexp.agent.tools.policy import ToolPolicy
from molexp.agent.tools.spec import (
    RegisteredTool,
    ToolCallable,
    ToolSchema,
    ToolSpec,
)

_NATIVE_TOOL_ATTR = "__molexp_native_tool_spec__"


def native_tool(spec: ToolSpec):
    """Tag a callable as a native-tool candidate.

    The decorator does **not** auto-register on a module-level global.
    :class:`AgentService` walks the ``molexp.agent.tools.native``
    package and registers every tagged callable on its own registry.

    Usage::

        @native_tool(ToolSpec(name="native:read_file", ...))
        async def read_file(args, ctx):
            ...
    """

    def decorator(fn: ToolCallable) -> ToolCallable:
        setattr(fn, _NATIVE_TOOL_ATTR, spec)
        return fn

    return decorator


def is_native_tool(obj: object) -> bool:
    """Return True if ``obj`` was tagged by :func:`native_tool`."""

    return callable(obj) and hasattr(obj, _NATIVE_TOOL_ATTR)


def get_native_spec(fn: ToolCallable) -> ToolSpec:
    """Return the :class:`ToolSpec` attached by :func:`native_tool`."""

    return getattr(fn, _NATIVE_TOOL_ATTR)


def iter_native_tools() -> Iterator[tuple[ToolSpec, ToolCallable]]:
    """Yield ``(spec, fn)`` for every ``@native_tool`` in :mod:`molexp.agent.tools.native`.

    Single source of truth for walking the native package — used by
    :class:`AgentService` (per-instance registration), the read-only
    admin introspection helper, and the ToolStore registrations bridge.
    """

    from molexp.agent.tools import native as native_pkg

    for module_info in pkgutil.iter_modules(native_pkg.__path__, prefix=f"{native_pkg.__name__}."):
        module = importlib.import_module(module_info.name)
        for attr_name in dir(module):
            obj = getattr(module, attr_name)
            if is_native_tool(obj):
                yield get_native_spec(obj), obj


class DuplicateToolError(ValueError):
    """Raised when registering two tools under the same canonical name."""


class ToolRegistry:
    """Per-service tool registry."""

    def __init__(self) -> None:
        self._tools: dict[str, RegisteredTool] = {}

    def register(self, spec: ToolSpec, fn: ToolCallable) -> None:
        """Register ``fn`` under the canonical name ``spec.name``."""

        if spec.name in self._tools:
            raise DuplicateToolError(f"Tool '{spec.name}' is already registered")
        self._tools[spec.name] = RegisteredTool(spec=spec, fn=fn)

    def unregister(self, name: str) -> None:
        """Remove a previously registered tool. Idempotent."""

        self._tools.pop(name, None)

    def get(self, name: str) -> RegisteredTool | None:
        """Return the registered tool with ``name`` or None."""

        return self._tools.get(name)

    def list(self, policy: ToolPolicy | None = None) -> tuple[ToolSpec, ...]:
        """Return tool specs visible under ``policy`` (sorted by name)."""

        specs = sorted(
            (t.spec for t in self._tools.values()),
            key=lambda s: s.name,
        )
        if policy is None:
            return tuple(specs)
        return tuple(s for s in specs if policy.visible(s))

    def schemas(self, policy: ToolPolicy | None = None) -> tuple[ToolSchema, ...]:
        """Return model-facing schemas for tools visible under ``policy``."""

        return tuple(
            ToolSchema(
                name=spec.name,
                description=spec.description,
                input_schema=spec.input_schema,
            )
            for spec in self.list(policy)
        )

    def __iter__(self) -> Iterator[RegisteredTool]:
        return iter(self._tools.values())

    def __len__(self) -> int:
        return len(self._tools)

    def __contains__(self, name: object) -> bool:
        return isinstance(name, str) and name in self._tools
