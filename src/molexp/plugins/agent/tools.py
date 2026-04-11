"""Tool base class and @agent_tool decorator.

Two equivalent ways to register tools:

Decorator style::

    @agent_tool(level="workspace", requires_approval=False)
    async def read_asset(ctx: ToolContext, asset_name: str) -> AssetContent:
        return ctx.workspace.assets.read(asset_name)

OOP style::

    class ReadAssetTool(Tool):
        name = "workspace.read_asset"
        level = "workspace"
        requires_approval = False

        async def call(self, ctx: ToolContext, asset_name: str) -> AssetContent:
            return ctx.workspace.assets.read(asset_name)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any

from .types import ToolContext


class Tool(ABC):
    """Abstract base class for OOP-style agent tools."""

    name: str = ""
    level: str = "workspace"
    requires_approval: bool = False

    @abstractmethod
    async def call(self, ctx: ToolContext, **kwargs: Any) -> Any:
        """Execute the tool."""
        ...


class FunctionTool(Tool):
    """Adapts a decorated function into a Tool instance."""

    def __init__(
        self,
        fn: Callable,
        name: str,
        level: str,
        requires_approval: bool,
    ) -> None:
        self._fn = fn
        self.name = name
        self.level = level
        self.requires_approval = requires_approval

    async def call(self, ctx: ToolContext, **kwargs: Any) -> Any:
        return await self._fn(ctx, **kwargs)


def agent_tool(
    fn: Callable | None = None,
    *,
    level: str = "workspace",
    requires_approval: bool = False,
    name: str | None = None,
) -> Callable:
    """Decorator to register a function as an agent tool.

    Usage::

        @agent_tool(level="product", requires_approval=True)
        async def run_training(ctx: ToolContext, ...) -> ...: ...
    """
    def decorator(f: Callable) -> Callable:
        tool_name = name or f.__name__
        tool_instance = FunctionTool(
            fn=f,
            name=tool_name,
            level=level,
            requires_approval=requires_approval,
        )
        f._tool_registration = tool_instance
        return f

    if fn is not None:
        return decorator(fn)
    return decorator
