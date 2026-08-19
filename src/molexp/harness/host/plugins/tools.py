"""``ctx.tools`` — reversible tool registrations and ``tools/*`` pipeline."""

from __future__ import annotations

import inspect
from collections.abc import Awaitable, Callable
from typing import Any

from molexp.harness.host.context import Context
from molexp.harness.host.keys import Keys

__all__ = ["ToolBelt", "ToolsPlugin"]


class ToolBelt:
    """Named tool list. ``register`` is an effect — unload drops the tool."""

    def __init__(self) -> None:
        self._tools: list[object] = []
        self._ctx: Context | None = None

    def bind(self, ctx: Context) -> None:
        """Attach the host context used by :meth:`execute`."""
        self._ctx = ctx

    def register(self, tool: object, ctx: Context) -> None:
        """Append *tool*. Unwind removes this exact object."""
        self._tools.append(tool)

        def _drop() -> None:
            try:
                self._tools.remove(tool)
            except ValueError:
                return

        ctx.effect(_drop)

    def snapshot(self) -> tuple[object, ...]:
        """Callables the model sees; named tools go through :meth:`execute`."""
        wrapped: list[object] = []
        for tool in self._tools:
            try:
                wrapped.append(self._wrap(tool))
            except KeyError:
                wrapped.append(tool)
        return tuple(wrapped)

    def _tool_name(self, tool: object) -> str:
        name = getattr(tool, "__name__", None) or getattr(tool, "name", None)
        if not name:
            raise KeyError("tool has no name")
        return str(name)

    def _lookup(self, name: str) -> object:
        for tool in self._tools:
            if self._tool_name(tool) == name:
                return tool
        raise KeyError(f"tool {name!r} is not registered")

    async def execute(self, name: str, args: dict[str, Any]) -> object:
        """Run ``tools/pre-execute`` → body → ``tools/post-execute``."""
        ctx = self._ctx
        if ctx is None:
            raise RuntimeError("ToolBelt.execute requires bind(ctx)")
        payload: dict[str, Any] = {"name": name, "args": dict(args)}

        async def _run(current: object) -> object:
            data: dict[str, Any] = payload
            if isinstance(current, dict):
                data = {str(k): v for k, v in current.items()}
            tool_name = str(data.get("name", name))
            raw_args = data.get("args", args)
            call_args: dict[str, Any] = (
                {str(k): v for k, v in raw_args.items()}
                if isinstance(raw_args, dict)
                else dict(args)
            )
            result = await self._invoke(self._lookup(tool_name), call_args)
            post_payload: dict[str, Any] = {
                "name": tool_name,
                "args": call_args,
                "result": result,
            }
            posted = await ctx.waterfall("tools/post-execute", post_payload)
            if isinstance(posted, dict):
                posted_map = {str(k): v for k, v in posted.items()}
                if "result" in posted_map:
                    return posted_map["result"]
            return result

        return await ctx.waterfall("tools/pre-execute", payload, then=_run)

    async def _invoke(self, tool: object, args: dict[str, Any]) -> object:
        fn: Any = tool if callable(tool) else getattr(tool, "fn", None)
        if not callable(fn):
            raise TypeError(f"tool {self._tool_name(tool)!r} is not callable")
        result = fn(**args)
        if inspect.isawaitable(result) or isinstance(result, Awaitable):
            return await result
        return result

    def _wrap(self, tool: object) -> Callable[..., Any]:
        tool_name = self._tool_name(tool)

        async def wrapped(**kwargs: object) -> object:
            return await self.execute(tool_name, dict(kwargs))

        wrapped.__name__ = tool_name
        wrapped.__doc__ = getattr(tool, "__doc__", None)
        return wrapped


class ToolsPlugin:
    """Publish an empty :class:`ToolBelt` as ``ctx.tools``."""

    name = "tools"
    inject: tuple[str, ...] = ()

    def apply(self, ctx: Context) -> None:
        """Provide :data:`Keys.TOOLS`."""
        belt = ToolBelt()
        belt.bind(ctx)
        ctx.provide(Keys.TOOLS, belt)
