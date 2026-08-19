"""Plugin ``Context`` — service map, reversible effects, typed events."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import Any, Literal

__all__ = ["Context"]

EventMode = Literal["emit", "waterfall", "serial", "parallel"]

# Listeners are plugin-provided callables; the host does not constrain
# their signatures beyond the dispatch mode documented on each event.
Listener = Callable[..., Any]


class Context:
    """Repository of services, effects, and event listeners for one Host.

    ``provide`` / ``on`` / ``effect`` all register a disposer so
    :meth:`unwind_to` can unload a plugin cleanly.
    """

    def __init__(self) -> None:
        self._services: dict[str, object] = {}
        self._effects: list[Callable[[], None]] = []
        self._listeners: dict[str, list[tuple[EventMode, Listener]]] = {}

    def effect_count(self) -> int:
        """Number of registered effects (used as an unload watermark)."""
        return len(self._effects)

    def effect(self, disposer: Callable[[], None]) -> None:
        """Register a disposer that ``unload`` will run (LIFO)."""
        self._effects.append(disposer)

    def provide(self, key: str, value: object) -> None:
        """Publish ``key``. Replacing an existing key is an error.

        Unload deletes the key only if it still points at *value*.
        """
        if key in self._services:
            raise ValueError(f"service {key!r} is already provided")
        self._services[key] = value

        def _drop() -> None:
            if self._services.get(key) is value:
                del self._services[key]

        self.effect(_drop)

    def service_keys(self) -> tuple[str, ...]:
        """Published service keys, sorted."""
        return tuple(sorted(self._services))

    def has(self, key: str) -> bool:
        """True iff *key* is published."""
        return key in self._services

    def get(self, key: str, default: object | None = None) -> object | None:
        """Return the service or *default* (never raises)."""
        return self._services.get(key, default)

    def require(self, key: str) -> object:
        """Return the service or fail loud."""
        if key not in self._services:
            raise KeyError(f"service {key!r} is not on the host")
        return self._services[key]

    def __getattr__(self, key: str) -> object:
        """DeepSeek/Cordis access: ``ctx.tools`` after ``provide("tools", …)``.

        Missing services raise :class:`AttributeError` so ``hasattr`` is false.
        """
        try:
            return self._services[key]
        except KeyError:
            raise AttributeError(f"ctx has no service {key!r}") from None

    def on(
        self,
        event: str,
        listener: Listener,
        *,
        mode: EventMode = "emit",
    ) -> None:
        """Subscribe *listener* to *event* under *mode*."""
        bucket = self._listeners.setdefault(event, [])
        entry = (mode, listener)
        bucket.append(entry)

        def _drop() -> None:
            current = self._listeners.get(event)
            if current is None:
                return
            try:
                current.remove(entry)
            except ValueError:
                return
            if not current:
                del self._listeners[event]

        self.effect(_drop)

    def unwind_to(self, mark: int) -> None:
        """Run disposers newer than *mark*, last-in first."""
        if mark < 0 or mark > len(self._effects):
            raise ValueError(f"effect mark {mark} is out of range")
        while len(self._effects) > mark:
            disposer = self._effects.pop()
            disposer()

    async def emit(self, event: str, *args: object) -> None:
        """Notify emit-mode listeners in registration order (await async)."""
        for mode, listener in list(self._listeners.get(event, ())):
            if mode != "emit":
                continue
            result = listener(*args)
            if asyncio.iscoroutine(result) or isinstance(result, Awaitable):
                await result

    async def waterfall(
        self,
        event: str,
        value: object,
        *,
        then: Callable[..., Any] | None = None,
    ) -> object:
        """Around-middleware. Listeners receive ``(value, next)``.

        Call ``next(value)`` to delegate; return without ``next`` to
        short-circuit. A listener with no ``next`` call that returns a
        value replaces the chain result. ``then`` is the innermost
        continuation (DeepSeek-style pipeline terminal).
        """
        chain = [fn for mode, fn in self._listeners.get(event, ()) if mode == "waterfall"]

        async def invoke(index: int, current: object) -> object:
            if index >= len(chain):
                if then is None:
                    return current
                result = then(current)
                if asyncio.iscoroutine(result) or isinstance(result, Awaitable):
                    return await result
                return result

            async def nxt(updated: object | None = None) -> object:
                return await invoke(index + 1, current if updated is None else updated)

            result = chain[index](current, nxt)
            if asyncio.iscoroutine(result) or isinstance(result, Awaitable):
                result = await result
            return result

        return await invoke(0, value)

    async def serial(self, event: str, *args: object) -> list[object]:
        """Run serial-mode listeners in order; collect return values."""
        out: list[object] = []
        for mode, listener in list(self._listeners.get(event, ())):
            if mode != "serial":
                continue
            result = listener(*args)
            if asyncio.iscoroutine(result) or isinstance(result, Awaitable):
                result = await result
            out.append(result)
        return out

    async def parallel(self, event: str, *args: object) -> list[object]:
        """Run parallel-mode listeners concurrently; collect return values."""

        async def _one(listener: Listener) -> object:
            result = listener(*args)
            if asyncio.iscoroutine(result) or isinstance(result, Awaitable):
                return await result
            return result

        jobs = [
            _one(listener)
            for mode, listener in list(self._listeners.get(event, ()))
            if mode == "parallel"
        ]
        if not jobs:
            return []
        return list(await asyncio.gather(*jobs))
