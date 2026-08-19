"""Plugin protocol: ``name`` + ``inject`` + ``apply(ctx)``."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from molexp.harness.host.context import Context

__all__ = ["Plugin"]


@runtime_checkable
class Plugin(Protocol):
    """A Service mounted on a :class:`~molexp.harness.host.host.Host`.

    ``inject`` lists service keys that must already exist. ``apply``
    publishes services, listeners, and other effects onto *ctx*.
    """

    name: str
    inject: tuple[str, ...]

    def apply(self, ctx: Context) -> None:
        """Publish this plugin's services and effects onto *ctx*."""
        ...
