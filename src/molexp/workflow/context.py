"""Run context propagation utilities."""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar, Token
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Iterator

if TYPE_CHECKING:
    from ..assets import AssetRepo
    from ..models import Run
    from ..workspace import Workspace


@dataclass(slots=True)
class RunContext:
    """Per-run context shared implicitly by all tasks.

    Attributes:
        asset_repo: Asset repository for storing/retrieving assets
        run_id: Unique run identifier
        run_metadata: Run metadata object
        workspace: Workspace instance
        extras: Additional context data
    """

    asset_repo: AssetRepo
    run_id: str | None = None
    run_metadata: Run | None = None
    workspace: Workspace | None = None
    extras: dict[str, Any] = field(default_factory=dict)


_current_ctx: ContextVar[RunContext | None] = ContextVar(
    "taskflow_run_context", default=None
)


def get_current_context() -> RunContext | None:
    """Return the active :class:`RunContext` if one is set."""

    return _current_ctx.get()


def require_current_context() -> RunContext:
    """Return the current :class:`RunContext` or raise if absent."""

    ctx = get_current_context()
    if ctx is None:  # pragma: no cover - simple branch
        raise RuntimeError("No active RunContext. Did you call TaskEngine.run(...)?")
    return ctx


@contextmanager
def use_run_context(ctx: RunContext) -> Iterator[RunContext]:
    """Context manager that temporarily sets ``ctx`` as current."""

    token: Token[RunContext | None] = _current_ctx.set(ctx)
    try:
        yield ctx
    finally:
        _current_ctx.reset(token)
