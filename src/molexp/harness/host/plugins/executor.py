"""``ctx.executor`` — compile/run subprocess seam."""

from __future__ import annotations

from typing import TYPE_CHECKING

from molexp.harness.host.context import Context
from molexp.harness.host.keys import Keys

if TYPE_CHECKING:
    from molexp.harness.executors import Executor

__all__ = ["ExecutorPlugin"]


class ExecutorPlugin:
    """Publish an :class:`~molexp.harness.executors.Executor` as ``ctx.executor``."""

    name = "executor"
    inject: tuple[str, ...] = ()

    def __init__(self, executor: Executor | None = None) -> None:
        self._executor = executor

    def apply(self, ctx: Context) -> None:
        """Provide :data:`Keys.EXECUTOR` (defaults to :class:`LocalExecutor`)."""
        executor = self._executor
        if executor is None:
            from molexp.harness.executors.local import LocalExecutor

            executor = LocalExecutor()
        ctx.provide(Keys.EXECUTOR, executor)
