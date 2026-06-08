"""Promote a bare callable into a single-Task ``Workflow``.

This used to live in ``molexp.workspace.experiment`` as the private
``_promote_to_workflow`` + ``_resolve_*_entrypoint`` helpers — which
forced workspace to import workflow types. The rectification spec
(2026-05-09) moved it into the workflow layer, where it conceptually
belongs (the output is a ``Workflow``).

Public surface:

- :func:`promote_callable` — wrap a ``fn(inputs, config)`` into a
  single-Task ``Workflow``.
- :func:`resolve_callable_entrypoint` — return ``"<file>:<qualname>"``
  for a module-level callable so the cluster worker can re-import it.
- :func:`resolve_spec_entrypoint` — return ``"<file>:<varname>"`` for
  a ``Workflow`` whose first task lives in a user module, so the
  worker can re-import the spec by name.
"""

from __future__ import annotations

import asyncio
import inspect
from collections.abc import Callable
from pathlib import Path

from .compiled import CompiledWorkflow
from .compiler import WorkflowCompiler
from .context import TaskContext
from .task import Task


class _EntryTask(Task):
    """Wraps a bare ``fn(inputs, config) -> object`` into a workflow Task.

    Module-private; users get to it through :func:`promote_callable`. The
    pure-task-context contract delivers data via ``ctx.inputs`` / ``ctx.config``
    (no ``run_context``); the promoted callable receives those two arguments.
    """

    def __init__(self, fn: Callable) -> None:
        self._fn = fn

    async def execute(self, ctx: TaskContext) -> object:
        if asyncio.iscoroutinefunction(self._fn):
            return await self._fn(ctx.inputs, ctx.config)
        # Run sync bodies in a worker thread so blocking I/O (e.g.
        # ``time.sleep``) does not stall sibling replicas in the
        # same event loop. Preserve the original semantics where a
        # sync callable that returns an awaitable is still awaited.
        result = await asyncio.to_thread(self._fn, ctx.inputs, ctx.config)
        if asyncio.iscoroutine(result) or inspect.isawaitable(result):
            return await result
        return result


def promote_callable(fn: Callable, name: str) -> CompiledWorkflow:
    """Promote a bare ``fn(inputs, config)`` to a single-Task ``CompiledWorkflow``.

    Args:
        fn: Callable that accepts ``(inputs, config)`` (sync or async), matching
            the pure-task-context contract.
        name: Workflow name; used for the artifact's ``name`` field. The
            single task inside it gets the callable's ``__name__``.

    Returns:
        A :class:`CompiledWorkflow` with one task wrapping *fn*.
    """
    fn_name = getattr(fn, "__name__", None) or "anonymous"
    return WorkflowCompiler(name=name).add(_EntryTask(fn), name=fn_name).compile()


def resolve_callable_entrypoint(fn: Callable) -> str:
    """Return ``"<file>:<qualname>"`` for a module-level callable.

    Raises:
        ValueError: If *fn* is nested, a lambda, or otherwise lacks an
            importable qualified name.
    """
    qualname = getattr(fn, "__qualname__", None) or getattr(fn, "__name__", None)
    if qualname is None or "<locals>" in qualname or "<lambda>" in qualname:
        raise ValueError(
            f"cannot determine an importable entrypoint for {fn!r}: "
            "define it at module scope (not nested / lambda) so the "
            "worker can re-import it."
        )
    file_path = Path(inspect.getfile(fn)).resolve()
    return f"{file_path}:{qualname}"


def resolve_spec_entrypoint(spec: CompiledWorkflow) -> str:
    """Return ``"<file>:<varname>"`` for *spec*.

    A ``CompiledWorkflow`` carries no source-level name; the worker re-imports it by
    looking up the variable that holds it. We find that module by
    asking the first registered task (which always lives in the same
    user module that assembled the spec) for its source, then scan
    that module's globals for a binding to *spec* by identity.

    Raises:
        ValueError: If the spec's first task has no associated module
            or the spec is not bound to a module-level name.
    """
    mod = inspect.getmodule(spec._tasks[0].fn_or_class)
    if mod is None:
        raise ValueError(
            f"cannot determine an importable entrypoint: the first "
            f"task body ({spec._tasks[0].fn_or_class!r}) does not "
            "belong to any module."
        )
    file_path = Path(inspect.getfile(mod)).resolve()
    for var, val in vars(mod).items():
        if val is spec:
            return f"{file_path}:{var}"
    raise ValueError(
        f"cannot determine an importable entrypoint: {spec!r} is not "
        f"bound to a module-level variable in {file_path}. Assign the "
        "spec to a name at module scope so the worker can re-import it."
    )


__all__ = [
    "promote_callable",
    "resolve_callable_entrypoint",
    "resolve_spec_entrypoint",
]
