"""Abstract workflow runtime and default factory.

The runtime is the execution backend for :class:`WorkflowSpec`.
The default implementation delegates to ``pydantic-graph`` if installed,
otherwise returns a placeholder that raises on every call.

Users never import the runtime directly — ``WorkflowSpec`` creates one
lazily on first ``execute()`` / ``start()``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from .types import WorkflowExecution, WorkflowResult


class WorkflowRuntime(ABC):
    """Backend that knows how to execute a compiled workflow."""

    @abstractmethod
    async def execute(
        self,
        spec: Any,
        run: Any = None,
        run_context: Any = None,
        *,
        dry_run: bool = False,
        **kwargs: Any,
    ) -> WorkflowResult:
        """Run the workflow to completion."""
        ...

    @abstractmethod
    async def start(
        self,
        spec: Any,
        run: Any = None,
        run_context: Any = None,
        *,
        dry_run: bool = False,
        **kwargs: Any,
    ) -> WorkflowExecution:
        """Launch the workflow in the background."""
        ...


class _NotImplementedRuntime(WorkflowRuntime):
    """Placeholder when pydantic-graph is not installed."""

    _MSG = (
        "Workflow execution requires pydantic-graph. "
        "Install with: pip install molexp[workflow]"
    )

    async def execute(
        self,
        spec: Any,
        run: Any = None,
        run_context: Any = None,
        *,
        dry_run: bool = False,
        **kwargs: Any,
    ) -> WorkflowResult:
        raise NotImplementedError(self._MSG)

    async def start(
        self,
        spec: Any,
        run: Any = None,
        run_context: Any = None,
        *,
        dry_run: bool = False,
        **kwargs: Any,
    ) -> WorkflowExecution:
        raise NotImplementedError(self._MSG)


def create_default_runtime() -> WorkflowRuntime:
    """Select the best available runtime backend.

    Tries ``pydantic-graph`` first, falls back to a stub.
    """
    try:
        from ._pydantic_graph.runtime import GraphWorkflowRuntime

        return GraphWorkflowRuntime()
    except ImportError:
        return _NotImplementedRuntime()
