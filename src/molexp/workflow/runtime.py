"""Abstract workflow runtime and default factory.

The runtime is the execution backend for :class:`WorkflowSpec`.
``GraphWorkflowRuntime`` (backed by pydantic-graph) is the only concrete
implementation.  Users never import the runtime directly —
``WorkflowSpec`` creates one lazily on first ``execute()`` / ``start()``.
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


def create_default_runtime() -> WorkflowRuntime:
    """Return the default pydantic-graph runtime."""
    from ._pydantic_graph.runtime import GraphWorkflowRuntime

    return GraphWorkflowRuntime()
