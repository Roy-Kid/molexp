"""Workflow engine internals — lowering, structural scheduler, persistence.

This subpackage is the sole permitted ``import pydantic_graph`` site
under ``src/molexp/workflow/`` — the surviving pg surface is the ``End``
sentinel re-export (``molexp.workflow.End is pydantic_graph.End``).
Execution itself is molexp-owned: the compiler lowers the topology to an
:class:`~.plan.ExecutionPlan` and :mod:`.engine` runs it with
values-on-edges semantics (inputs delivered from upstream outputs,
structural deadlock detection, no timing constants).
"""

from pydantic_graph import End as End

from .compiler import WorkflowGraphCompiler
from .runtime import WorkflowRuntime

__all__ = [
    "End",
    "WorkflowGraphCompiler",
    "WorkflowRuntime",
]
