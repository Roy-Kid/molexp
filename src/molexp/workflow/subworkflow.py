"""Sanctioned sub-workflow composition node.

:class:`SubWorkflow` is the supported way to embed a reusable inner workflow
as a single node of an outer workflow — including as the per-element ``body``
of :meth:`WorkflowCompiler.parallel`. It replaces the ad-hoc pattern of
hand-building a child :class:`~molexp.workflow.context.TaskContext` and calling
an inner task's ``execute`` directly.

The node IS a single registered task from the outer graph's perspective, so it
slots into ``builder.parallel(body="<subworkflow_name>")`` with zero changes to
``ParallelDecl`` / the compiler / the pg lowering: the outer engine fans out the
single ``SubWorkflow`` node per element, and the node itself runs the inner spec
end-to-end through the engine.

run_context-forwarding contract
-------------------------------
``SubWorkflow.execute`` runs the inner spec via
``WorkflowRuntime().execute(inner, run_context=ctx.run_context)`` — it forwards
the *outer* ``run_context`` object **by identity**. Inner tasks therefore observe
the very same workspace / run the outer execution received. The node never
constructs a :class:`TaskContext`; the engine builds every inner context.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ._pydantic_graph.runtime import WorkflowRuntime
from .compiled import CompiledWorkflow
from .task import Task

if TYPE_CHECKING:
    from .compiler import WorkflowCompiler
    from .context import TaskContext


class SubWorkflow(Task):
    """Run a reusable inner workflow as one node of an outer workflow.

    Construct from an already-compiled :class:`CompiledWorkflow` or from a
    :class:`WorkflowCompiler` (which is compiled eagerly at construction). Each
    :meth:`execute` call triggers one fresh inner execution through
    :class:`WorkflowRuntime`, forwarding the outer ``run_context`` so inner-task
    workspace helpers keep working.

    Args:
        inner: The inner workflow to embed — a :class:`CompiledWorkflow`, or a
            :class:`WorkflowCompiler` which is compiled on construction.
        output: Name of the inner task whose output is returned. When omitted,
            the inner spec's single dependency-leaf (a task no other task
            depends on) is used; if the inner spec has more than one leaf and no
            ``output`` is given, :meth:`execute` raises :class:`ValueError`.

    The inner reference is owned by this instance and never mutated.
    """

    def __init__(
        self, inner: CompiledWorkflow | WorkflowCompiler, *, output: str | None = None
    ) -> None:
        self._inner: CompiledWorkflow = (
            inner if isinstance(inner, CompiledWorkflow) else inner.compile()
        )
        self._output = output

    @property
    def inner(self) -> CompiledWorkflow:
        """The compiled inner workflow embedded by this node (immutable)."""
        return self._inner

    def _resolve_output_name(self) -> str:
        """Return the inner task name whose output this node yields.

        Uses ``output=`` when supplied; otherwise computes the single
        dependency-leaf (a task no other task depends on). Raises
        :class:`ValueError` when no explicit ``output`` is given and the inner
        spec does not have exactly one leaf.
        """
        if self._output is not None:
            return self._output
        registrations = self._inner.registration_by_name
        names = set(registrations)
        depended_on: set[str] = set()
        for reg in registrations.values():
            depended_on.update(reg.depends_on)
        leaves = sorted(names - depended_on)
        if len(leaves) == 1:
            return leaves[0]
        raise ValueError(
            f"SubWorkflow over inner workflow {self._inner.name!r} has "
            f"{len(leaves)} terminal leaf task(s) {leaves!r}; pass "
            f"output='<task_name>' to select which inner output to return."
        )

    async def execute(self, ctx: TaskContext) -> object:
        """Run the inner workflow through the engine and return its terminal output.

        Forwards ``ctx.run_context`` (by identity) and ``ctx.config`` to the
        inner execution. On a non-``"completed"`` inner status, raises a
        :class:`RuntimeError` so the failure propagates (under ``wf.parallel``
        the engine's existing ``ParallelExecutionError`` aggregation wraps it).

        Args:
            ctx: The outer :class:`TaskContext` supplied by the engine.

        Returns:
            The output of the configured inner task (``output=``) or the inner
            spec's single dependency-leaf output.
        """
        output_name = self._resolve_output_name()
        result = await WorkflowRuntime().execute(
            self._inner,
            run_context=ctx.run_context,
            config=ctx.config,
        )
        if result.status != "completed":
            raise RuntimeError(
                f"SubWorkflow inner workflow {self._inner.name!r} ended with "
                f"status {result.status!r}"
            )
        return result.outputs[output_name]


__all__ = ["SubWorkflow"]
