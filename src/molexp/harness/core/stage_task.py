"""``StageTask`` — adapt a harness :class:`Stage` to a ``molexp.workflow`` Runnable.

The harness pipeline runs on the ``molexp.workflow`` engine (see
:class:`molexp.harness.mode.Mode`): each :class:`Stage` is wrapped in a
``StageTask`` and registered on a ``WorkflowCompiler``. The engine schedules the
tasks by topology and threads the live :class:`HarnessRunContext` through every
task as ``ctx.run_context``.

The audit bracket — the ``stage_started`` / ``artifact_created`` /
``stage_completed`` / ``stage_failed`` events plus the ``derived_from``
provenance edges — is identical to the one :class:`StageRunner` has always
emitted. To keep the two execution paths byte-for-byte identical, that bracket
lives here in :func:`run_stage_bracketed`; ``StageRunner`` delegates to it, and
``StageTask.execute`` calls it too. The persistence writes are offloaded via
:func:`asyncio.to_thread` so the bracket never blocks the event loop (the
SQLite stores serialize concurrent worker access behind their shared per-file
lock).

``StageTask.execute`` returns the produced :class:`ArtifactRef`'s ``id`` (a
plain ``str``) as the workflow task output; the artifact itself flows through
``ctx.artifact_store``, never through the workflow output channel.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from molexp.harness.core.run_context import HarnessRunContext
from molexp.harness.core.stage import Stage
from molexp.harness.errors import StageExecutionError, StagePersistedFailureError

if TYPE_CHECKING:
    from molexp.harness.schemas import ArtifactRef

__all__ = ["StageTask", "run_stage_bracketed"]


async def _record_artifact(ctx: HarnessRunContext, stage: Stage, ref: ArtifactRef) -> None:
    """Emit ``artifact_created`` + the ref's ``derived_from`` provenance edges."""
    await asyncio.to_thread(
        ctx.event_log.append,
        run_id=ctx.run_id,
        type="artifact_created",
        actor="harness",
        payload={"stage": stage.name, "kind": ref.kind},
        artifact_ids=[ref.id],
    )
    for parent_id in ref.parent_ids:
        await asyncio.to_thread(
            ctx.provenance_store.add_edge,
            parent_id=parent_id,
            child_id=ref.id,
            relation="derived_from",
        )


async def run_stage_bracketed(ctx: HarnessRunContext, stage: Stage) -> ArtifactRef:
    """Run ``stage`` against ``ctx``, bracketed by the harness audit events.

    The single source of truth for the stage audit bracket — used by both
    :class:`StageRunner` and :class:`StageTask` so the legacy and engine-backed
    paths produce identical EventLog sequences and ProvenanceStore edges.

    Args:
        ctx: The run-scoped services container (artifact / event / provenance
            stores) bound to the current run.
        stage: The stage to execute.

    Returns:
        The :class:`ArtifactRef` the stage produced.

    Raises:
        StagePersistedFailureError: Re-raised unchanged after recording the
            persisted ref's ``artifact_created`` + edges and ``stage_failed``.
        StageExecutionError: For any other stage failure, after ``stage_failed``.
    """
    await asyncio.to_thread(
        ctx.event_log.append,
        run_id=ctx.run_id,
        type="stage_started",
        actor="harness",
        payload={"stage": stage.name},
    )

    try:
        ref = await stage.run(ctx)
    except StagePersistedFailureError as exc:
        await _record_artifact(ctx, stage, exc.persisted_ref)
        await asyncio.to_thread(
            ctx.event_log.append,
            run_id=ctx.run_id,
            type="stage_failed",
            actor="harness",
            payload={"stage": stage.name, "error": repr(exc)},
        )
        raise
    except Exception as exc:
        await asyncio.to_thread(
            ctx.event_log.append,
            run_id=ctx.run_id,
            type="stage_failed",
            actor="harness",
            payload={"stage": stage.name, "error": repr(exc)},
        )
        raise StageExecutionError(f"stage {stage.name!r} failed: {exc!r}") from exc

    await _record_artifact(ctx, stage, ref)
    await asyncio.to_thread(
        ctx.event_log.append,
        run_id=ctx.run_id,
        type="stage_completed",
        actor="harness",
        payload={"stage": stage.name},
    )
    return ref


class StageTask:
    """Wrap a :class:`Stage` as a ``molexp.workflow`` Runnable.

    The engine invokes :meth:`execute` with a ``TaskContext`` whose
    ``run_context`` payload is the live :class:`HarnessRunContext` (passed by
    :class:`~molexp.harness.mode.Mode` to ``Workflow.execute(run_context=...)``).
    The produced artifact's id is returned as the task output; the artifact
    itself is persisted via ``ctx.artifact_store`` inside the stage.
    """

    def __init__(self, stage: Stage) -> None:
        self._stage = stage

    @property
    def stage(self) -> Stage:
        return self._stage

    async def execute(self, ctx: object) -> str:
        """Run the wrapped stage and return its ``ArtifactRef.id``.

        Args:
            ctx: The workflow ``TaskContext``; its ``run_context`` must be the
                :class:`HarnessRunContext` for this run.

        Returns:
            The produced artifact's id, as the workflow task output.
        """
        harness_ctx = getattr(ctx, "run_context", None)
        if not isinstance(harness_ctx, HarnessRunContext):
            raise StageExecutionError(
                f"StageTask[{self._stage.name!r}] requires a HarnessRunContext as the "
                f"workflow run_context; got {type(harness_ctx).__name__}."
            )
        ref = await run_stage_bracketed(harness_ctx, self._stage)
        return ref.id
