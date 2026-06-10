"""The harness audit bracket — events + lineage edges around every ``Stage``.

:func:`run_stage_bracketed` is the single execution path for a harness stage.
Around every call to ``stage.run(ctx)`` it:

1. Appends ``stage_started`` with ``payload={"stage": stage.name}``.
2. ``await stage.run(ctx)`` → returns an :class:`ArtifactRef`.
3. Appends ``artifact_created`` carrying the returned ref's id.
4. Appends ``stage_completed``.
5. On exception: appends ``stage_failed`` with the ``repr(exc)`` payload
   and re-raises wrapped in :class:`StageExecutionError`.
6. For every ``parent_id`` listed in the returned ref's ``parent_ids``,
   calls ``lineage_store.add_edge(parent_id, ref.id,
   relation="derived_from", stage=stage.name, run_id=ctx.run_id)``.

If a Stage raises :class:`StagePersistedFailureError`, the bracket treats it
as a persisted-then-aborted failure: it still emits ``artifact_created`` +
``derived_from`` edges for the persisted ref, then emits ``stage_failed``
and re-raises unchanged. This preserves always-persist-then-raise
validators' audit trail (the :class:`ValidationReport` is visible to
``trace_backward`` even when the strict mode aborts the pipeline).

This way, individual stages stay pure (read inputs from the context, write
one artifact) while the harness owns the audit trail.
:class:`molexp.harness.mode.Mode` drives the bracket directly for its eager
pipeline; :class:`StageRunner` is the thin single-stage wrapper for direct
callers and tests.

Every store write (``event_log.append`` / ``lineage_store.add_edge``) is
blocking SQLite I/O, so the bracket dispatches each through
:func:`asyncio.to_thread`. That keeps the event loop responsive when the
pipeline runs behind a server route or alongside agent token-streaming; the
stores serialize concurrent worker-thread access behind their shared per-file
lock (see :mod:`molexp.harness.store._sqlite`).
"""

from __future__ import annotations

import asyncio

from molexp.harness.core.run_context import HarnessRunContext
from molexp.harness.core.stage import Stage
from molexp.harness.errors import StageExecutionError, StagePersistedFailureError
from molexp.harness.schemas import ArtifactRef

__all__ = ["StageRunner", "run_stage_bracketed"]


async def _record_artifact(ctx: HarnessRunContext, stage: Stage, ref: ArtifactRef) -> None:
    """Emit ``artifact_created`` + the ref's ``derived_from`` lineage edges.

    Each edge is stamped with the producing stage's name and the pipeline's
    ``run_id`` — what the pipeline legitimately knows at write time — so the
    lineage chain stays traversable end-to-end and links back to the
    ``workspace.Run`` the pipeline executed under.
    """
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
            ctx.lineage_store.add_edge,
            parent_id=parent_id,
            child_id=ref.id,
            relation="derived_from",
            stage=stage.name,
            run_id=ctx.run_id,
        )


async def run_stage_bracketed(ctx: HarnessRunContext, stage: Stage) -> ArtifactRef:
    """Run ``stage`` against ``ctx``, bracketed by the harness audit events.

    The single source of truth for the stage audit bracket — used by both
    :class:`StageRunner` and :class:`molexp.harness.mode.Mode` so every
    execution path produces identical EventLog sequences and
    ArtifactLineageStore edges.

    Args:
        ctx: The run-scoped services container (artifact / event / lineage
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


class StageRunner:
    """Bracket one :class:`Stage` with events + lineage edges.

    Thin wrapper over :func:`run_stage_bracketed` for direct single-stage
    callers; :class:`molexp.harness.mode.Mode` drives the bracket itself.
    """

    def __init__(self, ctx: HarnessRunContext) -> None:
        self._ctx = ctx

    async def run_stage(self, stage: Stage) -> ArtifactRef:
        return await run_stage_bracketed(self._ctx, stage)
