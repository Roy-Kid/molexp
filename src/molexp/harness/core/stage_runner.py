"""``StageRunner`` — wraps each ``Stage`` with events + provenance edges.

Around every call to ``stage.run(ctx)``, the runner:

1. Appends ``stage_started`` with ``payload={"stage": stage.name}``.
2. ``await stage.run(ctx)`` → returns an :class:`ArtifactRef`.
3. Appends ``artifact_created`` carrying the returned ref's id.
4. Appends ``stage_completed``.
5. On exception: appends ``stage_failed`` with the ``repr(exc)`` payload
   and re-raises wrapped in :class:`StageExecutionError`.
6. For every ``parent_id`` listed in the returned ref's ``parent_ids``,
   calls ``provenance_store.add_edge(parent_id, ref.id,
   relation="derived_from")``.

If a Stage raises :class:`StagePersistedFailureError`, the runner treats it
as a persisted-then-aborted failure: it still emits ``artifact_created`` +
``derived_from`` edges for the persisted ref, then emits ``stage_failed``
and re-raises as :class:`StageExecutionError`. This preserves
always-persist-then-raise validators' audit trail (the
:class:`ValidationReport` is visible to ``trace_backward`` even when the
strict mode aborts the pipeline).

This way, individual stages stay pure (read inputs from the context, write
one artifact) while the harness owns the audit trail.
"""

from __future__ import annotations

from molexp.harness.core.run_context import HarnessRunContext
from molexp.harness.core.stage import Stage
from molexp.harness.errors import StageExecutionError, StagePersistedFailureError
from molexp.harness.schemas import ArtifactRef

__all__ = ["StageRunner"]


class StageRunner:
    """Bracket each :class:`Stage` with events + lineage edges."""

    def __init__(self, ctx: HarnessRunContext) -> None:
        self._ctx = ctx

    async def run_stage(self, stage: Stage) -> ArtifactRef:
        ctx = self._ctx
        ctx.event_log.append(
            run_id=ctx.run_id,
            type="stage_started",
            actor="harness",
            payload={"stage": stage.name},
        )

        try:
            ref = await stage.run(ctx)
        except StagePersistedFailureError as exc:
            # Stage already persisted a failure artifact (e.g. parse-error
            # ValidationReport). Emit its artifact_created + edges so
            # trace_backward can still surface it, then fail the stage.
            # ``StagePersistedFailureError`` already extends
            # ``StageExecutionError`` so we re-raise unchanged rather than
            # double-wrapping.
            self._record_artifact(stage, exc.persisted_ref)
            ctx.event_log.append(
                run_id=ctx.run_id,
                type="stage_failed",
                actor="harness",
                payload={"stage": stage.name, "error": repr(exc)},
            )
            raise
        except Exception as exc:
            ctx.event_log.append(
                run_id=ctx.run_id,
                type="stage_failed",
                actor="harness",
                payload={"stage": stage.name, "error": repr(exc)},
            )
            raise StageExecutionError(f"stage {stage.name!r} failed: {exc!r}") from exc

        self._record_artifact(stage, ref)
        ctx.event_log.append(
            run_id=ctx.run_id,
            type="stage_completed",
            actor="harness",
            payload={"stage": stage.name},
        )
        return ref

    def _record_artifact(self, stage: Stage, ref: ArtifactRef) -> None:
        ctx = self._ctx
        ctx.event_log.append(
            run_id=ctx.run_id,
            type="artifact_created",
            actor="harness",
            payload={"stage": stage.name, "kind": ref.kind},
            artifact_ids=[ref.id],
        )
        for parent_id in ref.parent_ids:
            ctx.provenance_store.add_edge(
                parent_id=parent_id,
                child_id=ref.id,
                relation="derived_from",
            )
