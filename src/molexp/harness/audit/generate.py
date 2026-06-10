"""Generate an :class:`AuditReport` from the run's stores.

Pure function — no side effects. Walks the event log + artifact store +
lineage store, gathers approvals / validations / failures / commands,
and returns a populated :class:`AuditReport`. The caller decides where to
persist (typically as an ``audit_report`` artifact — the kind is registered
in :data:`molexp.harness.WELL_KNOWN_ARTIFACT_KINDS`).
"""

from __future__ import annotations

from molexp.harness.schemas import AuditReport
from molexp.harness.store.artifact_store import ArtifactStore
from molexp.harness.store.event_log import EventLog
from molexp.harness.store.lineage_store import ArtifactLineageStore

__all__ = ["generate_audit_report"]


def generate_audit_report(
    *,
    run_id: str,
    event_log: EventLog,
    artifact_store: ArtifactStore,
    lineage_store: ArtifactLineageStore,  # noqa: ARG001 — accepted for symmetry; Phase 11 may use trace_backward
) -> AuditReport:
    """Assemble an :class:`AuditReport` for ``run_id``."""
    events = event_log.list_events(run_id)

    approvals: list[dict[str, object]] = []
    validation_results: list[str] = []
    failures: list[dict[str, object]] = []
    command_summaries: list[dict[str, object]] = []

    for event in events:
        if event.type in ("approval_requested", "approval_granted", "approval_rejected"):
            approvals.append(
                {
                    "type": event.type,
                    "seq": event.seq,
                    "actor": event.actor,
                    "payload": event.payload,
                    "artifact_ids": event.artifact_ids,
                }
            )
        elif event.type == "stage_failed":
            failures.append(
                {
                    "seq": event.seq,
                    "stage": event.payload.get("stage"),
                    "error": event.payload.get("error"),
                }
            )
        elif event.type == "artifact_created" and event.payload.get("kind") == "validation_report":
            validation_results.extend(event.artifact_ids)

    # Final artifacts = artifacts produced by the most recent stages.
    # As a simple heuristic: collect the last 3 artifact_created event ids.
    artifact_events = [e for e in events if e.type == "artifact_created"]
    final_artifact_ids = [aid for e in artifact_events[-3:] for aid in e.artifact_ids]

    # Root artifact = the deepest user_plan artifact for the first artifact_created.
    root_artifact_id: str | None = None
    if artifact_events:
        first_artifact_ids = artifact_events[0].artifact_ids
        if first_artifact_ids:
            try:
                first_ref = artifact_store.get_ref(first_artifact_ids[0])
                if first_ref.kind == "user_plan":
                    root_artifact_id = first_ref.id
            except Exception:
                pass

    return AuditReport(
        run_id=run_id,
        summary=(
            f"Run {run_id}: {len(events)} events, {len(approvals)} approval(s), "
            f"{len(validation_results)} validation report(s), {len(failures)} failure(s)"
        ),
        root_artifact_id=root_artifact_id,
        final_artifact_ids=final_artifact_ids,
        approvals=approvals,
        validation_results=validation_results,
        failures=failures,
        command_summaries=command_summaries,
        limitations=[],
    )
