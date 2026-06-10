"""``GenerateAuditReport`` — persist the run-level audit report as an artifact.

Pure stage: delegates to the existing
:func:`molexp.harness.audit.generate_audit_report` over the run's event log,
artifact store, and lineage store, and persists the resulting
:class:`AuditReport` as an ``audit_report`` artifact. The report is a
run-level synthesis, so it carries no ``parent_ids`` — its provenance is the
whole event timeline, not a single upstream artifact.
"""

from __future__ import annotations

import json
from typing import ClassVar

from molexp.harness.audit import generate_audit_report
from molexp.harness.core.run_context import HarnessRunContext
from molexp.harness.core.stage import Stage
from molexp.harness.schemas import ArtifactRef

__all__ = ["GenerateAuditReport"]


class GenerateAuditReport(Stage):
    """Generate and persist the run's AuditReport artifact."""

    name: ClassVar[str] = "generate_audit_report"

    async def run(self, ctx: HarnessRunContext) -> ArtifactRef:
        report = generate_audit_report(
            run_id=ctx.run_id,
            event_log=ctx.event_log,
            artifact_store=ctx.artifact_store,
            lineage_store=ctx.lineage_store,
        )
        return ctx.artifact_store.put_json(
            kind="audit_report",
            obj=json.loads(report.model_dump_json()),
            created_by="GenerateAuditReport",
            parent_ids=[],
        )
