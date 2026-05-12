"""PlanMode capability-discovery nodes.

The workflow owns two orchestration nodes:

* :class:`DraftCapabilityNeeds` asks a generic discovery service to
  produce structured needs and persists ``capability/needs.yaml``.
* :class:`DiscoverCapabilities` asks the same service to resolve those
  needs into structured evidence and persists the evidence ledger.

The nodes do not know where evidence comes from. A compatibility
``NullCapabilityProbe`` remains for tests and callers that still inject
the older probe abstraction directly.
"""

from __future__ import annotations

from pathlib import Path

from mollog import get_logger

from molexp.agent.capability_discovery import (
    CapabilityDiscoveryService,
    DefaultCapabilityDiscoveryService,
)
from molexp.agent.capability_hints import CapabilityTriggerInput
from molexp.agent.modes.plan.capability import (
    CapabilityEvidenceBatch,
    CapabilityNeedReport,
)
from molexp.agent.modes.plan.context import PlanRepairContext
from molexp.agent.modes.plan.errors import CapabilityDiscoveryRequired
from molexp.agent.modes.plan.protocols import CapabilityProbe, PlanDeps
from molexp.agent.modes.plan.schemas import (
    PlanBrief,
    PlanBriefResult,
)
from molexp.agent.modes.plan.tasks import PlanLLMTask, PlanTask
from molexp.workflow import TaskContext

__all__ = [
    "DiscoverCapabilities",
    "DraftCapabilityNeeds",
    "NullCapabilityProbe",
]


_LOG = get_logger(__name__)


# ── Null fallback ─────────────────────────────────────────────────────────


class NullCapabilityProbe:
    """Fail-closed :class:`CapabilityProbe` used when no source is configured.

    :meth:`draft_needs` always returns a report with
    ``discovery_required=False`` so a pipeline running against pure
    stdlib code still completes; :meth:`discover` raises
    :class:`CapabilityDiscoveryRequired` whenever its input flips
    ``discovery_required=True``. Silently passing an empty evidence
    batch downstream would let codegen reference unevidenced APIs.
    """

    async def draft_needs(
        self,
        *,
        plan_brief: PlanBrief,
        repair_context: PlanRepairContext | None = None,
    ) -> CapabilityNeedReport:
        del repair_context
        del plan_brief
        return CapabilityNeedReport(
            discovery_required=False,
            needs=(),
            rationale_summary="no probe configured",
        )

    async def discover(
        self,
        report: CapabilityNeedReport,
        repair_context: PlanRepairContext | None = None,
    ) -> CapabilityEvidenceBatch:
        del repair_context
        if report.discovery_required:
            raise CapabilityDiscoveryRequired(
                "NullCapabilityProbe cannot perform discovery; configure a capability source.",
                reason="no_probe",
                detail="no capability probe configured",
            )
        return CapabilityEvidenceBatch(
            evidence=(),
            missing=(),
            discovery_skipped=True,
        )


# ── Pipeline nodes ────────────────────────────────────────────────────────


class DraftCapabilityNeeds(PlanLLMTask):
    """Ask the discovery service to draft per-stage capability needs.

    Runs immediately after ``DraftImplementationPlan`` so the workflow
    IR is not yet compiled — the only upstream is the
    :class:`PlanBriefResult`. The service drafts one
    :class:`CapabilityNeed` per stage that plausibly requires project
    code; the report is persisted to ``capability/needs.yaml`` and
    forwarded to ``DiscoverCapabilities`` which then resolves concrete
    evidence. ``CompileWorkflowIR`` /
    ``CompileTaskIR`` are downstream of discovery and consume the
    evidence batch.

    Inherits :class:`PlanLLMTask` so the policy table can route this
    node consistently, even though the actual implementation may live
    behind the service rather than :meth:`PlanLLMTask.invoke_llm`.
    """

    async def _execute(
        self,
        ctx: TaskContext[None, PlanDeps, PlanBriefResult],
    ) -> CapabilityNeedReport:
        plan_brief = ctx.inputs

        service = _resolve_discovery_service(ctx.deps)
        trigger = _trigger_input(ctx, plan_brief=plan_brief.plan_brief)
        _LOG.debug(
            f"[plan-node DraftCapabilityNeeds] start stages={len(plan_brief.plan_brief.stages)} "
            f"service={type(service).__name__}"
        )
        report = await service.draft_needs(
            input=trigger,
            repair_context=ctx.deps.repair_context,
        )
        ctx.deps.plan_folder.write_capability_needs(report)
        _LOG.debug(
            "[plan-node DraftCapabilityNeeds] done "
            f"discovery_required={report.discovery_required} needs={len(report.needs)}"
        )
        return report


class DiscoverCapabilities(PlanTask):
    """Resolve drafted needs into concrete API evidence.

    Delegates to ``ctx.deps.capability_discovery.discover(...)`` and
    persists ``capability/evidence.yaml`` + ``capability/missing.md``.
    Re-raises :class:`CapabilityDiscoveryRequired` (the fallthrough
    failure from :class:`NullCapabilityProbe`) so the
    ``planmode-review-repair-loop`` driver can intercept it without
    swallowing the error mid-pipeline.

    Pure :class:`PlanTask` — no LLM invocation happens here from the
    pipeline's perspective.
    """

    async def _execute(
        self,
        ctx: TaskContext[None, PlanDeps, CapabilityNeedReport],
    ) -> CapabilityEvidenceBatch:
        report = ctx.inputs
        service = _resolve_discovery_service(ctx.deps)
        trigger = _trigger_input(ctx, draft_capability_needs=report)
        _LOG.debug(
            f"[plan-node DiscoverCapabilities] start needs={len(report.needs)} "
            f"discovery_required={report.discovery_required} service={type(service).__name__}"
        )
        try:
            batch = await service.discover(
                report,
                input=trigger,
                repair_context=ctx.deps.repair_context,
            )
        except CapabilityDiscoveryRequired:
            _LOG.warning(
                "[plan-node DiscoverCapabilities] failed — discovery required but unavailable"
            )
            raise

        handle = ctx.deps.plan_folder
        handle.write_capability_evidence(batch)
        handle.write_capability_missing(batch.missing)
        _LOG.debug(
            "[plan-node DiscoverCapabilities] done "
            f"evidence={len(batch.evidence)} missing={len(batch.missing)} "
            f"skipped={batch.discovery_skipped}"
        )
        return batch


# ── Helpers ───────────────────────────────────────────────────────────────


def _resolve_discovery_service(deps: PlanDeps) -> CapabilityDiscoveryService:
    """Return the configured discovery service or wrap a compatibility probe."""
    if deps.capability_discovery is not None:
        return deps.capability_discovery
    return DefaultCapabilityDiscoveryService(probe=_resolve_probe(deps))


def _resolve_probe(deps: PlanDeps) -> CapabilityProbe:
    """Return ``deps.capability_probe`` or fall back to :class:`NullCapabilityProbe`.

    Compatibility path for tests and older callers. New runtime wiring
    should inject ``capability_discovery``.
    """
    probe = deps.capability_probe
    if probe is None:
        return NullCapabilityProbe()
    return probe


def _trigger_input(
    ctx: TaskContext[None, PlanDeps, object],
    *,
    plan_brief: object | None = None,
    draft_capability_needs: object | None = None,
) -> CapabilityTriggerInput:
    config = getattr(ctx, "config", {}) or {}
    raw_user_input = config.get("user_input", "") if isinstance(config, dict) else ""
    handle = ctx.deps.plan_folder
    return CapabilityTriggerInput(
        raw_user_input=raw_user_input if isinstance(raw_user_input, str) else "",
        project_digest=_read_optional(handle.report_dir() / "digest.md"),
        plan_brief=plan_brief
        if plan_brief is not None
        else _read_optional(handle.plan_dir() / "implementation_plan.md"),
        draft_capability_needs=draft_capability_needs,
    )


def _read_optional(path: Path) -> str | None:
    try:
        if path.exists():
            return path.read_text(encoding="utf-8")
    except OSError:
        return None
    return None
