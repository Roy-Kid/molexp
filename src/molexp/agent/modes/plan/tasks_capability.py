"""PlanMode capability-discovery nodes (Phase 4).

Two workflow nodes plus a fallback :class:`CapabilityProbe` implementation:

* :class:`NullCapabilityProbe` — used when no MCP server is configured.
  :meth:`draft_needs` returns ``CapabilityNeedReport(discovery_required=False, …)``;
  :meth:`discover` *fails closed* when its input report sets
  ``discovery_required=True`` so codegen never proceeds without
  evidence. Pure-stdlib paths (``discovery_required=False``) are still
  permitted.
* :class:`DraftCapabilityNeeds` — wraps the probe's ``draft_needs``
  call, persists the result to ``capability/needs.yaml``, and forwards
  the report downstream.
* :class:`DiscoverCapabilities` — wraps the probe's ``discover`` call,
  persists ``capability/evidence.yaml`` + ``capability/missing.md``,
  and re-raises :class:`CapabilityDiscoveryRequired` so the
  ``planmode-review-repair-loop`` driver can intercept and re-run the
  pair.

Both nodes route through the :class:`CapabilityProbe` Protocol on
``ctx.deps.capability_probe`` so production runs use the
``pydantic_ai``-backed probe while tests inject a :class:`StubCapabilityProbe`
without touching the SDK.

pydantic-ai does not own a *capability discovery* concept — it owns
model-side execution. The PlanMode workflow adds discovery as a
molexp-native pipeline concern, but the discovery agent itself is a
pydantic-ai ``Agent`` (see ``_pydanticai/capability_probe.py``).
"""

from __future__ import annotations

from mollog import get_logger

from molexp.agent.modes.plan.capability import (
    CapabilityEvidenceBatch,
    CapabilityNeedReport,
)
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
    """Fail-closed :class:`CapabilityProbe` used when no MCP server is configured.

    :meth:`draft_needs` always returns a report with
    ``discovery_required=False`` so a pipeline running against pure
    stdlib code still completes; :meth:`discover` raises
    :class:`CapabilityDiscoveryRequired` whenever its input flips
    ``discovery_required=True`` — there is no MCP server to call, so
    silently passing an empty evidence batch downstream would let
    codegen reference unevidenced Molcrafts APIs.
    """

    async def draft_needs(
        self,
        *,
        plan_brief: PlanBrief,
    ) -> CapabilityNeedReport:
        del plan_brief
        return CapabilityNeedReport(
            discovery_required=False,
            needs=(),
            rationale_summary="no probe configured",
        )

    async def discover(self, report: CapabilityNeedReport) -> CapabilityEvidenceBatch:
        if report.discovery_required:
            raise CapabilityDiscoveryRequired(
                "NullCapabilityProbe cannot perform discovery; configure molmcp.",
                reason="no_probe",
                detail="no probe configured; molmcp not available",
            )
        return CapabilityEvidenceBatch(
            evidence=(),
            missing=(),
            discovery_skipped=True,
        )


# ── Pipeline nodes ────────────────────────────────────────────────────────


class DraftCapabilityNeeds(PlanLLMTask):
    """Ask the probe to draft per-stage capability needs.

    Runs immediately after ``DraftImplementationPlan`` so the workflow
    IR is not yet compiled — the only upstream is the
    :class:`PlanBriefResult`. The probe drafts one
    :class:`CapabilityNeed` per stage that plausibly requires project
    code; the report is persisted to ``capability/needs.yaml`` and
    forwarded to ``DiscoverCapabilities`` which then queries the
    project MCP for concrete evidence. ``CompileWorkflowIR`` /
    ``CompileTaskIR`` are downstream of discovery and consume the
    evidence batch.

    Inherits :class:`PlanLLMTask` so the policy table can route this
    node to the heavy tier, even though the actual LLM call happens
    inside the probe rather than via :meth:`PlanLLMTask.invoke_llm`.
    The probe owns its own ``pydantic_ai.Agent`` and therefore its
    own model selection.
    """

    async def execute(
        self,
        ctx: TaskContext[None, PlanDeps, PlanBriefResult],
    ) -> CapabilityNeedReport:
        plan_brief = ctx.inputs

        probe = _resolve_probe(ctx.deps)
        _LOG.info(
            f"[plan-node DraftCapabilityNeeds] start stages={len(plan_brief.plan_brief.stages)} "
            f"probe={type(probe).__name__}"
        )
        report = await probe.draft_needs(plan_brief=plan_brief.plan_brief)
        ctx.deps.workspace_handle.write_capability_needs(report)
        _LOG.info(
            "[plan-node DraftCapabilityNeeds] done "
            f"discovery_required={report.discovery_required} needs={len(report.needs)}"
        )
        return report


class DiscoverCapabilities(PlanTask):
    """Resolve drafted needs into concrete API evidence.

    Delegates to ``ctx.deps.capability_probe.discover(...)`` and
    persists ``capability/evidence.yaml`` + ``capability/missing.md``.
    Re-raises :class:`CapabilityDiscoveryRequired` (the fallthrough
    failure from :class:`NullCapabilityProbe`) so the
    ``planmode-review-repair-loop`` driver can intercept it without
    swallowing the error mid-pipeline.

    Pure :class:`PlanTask` — no LLM invocation happens here from the
    pipeline's perspective; the probe owns its own ``pydantic_ai.Agent``
    and runs the MCP-attached agent loop internally.
    """

    async def execute(
        self,
        ctx: TaskContext[None, PlanDeps, CapabilityNeedReport],
    ) -> CapabilityEvidenceBatch:
        report = ctx.inputs
        probe = _resolve_probe(ctx.deps)
        _LOG.info(
            f"[plan-node DiscoverCapabilities] start needs={len(report.needs)} "
            f"discovery_required={report.discovery_required} probe={type(probe).__name__}"
        )
        try:
            batch = await probe.discover(report)
        except CapabilityDiscoveryRequired:
            _LOG.warning(
                "[plan-node DiscoverCapabilities] failed — discovery required but no probe"
            )
            raise

        handle = ctx.deps.workspace_handle
        handle.write_capability_evidence(batch)
        handle.write_capability_missing(batch.missing)
        _LOG.info(
            "[plan-node DiscoverCapabilities] done "
            f"evidence={len(batch.evidence)} missing={len(batch.missing)} "
            f"skipped={batch.discovery_skipped}"
        )
        return batch


# ── Helpers ───────────────────────────────────────────────────────────────


def _resolve_probe(deps: PlanDeps) -> CapabilityProbe:
    """Return ``deps.capability_probe`` or fall back to a fresh :class:`NullCapabilityProbe`.

    The runner injects a real probe in production; tests that don't
    bother to construct one transparently get the null fallback so a
    plain ``PlanDeps(...)`` instantiation never crashes the discovery
    nodes.
    """
    probe = deps.capability_probe
    if probe is None:
        return NullCapabilityProbe()
    return probe
