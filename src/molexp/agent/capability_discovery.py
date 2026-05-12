"""Agent-level capability discovery service boundary.

PlanMode depends on the service protocol shape, not on hint policy,
MCP transports, source indexes, docs search, or concrete namespaces.
The default implementation composes a generic probe with a hint policy
and enriches the probe's structured reports with discovery hints.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from molexp.agent.capability_hints import (
    CapabilityHint,
    CapabilityHintPolicy,
    CapabilityTriggerInput,
    TrustedNamespacePolicy,
)
from molexp.agent.modes.plan.context import PlanRepairContext

if TYPE_CHECKING:
    from molexp.agent.modes.plan.capability import (
        CapabilityEvidence,
        CapabilityEvidenceBatch,
        CapabilityNeed,
        CapabilityNeedReport,
    )
    from molexp.agent.modes.plan.schemas import PlanBrief

__all__ = [
    "CapabilityDiscoveryService",
    "DefaultCapabilityDiscoveryService",
]


@runtime_checkable
class _CapabilityProbeLike(Protocol):
    async def draft_needs(
        self,
        *,
        plan_brief: PlanBrief,
        repair_context: PlanRepairContext | None = None,
    ) -> CapabilityNeedReport: ...

    async def discover(
        self,
        report: CapabilityNeedReport,
        repair_context: PlanRepairContext | None = None,
    ) -> CapabilityEvidenceBatch: ...


@runtime_checkable
class CapabilityDiscoveryService(Protocol):
    """Resolve planning context into capability needs and evidence."""

    async def draft_needs(
        self,
        *,
        input: CapabilityTriggerInput,
        repair_context: PlanRepairContext | None = None,
    ) -> CapabilityNeedReport:
        """Draft capability needs from planning context."""
        ...

    async def discover(
        self,
        report: CapabilityNeedReport,
        *,
        input: CapabilityTriggerInput,
        repair_context: PlanRepairContext | None = None,
    ) -> CapabilityEvidenceBatch:
        """Resolve a drafted need report into structured evidence."""
        ...


class DefaultCapabilityDiscoveryService:
    """Compose hint extraction with an underlying capability probe."""

    def __init__(
        self,
        *,
        probe: _CapabilityProbeLike,
        hint_policy: CapabilityHintPolicy | None = None,
    ) -> None:
        self._probe = probe
        self._hint_policy = hint_policy if hint_policy is not None else TrustedNamespacePolicy()

    async def draft_needs(
        self,
        *,
        input: CapabilityTriggerInput,
        repair_context: PlanRepairContext | None = None,
    ) -> CapabilityNeedReport:
        if input.plan_brief is None:
            from molexp.agent.modes.plan.capability import CapabilityNeedReport

            report = CapabilityNeedReport(
                discovery_required=False,
                rationale_summary="no plan brief available",
            )
        else:
            report = await self._probe.draft_needs(
                plan_brief=input.plan_brief,
                repair_context=repair_context,
            )
        trigger = input.model_copy(update={"draft_capability_needs": report})
        return self._apply_hints(report, trigger)

    async def discover(
        self,
        report: CapabilityNeedReport,
        *,
        input: CapabilityTriggerInput,
        repair_context: PlanRepairContext | None = None,
    ) -> CapabilityEvidenceBatch:
        trigger = input.model_copy(update={"draft_capability_needs": report})
        hinted_report = self._apply_hints(report, trigger)
        batch = await self._probe.discover(hinted_report, repair_context=repair_context)
        return self._annotate_batch(batch, hinted_report)

    def _apply_hints(
        self,
        report: CapabilityNeedReport,
        input: CapabilityTriggerInput,
    ) -> CapabilityNeedReport:
        from molexp.agent.modes.plan.capability import CapabilityNeed

        existing_hints = tuple(getattr(report, "hints", ()))
        extracted = tuple(self._hint_policy.extract(input))
        hints = _dedupe_hints((*existing_hints, *extracted))
        if not hints:
            return report

        needs = list(getattr(report, "needs", ()))
        discovery_required = bool(getattr(report, "discovery_required", False)) or any(
            hint.strength in {"required", "preferred"} for hint in hints
        )
        for hint in hints:
            if hint.strength not in {"required", "preferred"} and not discovery_required:
                continue
            if _need_already_mentions_hint(needs, hint):
                continue
            needs.append(
                CapabilityNeed(
                    task_id=f"hint:{hint.namespace}",
                    capability=f"resolve requested capability for namespace {hint.namespace}",
                    rationale=hint.reason,
                    expected_kind="namespace",
                    query_hints=hint.query_hints or (hint.namespace,),
                )
            )

        return report.model_copy(
            update={
                "discovery_required": discovery_required,
                "needs": tuple(needs),
                "hints": hints,
            }
        )

    def _annotate_batch(
        self,
        batch: CapabilityEvidenceBatch,
        report: CapabilityNeedReport,
    ) -> CapabilityEvidenceBatch:
        from molexp.agent.modes.plan.capability import MissingCapability

        hints = _dedupe_hints((*getattr(batch, "hints", ()), *getattr(report, "hints", ())))
        if not hints:
            return batch
        tracked = _tracked_namespaces(batch, hints)
        evidence_namespaces = _evidence_namespaces(getattr(batch, "evidence", ()))
        missing = list(getattr(batch, "missing", ()))
        fallback_reasons = list(getattr(batch, "fallback_reasons", ()))

        for hint in hints:
            if hint.strength == "required" and hint.namespace not in evidence_namespaces:
                detail = f"required namespace {hint.namespace} has no evidence"
                if not any(detail in row.detail for row in missing):
                    missing.append(
                        MissingCapability(
                            need=None,
                            reason="required_namespace_missing",
                            detail=detail,
                            repairable=True,
                        )
                    )
            elif hint.strength == "preferred" and hint.namespace not in evidence_namespaces:
                reason = (
                    f"preferred namespace {hint.namespace} had no evidence; "
                    "fallback is allowed if implementation records why"
                )
                if not any(hint.namespace in existing for existing in fallback_reasons):
                    fallback_reasons.append(reason)

        return batch.model_copy(
            update={
                "hints": hints,
                "tracked_namespaces": tracked,
                "missing": tuple(missing),
                "fallback_reasons": tuple(fallback_reasons),
            }
        )


def _dedupe_hints(hints: Iterable[CapabilityHint]) -> tuple[CapabilityHint, ...]:
    out: dict[str, CapabilityHint] = {}
    rank = {"hint": 0, "preferred": 1, "required": 2}
    for hint in hints:
        existing = out.get(hint.namespace)
        if existing is None:
            out[hint.namespace] = hint
            continue
        strength = (
            hint.strength if rank[hint.strength] > rank[existing.strength] else existing.strength
        )
        out[hint.namespace] = existing.model_copy(
            update={
                "strength": strength,
                "query_hints": tuple(dict.fromkeys((*existing.query_hints, *hint.query_hints))),
                "constraint_tags": tuple(
                    dict.fromkeys((*existing.constraint_tags, *hint.constraint_tags))
                ),
                "reason": existing.reason or hint.reason,
                "phrase": existing.phrase or hint.phrase,
            }
        )
    return tuple(out.values())


def _need_already_mentions_hint(
    needs: Iterable[CapabilityNeed],
    hint: CapabilityHint,
) -> bool:
    needle = hint.namespace.lower()
    for need in needs:
        capability = str(getattr(need, "capability", "")).lower()
        query_hints = " ".join(str(q).lower() for q in getattr(need, "query_hints", ()))
        if needle in capability or needle in query_hints:
            return True
    return False


def _tracked_namespaces(
    batch: CapabilityEvidenceBatch,
    hints: tuple[CapabilityHint, ...],
) -> tuple[str, ...]:
    namespaces = set(getattr(batch, "tracked_namespaces", ()))
    namespaces.update(hint.namespace for hint in hints)
    namespaces.update(_evidence_namespaces(getattr(batch, "evidence", ())))
    return tuple(sorted(ns for ns in namespaces if ns))


def _evidence_namespaces(evidence: Iterable[CapabilityEvidence]) -> set[str]:
    out: set[str] = set()
    for row in evidence:
        namespace = getattr(row, "namespace", "") or getattr(row, "package", "")
        if namespace:
            out.add(str(namespace))
    return out
