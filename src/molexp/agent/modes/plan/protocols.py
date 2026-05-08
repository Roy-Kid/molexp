"""Runtime services consumed by the PlanMode workflow via ``ctx.deps``.

PlanMode's ``ctx.deps`` is a frozen :class:`PlanDeps` aggregate of
runtime services — provider routing, gate / repair policies, plan
store, artifact writer. ``ctx.config`` carries JSON-only values
(``user_input`` etc.); ``ctx.deps`` carries the callables and stateful
services that don't fit through a JSON channel.

Each service is declared as a :class:`typing.Protocol` so user code can
substitute its own implementation without subclassing molexp internals.
Default in-memory / no-op implementations live at the bottom of this
module so :class:`~molexp.agent.modes.plan.PlanMode` can be constructed
without any wiring.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol, TypeVar, runtime_checkable

from pydantic import BaseModel

from molexp.agent.modes.plan.schemas import (
    ApprovalDecision,
    CompileReport,
    DryRunReport,
    ExecutableWorkflowDraft,
    PlanPreview,
    RepairReport,
)

# ── Tier vocabulary ────────────────────────────────────────────────────────


class ModelTier(StrEnum):
    """Semantic model tier — the only model identifier a Task may name.

    Concrete provider / model IDs live in :class:`Provider`
    configuration, never on Task classes. Tasks declare what they need
    (cheap parsing vs. heavy reasoning); operators decide what fulfils
    that need.
    """

    CHEAP = "cheap"
    DEFAULT = "default"
    HEAVY = "heavy"


# ── Provider ───────────────────────────────────────────────────────────────


SchemaT = TypeVar("SchemaT", bound=BaseModel)


@runtime_checkable
class Provider(Protocol):
    """LLM dispatch gateway — tier-routed, schema-typed.

    Tasks call this exactly once per LLM step. The provider owns model
    resolution, retries, tracing, and rate-limiting; tasks only know
    their tier and the schema they expect back.
    """

    async def invoke(
        self,
        *,
        tier: ModelTier,
        system: str,
        user: str,
        schema: type[SchemaT],
        node_id: str = "",
    ) -> SchemaT: ...


# ── Gate / repair policies ─────────────────────────────────────────────────


@runtime_checkable
class GatePolicy(Protocol):
    """External approval source — never the LLM that authored the plan."""

    async def gate_a(self, preview: PlanPreview) -> ApprovalDecision: ...

    async def gate_b(
        self,
        executable: ExecutableWorkflowDraft,
        compile_report: CompileReport,
        dry_run_report: DryRunReport,
    ) -> ApprovalDecision: ...


@runtime_checkable
class RepairPolicy(Protocol):
    """Source of plan patches when a gate / check rejects the plan."""

    async def patch(self, preview: PlanPreview, *, reason: str) -> RepairReport: ...


# ── Persistence services ───────────────────────────────────────────────────


@runtime_checkable
class PlanStore(Protocol):
    """Per-session plan-mode bookkeeping (iteration counter, etc.)."""

    def get_iteration(self) -> int: ...

    def note_iteration(self) -> None: ...


@runtime_checkable
class ArtifactWriter(Protocol):
    """Persist plan artifacts (generated code, reports) to disk / store."""

    def write(self, name: str, payload: BaseModel) -> str: ...


# ── PlanDeps aggregate ─────────────────────────────────────────────────────


@dataclass(frozen=True)
class PlanDeps:
    """Runtime services bundle threaded through ``ctx.deps``."""

    provider: Provider
    gate_policy: GatePolicy
    repair_policy: RepairPolicy
    store: PlanStore
    artifact_writer: ArtifactWriter


# ── Default implementations ────────────────────────────────────────────────


class AutoApproveGatePolicy:
    """Approves everything — the safe default for non-interactive use."""

    async def gate_a(self, _preview: PlanPreview) -> ApprovalDecision:
        return ApprovalDecision(approved=True)

    async def gate_b(
        self,
        _executable: ExecutableWorkflowDraft,
        _compile_report: CompileReport,
        _dry_run_report: DryRunReport,
    ) -> ApprovalDecision:
        return ApprovalDecision(approved=True)


class IdentityRepairPolicy:
    """No-op repair — returns an empty patch set."""

    async def patch(self, _preview: PlanPreview, *, reason: str) -> RepairReport:
        del reason
        return RepairReport(iteration=0, patches=(), affected_nodes=(), stale_nodes=())


class InMemoryPlanStore:
    """Process-local iteration counter; resets per :class:`PlanMode` instance."""

    def __init__(self) -> None:
        self._iteration = 0

    def get_iteration(self) -> int:
        return self._iteration

    def note_iteration(self) -> None:
        self._iteration += 1


class NoOpArtifactWriter:
    """Discards artifacts — placeholder until a workspace-backed writer lands."""

    def write(self, name: str, _payload: BaseModel) -> str:
        return name


__all__ = [
    "ArtifactWriter",
    "AutoApproveGatePolicy",
    "GatePolicy",
    "IdentityRepairPolicy",
    "InMemoryPlanStore",
    "ModelTier",
    "NoOpArtifactWriter",
    "PlanDeps",
    "PlanStore",
    "Provider",
    "RepairPolicy",
]
