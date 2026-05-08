"""``PlanMode`` unit tests — provider-injected, structured-I/O workflow."""

from __future__ import annotations

import pytest
from pydantic import BaseModel, ValidationError

from molexp.agent.mode import AgentRunResult
from molexp.agent.modes import PlanMode, PlanModeConfig, PlanResult
from molexp.agent.modes.plan import PLAN_WORKFLOW
from molexp.agent.modes.plan.protocols import (
    AutoApproveGatePolicy,
    InMemoryPlanStore,
    ModelTier,
    NoOpArtifactWriter,
    PlanDeps,
)
from molexp.agent.modes.plan.schemas import (
    ApprovalDecision,
    ApprovedPlan,
    CodegenOutput,
    ContextSpec,
    Decomposition,
    GeneratedTaskSpec,
    GoalSpec,
    IntakeSpec,
    MethodSpec,
    PlanPatch,
    ProtocolDraft,
    ProtocolStep,
    RepairReport,
)
from molexp.agent.session import AgentSession

# ── Test fixtures ─────────────────────────────────────────────────────────


def _canned_presets() -> dict[type, BaseModel]:
    """Default canned schema instances — every PlanLLMTask step is covered."""
    return {
        IntakeSpec: IntakeSpec(
            request="seed", extracted_goal="screen Suzuki yields", constraints=("c1",)
        ),
        GoalSpec: GoalSpec(objective="optimise yield", success_criteria=("≥ 90%",)),
        ContextSpec: ContextSpec(constraints=("c1",), assumptions=("a1",)),
        MethodSpec: MethodSpec(name="Suzuki-Miyaura", rationale="tested"),
        Decomposition: Decomposition(stages=("prep", "couple", "isolate")),
        ProtocolDraft: ProtocolDraft(
            steps=(
                ProtocolStep(stage="prep", operation="weigh reagents"),
                ProtocolStep(stage="couple", operation="reflux 6h"),
                ProtocolStep(stage="isolate", operation="filter + dry"),
            )
        ),
        CodegenOutput: CodegenOutput(
            generated=(
                GeneratedTaskSpec(stage="prep", task_id="prep_task", code="pass"),
                GeneratedTaskSpec(stage="couple", task_id="couple_task", code="pass"),
            )
        ),
    }


class _FakeProvider:
    """Returns canned schema instances; tracks per-tier call counts."""

    def __init__(self, presets: dict[type, BaseModel] | None = None) -> None:
        self._presets = presets or _canned_presets()
        self.calls: list[tuple[ModelTier, type[BaseModel]]] = []

    async def invoke(
        self,
        *,
        tier: ModelTier,
        system: str,
        user: str,
        schema: type[BaseModel],
        node_id: str = "",
    ) -> BaseModel:
        del system, user, node_id
        self.calls.append((tier, schema))
        return self._presets[schema]


class _CountingGatePolicy:
    """Reject Gate B the first ``reject_b_until_call`` times, then approve."""

    def __init__(self, reject_b_until_call: int = 0) -> None:
        self._reject_b_until = reject_b_until_call
        self.gate_a_calls = 0
        self.gate_b_calls = 0

    async def gate_a(self, _preview):
        self.gate_a_calls += 1
        return ApprovalDecision(approved=True)

    async def gate_b(self, _executable, _compile_report, _dry_run_report):
        self.gate_b_calls += 1
        return ApprovalDecision(approved=self.gate_b_calls > self._reject_b_until)


class _RecordingRepairPolicy:
    def __init__(self) -> None:
        self.calls = 0

    async def patch(self, _preview, *, reason: str):
        del reason
        self.calls += 1
        return RepairReport(
            iteration=0,
            patches=(PlanPatch(target="method", new_value={"name": "patched"}),),
            affected_nodes=("method",),
            stale_nodes=("decomposition", "protocol", "codegen", "compile", "dry_run"),
        )


# ── Public-surface contract ───────────────────────────────────────────────


def test_plan_mode_carries_config() -> None:
    mode = PlanMode(provider=_FakeProvider(), max_iterations=5)
    assert mode.name == "plan"
    assert mode.config.max_iterations == 5


def test_plan_mode_config_is_frozen() -> None:
    cfg = PlanModeConfig()
    with pytest.raises(ValidationError):
        cfg.max_iterations = 99  # type: ignore[misc]


def test_plan_result_is_frozen() -> None:
    result = PlanResult(intake="i", design="d")
    with pytest.raises(ValidationError):
        result.intake = "x"  # type: ignore[misc]


# ── Workflow topology pin ─────────────────────────────────────────────────


def test_plan_workflow_topology() -> None:
    """The compiled workflow exposes every stage from the spec diagram."""
    assert {t.name for t in PLAN_WORKFLOW._tasks} == {
        "intake",
        "goal",
        "context",
        "method",
        "decomposition",
        "protocol",
        "preview",
        "gate_a",
        "codegen",
        "compile",
        "dry_run",
        "gate_b",
        "repair",
        "handoff",
    }
    # No wf.loop primitive — repair → preview is a plain control back-edge.
    assert PLAN_WORKFLOW._loops == ()
    assert ("repair", "preview") in PLAN_WORKFLOW._control_edges
    # No aggregator nodes (compose_plan / compose_executable).
    assert "compose_plan" not in {t.name for t in PLAN_WORKFLOW._tasks}
    assert "compose_executable" not in {t.name for t in PLAN_WORKFLOW._tasks}


# ── End-to-end happy path ─────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_plan_mode_runs_and_returns_structured_handoff() -> None:
    """Default auto-approve policy walks the pipeline once → ApprovedPlan."""

    provider = _FakeProvider()
    mode = PlanMode(provider=provider)
    result = await mode.run(
        harness=None,  # type: ignore[arg-type] — PlanMode ignores runner-supplied harness
        session=AgentSession(),
        user_input="design a workflow",
    )

    assert isinstance(result, AgentRunResult)
    assert result.mode_state is not None

    approved_dump = result.mode_state["approved_plan"]
    assert approved_dump is not None
    approved = ApprovedPlan.model_validate(approved_dump)
    assert approved.iterations == 0
    assert approved.compile_report.ok is True
    assert approved.dry_run_report.ok is True
    assert tuple(g.task_id for g in approved.executable.generated) == (
        "prep_task",
        "couple_task",
    )

    # Tier dispatch: cheap stages used CHEAP, decomposition/protocol/codegen used HEAVY.
    tiers = {schema: tier for tier, schema in provider.calls}
    assert tiers[IntakeSpec] is ModelTier.CHEAP
    assert tiers[GoalSpec] is ModelTier.CHEAP
    assert tiers[ContextSpec] is ModelTier.CHEAP
    assert tiers[MethodSpec] is ModelTier.DEFAULT
    assert tiers[Decomposition] is ModelTier.HEAVY
    assert tiers[ProtocolDraft] is ModelTier.HEAVY
    assert tiers[CodegenOutput] is ModelTier.HEAVY

    plan_compat = result.mode_state["plan"]
    assert plan_compat["intake"] == "screen Suzuki yields"
    assert plan_compat["approved"] is True


# ── Repair cycle ─────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_plan_mode_repair_cycle_recovers_from_gate_b_rejection() -> None:
    """Two Gate B rejections route through Repair → Preview, then approve."""

    provider = _FakeProvider()
    gate = _CountingGatePolicy(reject_b_until_call=2)
    repair = _RecordingRepairPolicy()
    mode = PlanMode(provider=provider, gate_policy=gate, repair_policy=repair)

    result = await mode.run(
        harness=None,  # type: ignore[arg-type]
        session=AgentSession(),
        user_input="design",
    )

    approved = ApprovedPlan.model_validate(result.mode_state["approved_plan"])
    assert approved.iterations == 2
    assert repair.calls == 2
    assert gate.gate_b_calls == 3
    # Specification chain runs once; only the codegen-onward sub-pipeline
    # re-runs across the repair cycle.
    intake_calls = sum(1 for _, s in provider.calls if s is IntakeSpec)
    goal_calls = sum(1 for _, s in provider.calls if s is GoalSpec)
    codegen_calls = sum(1 for _, s in provider.calls if s is CodegenOutput)
    assert intake_calls == 1
    assert goal_calls == 1
    assert codegen_calls == 3


# ── PlanDeps default wiring ──────────────────────────────────────────────


def test_plandeps_defaults_compose_cleanly() -> None:
    """``PlanMode()`` defaults to AutoApprove + IdentityRepair + InMemory store."""

    provider = _FakeProvider()
    mode = PlanMode(provider=provider)
    deps = mode._deps  # exercising the default wiring on purpose
    assert isinstance(deps, PlanDeps)
    assert isinstance(deps.gate_policy, AutoApproveGatePolicy)
    assert isinstance(deps.store, InMemoryPlanStore)
    assert isinstance(deps.artifact_writer, NoOpArtifactWriter)
