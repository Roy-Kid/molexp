"""Tests for :mod:`molexp.agent.modes.plan.policy`.

Covers acceptance criteria ac-001..ac-010 for sub-spec
``planmode-workspace-pipeline-04-plan-model-policy``.
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Any, ClassVar, TypeVar

import pytest
from pydantic import BaseModel, ValidationError

from molexp.agent.modes.plan import (
    PLAN_NODE_NAMES,
    PLAN_WORKFLOW,
    STANDARD_PLAN_POLICY,
    PlanMode,
    PlanModelPolicy,
)
from molexp.agent.modes.plan.protocols import (
    AutoApproveGatePolicy,
    IdentityRepairPolicy,
    InMemoryPlanStore,
    ModelTier,
    NoOpArtifactWriter,
    PlanDeps,
    Provider,
)

# ── tier_for (ac-001) ──────────────────────────────────────────────────────


def test_tier_for_returns_override_when_present() -> None:
    policy = PlanModelPolicy(
        default_tier=ModelTier.DEFAULT,
        node_tiers={"IntakeTask": ModelTier.CHEAP},
    )
    assert policy.tier_for("IntakeTask") is ModelTier.CHEAP


def test_tier_for_returns_default_when_node_unmapped() -> None:
    policy = PlanModelPolicy(default_tier=ModelTier.DEFAULT)
    assert policy.tier_for("MethodTask") is ModelTier.DEFAULT


def test_tier_for_unknown_but_allowed_node_returns_default() -> None:
    """A node id that is in PLAN_NODE_NAMES but not in node_tiers
    falls back to default_tier."""
    policy = PlanModelPolicy(default_tier=ModelTier.HEAVY)
    assert policy.tier_for("HandoffTask") is ModelTier.HEAVY


# ── Validation (ac-002) ────────────────────────────────────────────────────


def test_plan_model_policy_rejects_unknown_node_id() -> None:
    with pytest.raises(ValidationError) as exc_info:
        PlanModelPolicy(node_tiers={"NotAPlanNode": ModelTier.HEAVY})
    assert "NotAPlanNode" in str(exc_info.value)


def test_plan_model_policy_is_frozen() -> None:
    policy = PlanModelPolicy()
    with pytest.raises(ValidationError):
        policy.default_tier = ModelTier.HEAVY  # type: ignore[misc]


def test_plan_model_policy_rejects_unknown_field() -> None:
    with pytest.raises(ValidationError):
        PlanModelPolicy(stray=1)  # type: ignore[call-arg]


# ── PLAN_NODE_NAMES (ac-003) ───────────────────────────────────────────────


def test_plan_node_names_is_frozenset_of_str() -> None:
    assert isinstance(PLAN_NODE_NAMES, frozenset)
    assert all(isinstance(n, str) for n in PLAN_NODE_NAMES)


def test_plan_node_names_covers_current_pipeline_classes() -> None:
    expected = {
        "IntakeTask",
        "GoalTask",
        "ContextTask",
        "MethodTask",
        "DecompositionTask",
        "ProtocolTask",
        "PreviewTask",
        "GateATask",
        "CodegenTask",
        "CompileTask",
        "DryRunTask",
        "GateBTask",
        "RepairTask",
        "HandoffTask",
    }
    assert expected.issubset(PLAN_NODE_NAMES)


def test_plan_node_names_covers_subspec_05_node_ids() -> None:
    expected = {
        "IngestReport",
        "DraftReportDigest",
        "ValidateWorkspace",
        "DraftImplementationPlan",
        "CompileWorkflowIR",
        "CompileTaskIR",
        "GenerateWorkflowSkeleton",
        "GenerateTaskTests",
        "GenerateTaskImplementations",
        "RepairOnValidationFailure",
    }
    assert expected.issubset(PLAN_NODE_NAMES)


# ── STANDARD_PLAN_POLICY (ac-004 + ac-005) ─────────────────────────────────


@pytest.mark.parametrize(
    ("node", "expected_tier"),
    [
        ("IngestReport", ModelTier.CHEAP),
        ("DraftReportDigest", ModelTier.CHEAP),
        ("ValidateWorkspace", ModelTier.CHEAP),
        ("DraftImplementationPlan", ModelTier.DEFAULT),
        ("CompileWorkflowIR", ModelTier.DEFAULT),
        ("CompileTaskIR", ModelTier.DEFAULT),
        ("GenerateWorkflowSkeleton", ModelTier.DEFAULT),
        ("GenerateTaskTests", ModelTier.DEFAULT),
        ("GenerateTaskImplementations", ModelTier.HEAVY),
        ("RepairOnValidationFailure", ModelTier.HEAVY),
    ],
)
def test_standard_policy_subspec_05_table(node: str, expected_tier: ModelTier) -> None:
    assert STANDARD_PLAN_POLICY.tier_for(node) is expected_tier


@pytest.mark.parametrize(
    ("node", "expected_tier"),
    [
        ("IntakeTask", ModelTier.CHEAP),
        ("GoalTask", ModelTier.CHEAP),
        ("ContextTask", ModelTier.CHEAP),
        ("MethodTask", ModelTier.DEFAULT),
        ("DecompositionTask", ModelTier.HEAVY),
        ("ProtocolTask", ModelTier.HEAVY),
        ("CodegenTask", ModelTier.HEAVY),
    ],
)
def test_standard_policy_preserves_current_tier_table(node: str, expected_tier: ModelTier) -> None:
    assert STANDARD_PLAN_POLICY.tier_for(node) is expected_tier


# ── tasks.py source-level invariants (ac-006) ──────────────────────────────


_TASKS_PY_PATH = (
    Path(__file__).parent.parent.parent.parent.parent
    / "src"
    / "molexp"
    / "agent"
    / "modes"
    / "plan"
    / "tasks.py"
)


def _ast_classvar_targets(tree: ast.AST) -> set[str]:
    """Collect names assigned as ClassVar within class bodies."""
    found: set[str] = set()
    for class_node in ast.walk(tree):
        if not isinstance(class_node, ast.ClassDef):
            continue
        for stmt in class_node.body:
            if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
                found.add(stmt.target.id)
            elif isinstance(stmt, ast.Assign):
                for tgt in stmt.targets:
                    if isinstance(tgt, ast.Name):
                        found.add(tgt.id)
    return found


def test_tasks_py_has_no_tier_classvar() -> None:
    """No subclass should still declare ``TIER`` as a ClassVar — the
    policy is the single source of truth."""
    source = _TASKS_PY_PATH.read_text()
    tree = ast.parse(source)
    assignments = _ast_classvar_targets(tree)
    assert "TIER" not in assignments, (
        "TIER ClassVar leftover in tasks.py — should be removed in favor of "
        "ctx.deps.model_policy.tier_for(...)"
    )


def test_tasks_py_invoke_llm_uses_model_policy() -> None:
    """``invoke_llm`` must resolve tier via the deps' model_policy."""
    source = _TASKS_PY_PATH.read_text()
    assert "ctx.deps.model_policy.tier_for(" in source
    # And no literal ModelTier.CHEAP/DEFAULT/HEAVY appears in tasks.py
    # — every site goes through the policy.
    assert "ModelTier.CHEAP" not in source
    assert "ModelTier.DEFAULT" not in source
    assert "ModelTier.HEAVY" not in source


# ── PlanDeps additive field (ac-008) ───────────────────────────────────────


def test_plan_deps_has_model_policy_field() -> None:
    fields = PlanDeps.__dataclass_fields__
    assert "model_policy" in fields


def test_plan_deps_default_construction_uses_standard_policy() -> None:
    deps = PlanDeps(
        provider=_RecordingProvider(),  # type: ignore[arg-type]
        gate_policy=AutoApproveGatePolicy(),
        repair_policy=IdentityRepairPolicy(),
        store=InMemoryPlanStore(),
        artifact_writer=NoOpArtifactWriter(),
    )
    assert deps.model_policy is STANDARD_PLAN_POLICY


# ── Public re-exports (ac-009) ─────────────────────────────────────────────


def test_planmode_modes_plan_re_exports_policy_names() -> None:
    import molexp.agent.modes.plan as plan_pkg

    assert plan_pkg.PlanModelPolicy is PlanModelPolicy
    assert plan_pkg.STANDARD_PLAN_POLICY is STANDARD_PLAN_POLICY
    assert "PlanModelPolicy" in plan_pkg.__all__
    assert "STANDARD_PLAN_POLICY" in plan_pkg.__all__


# ── End-to-end policy injection (ac-007) ───────────────────────────────────


SchemaT = TypeVar("SchemaT", bound=BaseModel)


class _RecordingProvider:
    """Stub Provider that records every ``(node_id, tier)`` it sees.

    Returns canned schema instances so PLAN_WORKFLOW can run end-to-end.
    The mapping from schema name → constructor kwargs is hand-tuned for
    the present-day plan workflow.
    """

    def __init__(self) -> None:
        self.calls: list[tuple[str, ModelTier]] = []

    async def invoke(
        self,
        *,
        tier: ModelTier,
        system: str,
        user: str,
        schema: type[SchemaT],
        node_id: str = "",
    ) -> SchemaT:
        self.calls.append((node_id, tier))
        return _build_canned_schema(schema)

    async def invoke_with_template(
        self,
        *,
        tier: ModelTier,
        system: str,
        user_template: str,
        user_context: Any,
        schema: type[SchemaT],
        node_id: str = "",
    ) -> SchemaT:
        return await self.invoke(tier=tier, system=system, user="", schema=schema, node_id=node_id)


def _build_canned_schema[SchemaT: BaseModel](schema: type[SchemaT]) -> SchemaT:
    """Construct a minimal valid instance of one of the plan-mode schemas."""
    from molexp.agent.modes.plan.schemas import (
        ContextSpec,
        Decomposition,
        GoalSpec,
        IntakeSpec,
        MethodSpec,
        ProtocolDraft,
        ProtocolStep,
    )

    name = schema.__name__
    if name == "IntakeSpec":
        return IntakeSpec(request="r", extracted_goal="g")  # type: ignore[return-value]
    if name == "GoalSpec":
        return GoalSpec(objective="o")  # type: ignore[return-value]
    if name == "ContextSpec":
        return ContextSpec()  # type: ignore[return-value]
    if name == "MethodSpec":
        return MethodSpec(name="m")  # type: ignore[return-value]
    if name == "Decomposition":
        return Decomposition(stages=("s1",))  # type: ignore[return-value]
    if name == "ProtocolDraft":
        return ProtocolDraft(
            steps=(ProtocolStep(stage="s1", operation="op"),),
        )  # type: ignore[return-value]
    if name == "CodegenOutput":
        from molexp.agent.modes.plan.schemas import CodegenOutput, GeneratedTaskSpec

        return CodegenOutput(
            tasks=(GeneratedTaskSpec(stage="s1", code="class S1: pass"),),
        )  # type: ignore[return-value]
    raise NotImplementedError(f"_build_canned_schema does not handle {name!r}")


def test_provider_protocol_runtime_check() -> None:
    """The recording stub matches the Provider protocol."""
    assert isinstance(_RecordingProvider(), Provider)


@pytest.mark.asyncio
async def test_custom_policy_observed_by_provider_invoke() -> None:
    """Driving the present-day workflow with a custom policy puts the
    overridden tier on every LLM-bearing call."""
    provider = _RecordingProvider()
    policy = PlanModelPolicy(default_tier=ModelTier.CHEAP)
    deps = PlanDeps(
        provider=provider,  # type: ignore[arg-type]
        gate_policy=AutoApproveGatePolicy(),
        repair_policy=IdentityRepairPolicy(),
        store=InMemoryPlanStore(),
        artifact_writer=NoOpArtifactWriter(),
        model_policy=policy,
    )
    await PLAN_WORKFLOW.execute(config={"user_input": "hello"}, deps=deps)
    assert provider.calls, "stub provider should have been invoked at least once"
    for node_id, tier in provider.calls:
        assert tier is ModelTier.CHEAP, (
            f"node {node_id!r} received tier={tier} under default-CHEAP policy"
        )


@pytest.mark.asyncio
async def test_planmode_constructor_threads_custom_policy_into_deps() -> None:
    """Sanity check on ``PlanMode(model_policy=...)`` plumbing — the
    resulting ``_deps.model_policy`` must be the supplied policy."""
    custom = PlanModelPolicy(default_tier=ModelTier.HEAVY)
    mode = PlanMode(provider=_RecordingProvider(), model_policy=custom)  # type: ignore[arg-type]
    # ``_deps`` is private but the test is the consumer-of-record for
    # the wiring contract; reaching in is fine inside the test module.
    assert mode._deps.model_policy is custom


@pytest.mark.asyncio
async def test_planmode_default_constructor_uses_standard_policy() -> None:
    mode = PlanMode(provider=_RecordingProvider())  # type: ignore[arg-type]
    assert mode._deps.model_policy is STANDARD_PLAN_POLICY


# Suppress unused-attribute warning on ClassVar import in environments
# where ruff doesn't recognize it as TYPE_CHECKING-style usage.
_: ClassVar[None] | None = None
