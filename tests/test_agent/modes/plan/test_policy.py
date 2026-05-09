"""Tests for :mod:`molexp.agent.modes.plan.policy`.

Covers acceptance criteria ac-001..ac-010 for sub-spec
``planmode-workspace-pipeline-04-plan-model-policy``. The end-to-end
"custom policy is observed by Provider.invoke" coverage moved to
``test_pipeline_core.py`` once sub-spec 05 rewrote the pipeline; the
unit-level tests live here.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest
from pydantic import ValidationError

from molexp.agent.modes.plan import (
    PLAN_NODE_NAMES,
    STANDARD_PLAN_POLICY,
    PlanModelPolicy,
)
from molexp.agent.modes.plan.protocols import ModelTier, PlanDeps

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


def test_plan_node_names_covers_legacy_pipeline_classes() -> None:
    """The legacy 14-task ids stay in the bridging set so an
    operator can still author a policy referring to them."""
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
def test_standard_policy_preserves_legacy_tier_table(node: str, expected_tier: ModelTier) -> None:
    assert STANDARD_PLAN_POLICY.tier_for(node) is expected_tier


# ── tasks.py source-level invariants (ac-005 + ac-006) ─────────────────────


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
    assert "TIER" not in assignments


def test_tasks_py_invoke_llm_uses_model_policy() -> None:
    """``invoke_llm`` must resolve tier via ``ctx.deps.policy.tier_for(...)``."""
    source = _TASKS_PY_PATH.read_text()
    assert "ctx.deps.policy.tier_for(" in source
    assert "ModelTier.CHEAP" not in source
    assert "ModelTier.DEFAULT" not in source
    assert "ModelTier.HEAVY" not in source


# ── PlanDeps shape (ac-008) ────────────────────────────────────────────────


def test_plan_deps_required_fields() -> None:
    fields = PlanDeps.__dataclass_fields__
    assert {"provider", "policy", "workspace_handle"}.issubset(fields.keys())


def test_plan_deps_drops_legacy_service_fields() -> None:
    fields = PlanDeps.__dataclass_fields__
    legacy = {"gate_policy", "repair_policy", "store", "artifact_writer", "model_policy"}
    assert legacy.isdisjoint(fields.keys())


# ── Public re-exports (ac-009) ─────────────────────────────────────────────


def test_planmode_modes_plan_re_exports_policy_names() -> None:
    import molexp.agent.modes.plan as plan_pkg

    assert plan_pkg.PlanModelPolicy is PlanModelPolicy
    assert plan_pkg.STANDARD_PLAN_POLICY is STANDARD_PLAN_POLICY
    assert "PlanModelPolicy" in plan_pkg.__all__
    assert "STANDARD_PLAN_POLICY" in plan_pkg.__all__
