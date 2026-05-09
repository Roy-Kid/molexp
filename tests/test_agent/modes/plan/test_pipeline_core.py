"""Integration + per-node tests for the materialize-to-workspace pipeline.

Covers acceptance criteria ac-001..ac-012 for sub-spec
``planmode-workspace-pipeline-05-pipeline-rewrite-core``.
"""

from __future__ import annotations

import ast
import importlib
from pathlib import Path

import pytest
import yaml

from molexp.agent.mode import AgentRunResult
from molexp.agent.modes.plan import (
    PLAN_WORKFLOW,
    PlanMode,
    PlanModelPolicy,
    PlanWorkspaceHandle,
    SkeletonCompileError,
)
from molexp.agent.modes.plan._pipeline import build_plan_workflow
from molexp.agent.modes.plan.protocols import ModelTier, PlanDeps
from molexp.agent.modes.plan.schemas import (
    DigestResult,
    IngestReportResult,
    PlanBriefResult,
    SkeletonResult,
    TaskIRResult,
    WorkflowIRResult,
)
from molexp.agent.modes.plan.tasks import (
    CompileTaskIR,
    CompileWorkflowIR,
    DraftImplementationPlan,
    DraftReportDigest,
    GenerateWorkflowSkeleton,
    IngestReport,
    PlanLLMTask,
    PlanTask,
)
from molexp.agent.session import AgentSession
from molexp.workflow import Workflow

from .conftest import FakeProvider

# ── PLAN_WORKFLOW shape (ac-007) ───────────────────────────────────────────


def test_plan_workflow_is_workflow_with_ten_named_nodes() -> None:
    """Sub-spec 06 grew the pipeline from 6 to 10 nodes."""
    assert isinstance(PLAN_WORKFLOW, Workflow)
    assert PLAN_WORKFLOW._entries == ("IngestReport",)
    assert {t.name for t in PLAN_WORKFLOW._tasks} == {
        "IngestReport",
        "DraftReportDigest",
        "DraftImplementationPlan",
        "CompileWorkflowIR",
        "CompileTaskIR",
        "GenerateWorkflowSkeleton",
        "GenerateTaskTests",
        "GenerateTaskImplementations",
        "ValidateWorkspace",
        "HumanReview",
    }


def test_build_plan_workflow_returns_independent_instance() -> None:
    """Re-building yields a fresh :class:`Workflow` (no shared mutable state)."""
    wf2 = build_plan_workflow()
    assert wf2 is not PLAN_WORKFLOW
    assert {t.name for t in wf2._tasks} == {t.name for t in PLAN_WORKFLOW._tasks}


# ── Schema surface ─────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "removed",
    [
        "IntakeSpec",
        "ProtocolDraft",
        "ExecutableWorkflowDraft",
        "Decomposition",
        "MethodSpec",
        "GoalSpec",
        "ContextSpec",
        "PlanPreview",
        "CodegenOutput",
        "GeneratedTaskSpec",
        "CompileReport",
        "DryRunReport",
        "RepairReport",
        "PlanPatch",
        "PlanSpec",
        "ApprovedPlan",
    ],
)
def test_subsumed_schemas_are_no_longer_importable(removed: str) -> None:
    """ac-001 — removed names raise ``ImportError`` from the schemas module."""
    import importlib

    mod = importlib.import_module("molexp.agent.modes.plan.schemas")
    assert not hasattr(mod, removed), (
        f"schemas.py still exposes the removed name {removed!r}; should be gone"
    )


def test_approval_decision_remains_importable() -> None:
    from molexp.agent.modes.plan.schemas import ApprovalDecision

    decision = ApprovalDecision(approved=True, reason="ok")
    assert decision.approved is True


@pytest.mark.parametrize(
    "schema_name",
    [
        "ReportDigest",
        "PlanBrief",
        "TaskIRBrief",
        "IngestReportResult",
        "DigestResult",
        "PlanBriefResult",
        "WorkflowIRResult",
        "TaskIRResult",
        "SkeletonResult",
    ],
)
def test_new_schemas_importable_and_frozen(schema_name: str) -> None:
    """ac-002 — every new schema imports and is frozen pydantic."""
    mod = importlib.import_module("molexp.agent.modes.plan.schemas")
    cls = getattr(mod, schema_name)
    assert cls.model_config.get("frozen") is True


# ── PlanDeps shape (ac-003) ────────────────────────────────────────────────


@pytest.mark.parametrize(
    "name",
    [
        # Sub-spec 06 reintroduced ``GatePolicy`` + ``AutoApproveGatePolicy``
        # for the terminal HumanReview node — they are NOT in this list.
        "IdentityRepairPolicy",
        "InMemoryPlanStore",
        "NoOpArtifactWriter",
        "RepairPolicy",
        "PlanStore",
        "ArtifactWriter",
    ],
)
def test_legacy_protocol_names_no_longer_importable(name: str) -> None:
    mod = importlib.import_module("molexp.agent.modes.plan.protocols")
    assert not hasattr(mod, name), f"protocols.py still exposes legacy name {name!r}"


# ── Task class surface (ac-004) ────────────────────────────────────────────


_NEW_TASK_CLASSES: list[type] = [
    IngestReport,
    DraftReportDigest,
    DraftImplementationPlan,
    CompileWorkflowIR,
    CompileTaskIR,
    GenerateWorkflowSkeleton,
]
_LLM_TASK_CLASSES = {
    DraftReportDigest,
    DraftImplementationPlan,
    CompileWorkflowIR,
    CompileTaskIR,
}


@pytest.mark.parametrize("task_cls", _NEW_TASK_CLASSES)
def test_new_task_class_subclasses_plan_task(task_cls: type) -> None:
    assert issubclass(task_cls, PlanTask)


@pytest.mark.parametrize("task_cls", sorted(_LLM_TASK_CLASSES, key=lambda c: c.__name__))
def test_llm_task_classes_subclass_plan_llm_task(task_cls: type) -> None:
    assert issubclass(task_cls, PlanLLMTask)


@pytest.mark.parametrize(
    "removed",
    [
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
    ],
)
def test_legacy_task_classes_no_longer_importable(removed: str) -> None:
    mod = importlib.import_module("molexp.agent.modes.plan.tasks")
    assert not hasattr(mod, removed)


# ── tasks.py source-level guards (ac-005, ac-006) ──────────────────────────


_TASKS_PY_PATH = (
    Path(__file__).parent.parent.parent.parent.parent
    / "src"
    / "molexp"
    / "agent"
    / "modes"
    / "plan"
    / "tasks.py"
)


def test_tasks_py_has_no_path_write_text_calls() -> None:
    """ac-006 — every artifact write goes through the workspace handle."""
    source = _TASKS_PY_PATH.read_text()
    tree = ast.parse(source)
    forbidden_method_names = {"write_text", "write_bytes"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and node.attr in forbidden_method_names:
            # The forbidden pattern: ``something.write_text(...)``.
            pytest.fail(
                f"tasks.py calls .{node.attr}(...) at line {node.lineno}; "
                "use ctx.deps.workspace_handle helpers instead."
            )
    # Also: no direct ``open(..., 'w')`` call.
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "open"
        ):
            pytest.fail(
                f"tasks.py calls open(...) at line {node.lineno}; "
                "use ctx.deps.workspace_handle helpers instead."
            )


# ── Per-node tests (drives execute() directly) ─────────────────────────────


@pytest.mark.asyncio
async def test_ingest_report_writes_original_md(
    workspace_handle: PlanWorkspaceHandle, fake_provider: FakeProvider
) -> None:
    deps = PlanDeps(
        provider=fake_provider,
        policy=PlanModelPolicy(),
        workspace_handle=workspace_handle,
    )
    result = await PLAN_WORKFLOW.execute(
        config={"user_input": "user-supplied report text"}, deps=deps
    )
    ingest_out = result.outputs["IngestReport"]
    assert isinstance(ingest_out, IngestReportResult)
    assert ingest_out.report_path == workspace_handle.report_dir() / "original.md"
    assert ingest_out.report_path.read_text() == "user-supplied report text"
    assert len(ingest_out.report_hash) == 64  # sha256 hex


@pytest.mark.asyncio
async def test_ingest_report_rejects_empty_input(
    workspace_handle: PlanWorkspaceHandle, fake_provider: FakeProvider
) -> None:
    """The workflow runtime catches the ValueError IngestReport raises and
    returns ``WorkflowResult(status='failed')`` rather than propagating
    (see ``runtime.py``); the failure is what we assert on."""
    deps = PlanDeps(
        provider=fake_provider,
        policy=PlanModelPolicy(),
        workspace_handle=workspace_handle,
    )
    result = await PLAN_WORKFLOW.execute(config={"user_input": ""}, deps=deps)
    assert result.status == "failed"
    # No downstream task should have produced output.
    assert "DraftReportDigest" not in result.outputs


# ── End-to-end pipeline (ac-008) ───────────────────────────────────────────


@pytest.mark.asyncio
async def test_pipeline_end_to_end_creates_all_artifacts(
    workspace_handle: PlanWorkspaceHandle, fake_provider: FakeProvider
) -> None:
    """ac-008 — the load-bearing integration test."""
    deps = PlanDeps(
        provider=fake_provider,
        policy=PlanModelPolicy(),
        workspace_handle=workspace_handle,
    )
    result = await PLAN_WORKFLOW.execute(
        config={"user_input": "Investigate Suzuki coupling at varying temperatures."},
        deps=deps,
    )

    skeleton_out = result.outputs["GenerateWorkflowSkeleton"]
    assert isinstance(skeleton_out, SkeletonResult)

    root = workspace_handle.root()
    expected = [
        root / "report" / "original.md",
        root / "report" / "digest.md",
        root / "plan" / "implementation_plan.md",
        root / "ir" / "workflow.yaml",
        root / "src" / "experiment" / "__init__.py",
        root / "src" / "experiment" / "workflow.py",
        root / "src" / "experiment" / "tasks" / "__init__.py",
    ]
    for path in expected:
        assert path.exists(), f"missing artifact: {path}"
        assert path.stat().st_size > 0, f"empty artifact: {path}"

    # ir/tasks/*.yaml — at least one (canned contract has 3).
    tasks_yaml = list((root / "ir" / "tasks").glob("*.yaml"))
    assert len(tasks_yaml) >= 1
    # And every per-task YAML round-trips via safe_load.
    for tyaml in tasks_yaml:
        assert isinstance(yaml.safe_load(tyaml.read_text()), dict)

    # Per-node Result types are well-formed.
    assert isinstance(result.outputs["IngestReport"], IngestReportResult)
    assert isinstance(result.outputs["DraftReportDigest"], DigestResult)
    assert isinstance(result.outputs["DraftImplementationPlan"], PlanBriefResult)
    assert isinstance(result.outputs["CompileWorkflowIR"], WorkflowIRResult)
    assert isinstance(result.outputs["CompileTaskIR"], TaskIRResult)


# ── Skeleton compile guard (ac-009) ────────────────────────────────────────


@pytest.mark.asyncio
async def test_generate_workflow_skeleton_raises_skeleton_compile_error(
    workspace_handle: PlanWorkspaceHandle, fake_provider: FakeProvider
) -> None:
    """Driving the task directly (outside the workflow runtime) lets the
    raised :class:`SkeletonCompileError` propagate — the workflow
    runtime would otherwise wrap it as ``WorkflowResult(status='failed')``
    (see ``runtime.py``)."""
    from molexp.agent.modes.plan import tasks as tasks_module
    from molexp.workflow.context import TaskContext

    # First run the pipeline up through CompileTaskIR so the upstream
    # *Result objects exist for the skeleton task to consume.
    deps = PlanDeps(
        provider=fake_provider,
        policy=PlanModelPolicy(),
        workspace_handle=workspace_handle,
    )
    ir_contract = await fake_provider.invoke(
        tier=ModelTier.DEFAULT,
        system="",
        user="",
        schema=__import__(
            "molexp.agent.modes.plan.schemas", fromlist=["WorkflowContract"]
        ).WorkflowContract,
        node_id="CompileWorkflowIR",
    )
    ir_path = workspace_handle.ir_dir() / "workflow.yaml"
    workspace_handle.tasks_ir_dir()
    workflow_ir = WorkflowIRResult(
        workflow_yaml_path=ir_path,
        contract=ir_contract,  # type: ignore[arg-type]
    )
    task_ir = TaskIRResult(task_ir_paths=(), briefs=())

    # Patch the renderer to emit deliberately-broken source.
    def _broken_render(_contract: object) -> str:
        return "def def def\n"

    original = tasks_module._render_workflow_module
    tasks_module._render_workflow_module = _broken_render  # type: ignore[assignment]
    try:
        ctx = TaskContext(
            state=None,
            deps=deps,
            inputs={
                "CompileWorkflowIR": workflow_ir,
                "CompileTaskIR": task_ir,
            },
            config={},
        )
        with pytest.raises(SkeletonCompileError) as exc_info:
            await GenerateWorkflowSkeleton().execute(ctx)
    finally:
        tasks_module._render_workflow_module = original  # type: ignore[assignment]
    assert isinstance(exc_info.value.__cause__, SyntaxError)


# ── PlanMode.run mode_state shape (ac-010) ─────────────────────────────────


@pytest.mark.asyncio
async def test_plan_mode_run_exposes_workspace_path_and_back_compat_shim(
    workspace_handle: PlanWorkspaceHandle, fake_provider: FakeProvider
) -> None:
    mode = PlanMode(
        workspace_handle=workspace_handle,
        provider=fake_provider,  # type: ignore[arg-type]
    )
    session = AgentSession()
    result = await mode.run(
        harness=None,  # type: ignore[arg-type] — runner-supplied; ignored
        session=session,
        user_input="Investigate Suzuki coupling.",
    )
    assert isinstance(result, AgentRunResult)
    assert result.mode_state is not None
    assert result.mode_state["workspace_path"] == workspace_handle.root()
    plan_compat = result.mode_state["plan"]
    # Sub-spec 06 added ``handoff`` to the back-compat shim; ``approved``
    # is now driven by the gate's decision (auto-approve here ⇒ True).
    assert set(plan_compat.keys()) == {"intake", "design", "approved", "iterations", "handoff"}
    assert plan_compat["approved"] is True
    assert plan_compat["iterations"] is None
    assert isinstance(plan_compat["handoff"], dict)
    # Session shim preserved.
    assert session.mode_state["plan"] == plan_compat


# ── Custom policy injection ────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_custom_policy_observed_by_provider_invoke(
    workspace_handle: PlanWorkspaceHandle,
) -> None:
    """A non-default :class:`PlanModelPolicy` lands on every provider call."""
    provider = FakeProvider()
    custom = PlanModelPolicy(default_tier=ModelTier.CHEAP)
    deps = PlanDeps(
        provider=provider,
        policy=custom,
        workspace_handle=workspace_handle,
    )
    await PLAN_WORKFLOW.execute(config={"user_input": "report"}, deps=deps)
    assert provider.calls, "fake provider should have been invoked"
    for node_id, tier, _schema_name in provider.calls:
        assert tier is ModelTier.CHEAP, (
            f"node {node_id!r} received tier={tier} under default-CHEAP policy"
        )


# ── ApprovalDecision still importable (sanity) ─────────────────────────────


def test_approval_decision_round_trips() -> None:
    from molexp.agent.modes.plan.schemas import ApprovalDecision

    original = ApprovalDecision(approved=False, reason="missing data")
    rebuilt = ApprovalDecision.model_validate_json(original.model_dump_json())
    assert rebuilt == original
