"""``PlanMode`` offline verification — the single 9-step pipeline.

Deterministic, network-free: a ``StubAgentGateway`` returns canned valid
outputs for every LLM agent, so ``PlanMode`` runs its full sequence on a tmp
``workspace.Run``. The canned workflow source compiles AND executes under
``WorkflowRuntime`` (``outputs["summarize"]["total"] == 6``) and the canned
test passes in the materialized layout, so the default ``LocalExecutor``
spawns genuine pytest + compile subprocesses for step 7 (and the real
workflow in the ``execute=True`` tail). No LLM, no network.

This file also covers what used to be ``test_run_mode.py``: the real
execution + final/audit reports now live in PlanMode's opt-in ``--execute``
tail (RunMode is retired).
"""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from pathlib import Path

import pytest

from molexp.harness import Mode, ModeResult, PlanMode
from molexp.harness.gateways.stub import StubAgentGateway
from molexp.harness.schemas import ExecutionResult, WorkflowSource
from molexp.harness.store.file_artifact_store import FileArtifactStore
from molexp.harness.store.sqlite_lineage_store import SQLiteArtifactLineageStore
from molexp.workspace import Workspace

_DRAFT = "Simulate NEMD ionic mobility"

# Multi-file workflow (per-task modules + assembly). The assembly imports the
# two task modules and registers them; compiles AND executes under
# WorkflowRuntime (summarize binds make_data's {"values": ...} → total == 6).
_ASSEMBLY = """\
from molexp.workflow import WorkflowCompiler

from workflow.make_data import make_data
from workflow.summarize import summarize


def build_workflow() -> WorkflowCompiler:
    wf = WorkflowCompiler(name="canned_run")
    wf.task(make_data)
    wf.task(depends_on=["make_data"])(summarize)
    return wf
"""
_TASK_MAKE_DATA = """\
async def make_data() -> dict:
    return {"values": [1, 2, 3]}
"""
_TASK_SUMMARIZE = """\
async def summarize(values) -> dict:
    return {"total": sum(values)}
"""
# One test file per task; each imports the assembled `workflow` package.
_TEST_MAKE_DATA = """\
from workflow import build_workflow


def test_make_data_compiles() -> None:
    assert build_workflow().compile() is not None
"""
_TEST_SUMMARIZE = """\
from workflow import build_workflow


def test_summarize_runs() -> None:
    assert build_workflow().compile() is not None
"""

_EXPERIMENT_REPORT = {
    "title": "Water NEMD",
    "objective": "Measure ionic mobility",
    "system_description": "SPC/E water box under an applied field",
    "experimental_design": "Apply field; record current",
}
_EXPERIMENT_SPEC = {
    "id": "spec-water",
    "experiment_report_id": "rep-water",
    "title": "Water NEMD",
    "objective": "Measure ionic mobility",
    "variables": [],
    "controlled_conditions": [],
    "resolved_questions": [],
    "assumptions": [],
}
_WORKFLOW_IR = {
    "id": "wf-water",
    "name": "water_nemd",
    "objective": "Compute mobility",
    "inputs": {},
    "tasks": [
        {
            "id": "make_data",
            "name": "Make data",
            "purpose": "Generate the sample values",
            "task_type": "simulation",
            "inputs": {},
            "outputs": {"values": "list"},
        },
        {
            "id": "summarize",
            "name": "Summarize",
            "purpose": "Reduce to a scalar",
            "task_type": "analysis",
            "inputs": {},
            "outputs": {"total": "scalar"},
        },
    ],
    "edges": [{"source_task_id": "make_data", "target_task_id": "summarize"}],
    "expected_outputs": [],
}
_BOUND_WORKFLOW = {
    "id": "bw-water",
    "workflow_ir_id": "wf-water",
    "tasks": [
        {
            "id": "b-make_data",
            "ir_task_id": "make_data",
            "capability_id": "stdlib.make_data",
            "package": "python-stdlib",
            "callable": "workflow.make_data.make_data",
            "parameters": {},
            "inputs": {},
            "outputs": {"values": "list"},
        },
        {
            "id": "b-summarize",
            "ir_task_id": "summarize",
            "capability_id": "stdlib.summarize",
            "package": "python-stdlib",
            "callable": "workflow.summarize.summarize",
            "parameters": {},
            "inputs": {},
            "outputs": {"total": "scalar"},
        },
    ],
    "edges": [{"source_task_id": "b-make_data", "target_task_id": "b-summarize"}],
    "execution_backend": "local",
    "environment": {},
    "resource_policy": {"backend": "local", "max_runtime_s": 3600, "denied_paths": ["/", "~/.ssh"]},
}
_WORKFLOW_SOURCE = {
    "source": _ASSEMBLY,
    "module_name": "workflow",
    "bound_workflow_id": "bw-water",
    "symbols": ["WorkflowCompiler"],
    "files": [
        {"path": "workflow/__init__.py", "source": _ASSEMBLY},
        {"path": "workflow/make_data.py", "source": _TASK_MAKE_DATA},
        {"path": "workflow/summarize.py", "source": _TASK_SUMMARIZE},
    ],
}
_INPUT_SET = {
    "id": "is-water",
    "experiment_spec_id": "spec-water",
    "title": "single-cell sweep",
    "sweep_axes": [],
    "strategy": "grid",
    "total_runs": 1,
}
_TEST_SPEC = {
    "id": "tsb-water",
    "bound_workflow_id": "bw-water",
    "specs": [
        {
            "id": "ts-make_data",
            "name": "make_data compiles",
            "kind": "unit_test",
            "target_task_id": "make_data",
            "description": "The make_data task is present in the assembled workflow.",
        },
        {
            "id": "ts-summarize",
            "name": "summarize runs",
            "kind": "unit_test",
            "target_task_id": "summarize",
            "description": "The summarize task is present in the assembled workflow.",
        },
    ],
}
_TEST_SOURCE = {
    "source": _TEST_MAKE_DATA,
    "module_name": "test_make_data",
    "test_spec_id": "tsb-water",
    "bound_workflow_id": "bw-water",
    "symbols": ["build_workflow"],
    "files": [
        {"path": "tests/test_make_data.py", "source": _TEST_MAKE_DATA},
        {"path": "tests/test_summarize.py", "source": _TEST_SUMMARIZE},
    ],
}
# Structurally valid (covers both tasks) but red at runtime → blocks at ExecuteTests.
_FAILING_TEST_SOURCE = {
    **_TEST_SOURCE,
    "files": [
        {"path": "tests/test_make_data.py", "source": _TEST_MAKE_DATA},
        {
            "path": "tests/test_summarize.py",
            "source": "def test_summarize_runs() -> None:\n    assert False\n",
        },
    ],
}
_FINAL_REPORT = {
    "title": "CannedWaterNemdFinalReport",
    "objective": "Measure ionic mobility from real execution outputs.",
    "methods_summary": "Two-task canned workflow executed by the harness driver.",
    "test_summary": "Generated unit test compiled the workflow and passed.",
    "execution_summary": "Driver subprocess exited 0; outputs.json collected.",
    "results": "summarize reported total == 6 from values [1, 2, 3].",
    "conclusions": "PlanMode --execute carries plan artifacts to real execution.",
    "limitations": ["single canned seed"],
    "next_steps": ["sweep field strengths"],
}


def _make_run(tmp_path: Path):
    ws = Workspace(tmp_path / "lab", name="plan-lab")
    ws.materialize()
    return ws.add_project("demo").add_experiment("nemd").add_run(params={})


def _fixture_gateway(run, *, test_source: Mapping[str, object] | None = None) -> StubAgentGateway:
    store = FileArtifactStore(root=run.run_dir / "artifacts")
    gw = StubAgentGateway(store)
    gw.register("experiment_report_writer", _EXPERIMENT_REPORT, output_kind="experiment_report")
    gw.register("experiment_spec_generator", _EXPERIMENT_SPEC, output_kind="experiment_spec")
    gw.register("workflow_ir_extractor", _WORKFLOW_IR, output_kind="workflow_ir")
    gw.register("bound_workflow_binder", _BOUND_WORKFLOW, output_kind="bound_workflow")
    gw.register("workflow_source_writer", _WORKFLOW_SOURCE, output_kind="workflow_source")
    gw.register(
        "plan_reviewer",
        {"passed": True, "findings": [], "summary": "faithful"},
        output_kind="plan_review",
    )
    gw.register("test_spec_writer", _TEST_SPEC, output_kind="test_spec")
    gw.register("test_code_writer", dict(test_source or _TEST_SOURCE), output_kind="test_source")
    gw.register("input_set_generator", _INPUT_SET, output_kind="input_set")
    gw.register("final_report_writer", _FINAL_REPORT, output_kind="final_report")
    return gw


# ───────────────────────────────────────────────────────── shape


class TestPlanModeShape:
    def test_plan_mode_is_a_mode_subclass_exported(self) -> None:
        import molexp.harness as harness

        assert issubclass(PlanMode, Mode)
        assert "PlanMode" in harness.__all__
        assert not hasattr(harness, "RunMode"), "RunMode is retired"

    def test_plan_mode_declares_the_nine_step_sequence(self) -> None:
        names = [s.name for s in PlanMode().stages("draft")]
        assert names == [
            "save_user_plan",
            "generate_experiment_report",
            "generate_experiment_spec",
            "approve_experiment_spec",  # human approves the spec BEFORE the IR is built
            "resolve_capabilities",
            "extract_workflow_ir",
            "bind_molcrafts_tasks",
            "generate_workflow_source",
            "generate_test_spec",
            "validate_test_spec",
            "generate_test_code",
            "validate_test_source",
            "generate_input_set",
            "materialize_execution",
            "execute_tests",
            "compile_workflow",
            "approve_plan",
            "generate_execution_report",
        ]

    def test_execute_tail_appends_real_execution_stages(self) -> None:
        plan = [s.name for s in PlanMode().stages("draft")]
        full = [s.name for s in PlanMode(execute=True).stages("draft")]
        assert full[len(plan) :] == [
            "execute_workflow",
            "generate_final_report",
            "approve_execution",
            "generate_audit_report",
        ]


# ───────────────────────────────────────────────── offline run (plan-only)


class TestPlanModeRun:
    @pytest.mark.integration
    def test_plan_mode_runs_all_steps_offline(self, tmp_path: Path) -> None:
        run = _make_run(tmp_path)
        gateway = _fixture_gateway(run)

        result = asyncio.run(PlanMode().run(run=run, user_input=_DRAFT, gateway=gateway))

        assert isinstance(result, ModeResult)
        assert result.mode_name == "plan"
        kinds = {a.kind for a in result.stage_artifacts}
        assert kinds >= {
            "user_plan",
            "experiment_report",
            "experiment_spec",
            "capability_catalog",
            "workflow_ir",
            "bound_workflow",
            "workflow_source",
            "test_spec",
            "test_source",
            "input_set",
            "execution_result",  # the compile-only dry run
            "analysis_result",  # step-8 gate
            "execution_report",  # step 9
        }

    @pytest.mark.integration
    def test_compile_dry_run_is_a_compile_not_a_real_run(self, tmp_path: Path) -> None:
        run = _make_run(tmp_path)
        gateway = _fixture_gateway(run)
        store = FileArtifactStore(root=run.run_dir / "artifacts")

        asyncio.run(PlanMode().run(run=run, user_input=_DRAFT, gateway=gateway))

        # The only execution_result in plan-only mode is the compile dry run.
        exec_ref = store.latest_by_kind("execution_result")
        execution = ExecutionResult.model_validate_json(store.get(exec_ref.id))
        assert execution.status == "succeeded"
        assert execution.metadata.get("mode") == "compile"
        assert execution.outputs == {}  # no science ran

    @pytest.mark.integration
    def test_workflow_source_lineage_reaches_user_plan(self, tmp_path: Path) -> None:
        run = _make_run(tmp_path)
        gateway = _fixture_gateway(run)
        store = FileArtifactStore(root=run.run_dir / "artifacts")

        result = asyncio.run(PlanMode().run(run=run, user_input=_DRAFT, gateway=gateway))
        src_ref = next(a for a in result.stage_artifacts if a.kind == "workflow_source")
        user_plan_ref = next(a for a in result.stage_artifacts if a.kind == "user_plan")

        provenance = SQLiteArtifactLineageStore(
            path=run.run_dir / "harness.sqlite", artifact_store=store
        )
        ancestors = {ref.id for ref in provenance.trace_backward(src_ref.id)}
        assert user_plan_ref.id in ancestors

    @pytest.mark.integration
    def test_recoverable_workflow_source(self, tmp_path: Path) -> None:
        run = _make_run(tmp_path)
        gateway = _fixture_gateway(run)
        store = FileArtifactStore(root=run.run_dir / "artifacts")

        result = asyncio.run(PlanMode().run(run=run, user_input=_DRAFT, gateway=gateway))
        src_refs = [a for a in result.stage_artifacts if a.kind == "workflow_source"]
        assert len(src_refs) == 1
        ws_obj = WorkflowSource.model_validate_json(store.get(src_refs[0].id))
        assert "build_workflow" in ws_obj.source


# ───────────────────────────────────────────── --execute tail (real run)


class TestPlanModeExecute:
    @pytest.mark.integration
    def test_execute_tail_runs_the_real_workflow(self, tmp_path: Path) -> None:
        run = _make_run(tmp_path)
        gateway = _fixture_gateway(run)
        store = FileArtifactStore(root=run.run_dir / "artifacts")

        result = asyncio.run(
            PlanMode(execute=True).run(run=run, user_input=_DRAFT, gateway=gateway)
        )

        kinds = {a.kind for a in result.stage_artifacts}
        assert kinds >= {"final_report", "audit_report"}

        # The driver really ran the canned workflow: outputs.json round-trips.
        # The execute tail's execution_result (no compile mode) is the latest.
        execs = [a for a in result.stage_artifacts if a.kind == "execution_result"]
        real = ExecutionResult.model_validate_json(store.get(execs[-1].id))
        assert real.status == "succeeded"
        assert real.outputs["summarize"]["total"] == 6

    @pytest.mark.integration
    def test_failing_generated_tests_block_the_plan(self, tmp_path: Path) -> None:
        run = _make_run(tmp_path)
        gateway = _fixture_gateway(run, test_source=_FAILING_TEST_SOURCE)
        store = FileArtifactStore(root=run.run_dir / "artifacts")

        from molexp.harness import StagePersistedFailureError

        with pytest.raises(StagePersistedFailureError):
            asyncio.run(PlanMode(execute=True).run(run=run, user_input=_DRAFT, gateway=gateway))

        # In-function import: a module-level `TestResult` would be collected by
        # pytest as a test class (house pattern for Test*-named schemas).
        from molexp.harness import TestResult

        test_ref = store.latest_by_kind("test_result")
        assert test_ref is not None
        assert TestResult.model_validate_json(store.get(test_ref.id)).status == "failed"
        # The plan never reached the execution report or the real run.
        assert store.latest_by_kind("execution_report") is None


# ───────────────────────────────────────────── review gate + ledger resume


class TestPlanModeGateAndResume:
    @pytest.mark.integration
    def test_rejecting_approver_aborts_before_the_workflow_is_built(self, tmp_path: Path) -> None:
        """A rejecting approver stops at the SPEC gate — before capabilities/IR.

        The user's law: the spec is approved BEFORE it is fed to the LLM to
        build the workflow. So a rejection leaves the spec on disk but produces
        no capability_catalog / workflow_ir / workflow_source.
        """
        from datetime import UTC, datetime

        from molexp.harness import StageExecutionError
        from molexp.harness.schemas import ApprovalDecision, ApprovalRequest

        run = _make_run(tmp_path)
        gateway = _fixture_gateway(run)
        store = FileArtifactStore(root=run.run_dir / "artifacts")

        async def reject(request: ApprovalRequest) -> ApprovalDecision:
            return ApprovalDecision(
                request_id=request.id,
                granted=False,
                decided_by="test",
                decided_at=datetime.now(tz=UTC),
                reason="nope",
            )

        with pytest.raises(StageExecutionError):
            asyncio.run(PlanMode(approver=reject).run(run=run, user_input=_DRAFT, gateway=gateway))

        # The spec exists, but the rejection stopped before any workflow build.
        assert store.latest_by_kind("experiment_spec") is not None
        assert store.latest_by_kind("capability_catalog") is None
        assert store.latest_by_kind("workflow_ir") is None
        assert store.latest_by_kind("workflow_source") is None

    @pytest.mark.integration
    def test_second_run_reuses_ledger_with_unregistered_gateway(self, tmp_path: Path) -> None:
        run = _make_run(tmp_path)
        gateway = _fixture_gateway(run)

        first = asyncio.run(PlanMode().run(run=run, user_input=_DRAFT, gateway=gateway))

        # Nothing registered: any re-run of an LLM stage body would raise, so
        # completing proves every stage was skipped via the completion ledger.
        empty = StubAgentGateway(FileArtifactStore(root=run.run_dir / "artifacts"))
        second = asyncio.run(PlanMode().run(run=run, user_input=_DRAFT, gateway=empty))

        assert [a.id for a in second.stage_artifacts] == [a.id for a in first.stage_artifacts]


# ─────────────────────────────────────────────────── example guards


def test_flagship_example_imports_without_network_or_key(monkeypatch) -> None:
    import os

    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    import examples.harness.experiment_pipeline as pipeline

    assert callable(pipeline.main)
    assert os.environ.get("DEEPSEEK_API_KEY") is None


def test_examples_dir_has_no_pytest_tests() -> None:
    examples = Path(__file__).resolve().parents[2] / "examples"
    assert not list(examples.rglob("test_*.py"))
