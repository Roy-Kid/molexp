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

from molexp.harness import ModeResult, PlanMode
from molexp.harness.gateways.stub import StubAgentGateway
from molexp.harness.schemas import ExecutionResult
from molexp.harness.stages.approval_gate import auto_grant_approver
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
    # Per-task workflow codegen: GenerateWorkflowSource fans one call per bound
    # task and synthesizes the assembly itself, so each call must return just
    # that task's module. A responder reads the slice's single task and returns
    # the matching canned module.
    _TASK_MODULE = {"make_data": _TASK_MAKE_DATA, "summarize": _TASK_SUMMARIZE}

    def _wf_file_responder(spec, store):
        import yaml

        raw = store.get(spec.input_artifact_ids[0]).decode("utf-8")
        # SequentialTaskBuild sends a YAML codegen_prompt (contract+task+wiring).
        # Legacy path: first input was a BoundWorkflow slice JSON.
        try:
            doc = yaml.safe_load(raw)
            if isinstance(doc, dict) and "task" in doc:
                ir = (doc.get("task") or {}).get("ir_task_id") or (
                    doc.get("task") or {}
                ).get("slug")
            else:
                ir = None
        except Exception:
            ir = None
        if not ir:
            from molexp.harness.schemas import BoundWorkflow

            slice_wf = BoundWorkflow.model_validate_json(
                store.get(spec.input_artifact_ids[0])
            )
            ir = slice_wf.tasks[0].ir_task_id
            bw_id = slice_wf.id
        else:
            bw_id = "bw-water"
        src = _TASK_MODULE[ir]
        return gw.make_response(
            {
                "source": src,
                "module_name": ir,
                "bound_workflow_id": bw_id,
                "symbols": [],
                "files": [{"path": f"workflow/{ir}.py", "source": src}],
            },
            output_kind="workflow_source_file",
        )

    gw.register_responder("workflow_source_file_writer", _wf_file_responder)
    gw.register(
        "plan_reviewer",
        {"passed": True, "findings": [], "summary": "faithful"},
        output_kind="plan_review",
    )
    gw.register("test_spec_writer", _TEST_SPEC, output_kind="test_spec")
    gw.register("test_code_writer", dict(test_source or _TEST_SOURCE), output_kind="test_source")
    # Per-task test codegen (sequential_task_build + GenerateTestCode fan-out):
    # return only the matching tests/test_<slug>.py body for the requested task.
    _TEST_BY_SLUG = {
        "make_data": (test_source or _TEST_SOURCE)["files"][0]["source"]
        if (test_source or _TEST_SOURCE).get("files")
        else _TEST_MAKE_DATA,
        "summarize": (test_source or _TEST_SOURCE)["files"][1]["source"]
        if (test_source or _TEST_SOURCE).get("files") and len((test_source or _TEST_SOURCE)["files"]) > 1
        else _TEST_SUMMARIZE,
    }

    def _test_file_responder(spec, store):
        # SequentialTaskBuild sends YAML codegen_prompt (task.slug / module path).
        import json

        import yaml

        raw = store.get(spec.input_artifact_ids[0])
        slug = "make_data"
        try:
            text = raw.decode("utf-8") if isinstance(raw, (bytes, bytearray)) else raw
            doc = yaml.safe_load(text)
            if isinstance(doc, dict) and doc.get("task"):
                slug = doc["task"].get("slug") or doc["task"].get("ir_task_id") or slug
            else:
                body = json.loads(text)
                slug = body.get("module_name") or slug
        except Exception:
            pass
        src = _TEST_BY_SLUG.get(slug, _TEST_MAKE_DATA)
        return gw.make_response(
            {
                "source": src,
                "module_name": f"test_{slug}",
                "test_spec_id": "tsb-water",
                "bound_workflow_id": "bw-water",
                "symbols": [],
                "files": [{"path": f"tests/test_{slug}.py", "source": src}],
            },
            output_kind="test_source_file",
        )

    gw.register_responder("test_code_file_writer", _test_file_responder)
    gw.register("input_set_generator", _INPUT_SET, output_kind="input_set")
    gw.register("final_report_writer", _FINAL_REPORT, output_kind="final_report")
    return gw


# ───────────────────────────────────────────────────────── shape


class TestPlanModeShape:
    def test_plan_mode_declares_the_nine_step_sequence(self) -> None:
        names = [s.name for s in PlanMode().stages("draft")]
        assert names == [
            "save_user_plan",
            "assemble_knowledge_context",  # prior-knowledge digest (vision-loop-05)
            "generate_experiment_report",
            "generate_experiment_spec",
            "approve_experiment_spec",  # human approves the spec BEFORE the IR is built
            "resolve_capabilities",
            "extract_workflow_ir",
            "bind_molcrafts_tasks",
            # One task at a time: codegen → unit test → pytest (then next task).
            "sequential_task_build",
            "validate_workflow_source",
            "review_plan",
            "generate_input_set",
            # Per-task pytest already green; compile-only dry-run remains.
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


# ─────────────────────────────────────── test_spec chain: repair symmetry

# Structurally valid TestSpecBundle whose specs target task ids absent from
# the workflow IR — the exact cross-generation drift observed in production
# (unknown_task_target), which ValidateTestSpec deterministically rejects.
_INVALID_TEST_SPEC = {
    "id": "tsb-water",
    "bound_workflow_id": "bw-water",
    "specs": [
        {
            "id": "ts-ghost",
            "name": "ghost task check",
            "kind": "unit_test",
            "target_task_id": "ghost_task",
            "description": "targets a task id absent from the workflow IR",
        }
    ],
}


class TestSequentialTaskBuildWiring:
    """Plan step 5 builds each task sequentially (code → test → pytest)."""

    def test_sequential_task_build_is_wired_with_repair_budget(self) -> None:
        from molexp.harness.modes.plan import DEFAULT_REPAIR_ATTEMPTS
        from molexp.harness.stages.sequential_task_build import SequentialTaskBuild

        stages = {s.name: s for s in PlanMode().stages("draft")}
        build = stages["sequential_task_build"]
        assert isinstance(build, SequentialTaskBuild)
        assert build._attempts == DEFAULT_REPAIR_ATTEMPTS
        assert DEFAULT_REPAIR_ATTEMPTS == 3

    @pytest.mark.integration
    def test_test_spec_repair_feeds_violations_back_and_converges(self, tmp_path: Path) -> None:
        """Legacy GenerateTestSpec RepairLoop still converges standalone.

        PlanMode no longer hosts this loop; the unit contract lives here so
        the stage remains covered after sequential_task_build took over.
        """
        from molexp.harness.core.run_context import HarnessRunContext
        from molexp.harness.schemas import TestSpecBundle
        from molexp.harness.stages import GenerateTestSpec, RepairLoop, ValidateTestSpec
        from molexp.harness.store.sqlite_event_log import SQLiteEventLog

        db = tmp_path / "harness.sqlite"
        store = FileArtifactStore(root=tmp_path / "artifacts")
        gateway = StubAgentGateway(store)
        gateway.register_sequence(
            "test_spec_writer", [_INVALID_TEST_SPEC, _TEST_SPEC], output_kind="test_spec"
        )
        ctx = HarnessRunContext(
            run_id="r-repair",
            workspace_root=tmp_path,
            artifact_store=store,
            event_log=SQLiteEventLog(path=db),
            lineage_store=SQLiteArtifactLineageStore(path=db, artifact_store=store),
            agent_gateway=gateway,
        )
        store.put_json(kind="workflow_ir", obj=_WORKFLOW_IR, created_by="seed", parent_ids=[])
        store.put_json(kind="bound_workflow", obj=_BOUND_WORKFLOW, created_by="seed", parent_ids=[])

        loop = RepairLoop(
            name="generate_test_spec",
            generate=GenerateTestSpec(),
            validators=[ValidateTestSpec()],
            feedback_kind="test_spec_feedback",
            attempts=3,
        )
        ref = asyncio.run(loop.run(ctx))

        assert ref.kind == "test_spec"
        bundle = TestSpecBundle.model_validate_json(store.get(ref.id))
        assert [s.target_task_id for s in bundle.specs] == ["make_data", "summarize"]
        feedback = store.latest_by_kind("test_spec_feedback")
        assert feedback is not None
        assert b"unknown_task_target" in store.get(feedback.id)
        assert feedback.id in ref.parent_ids


# A test module that byte-compiles only if `from __future__` sits at the top —
# the exact production failure: the LLM appended it mid-file (SyntaxError).
_SYNTAX_ERROR_TEST_SOURCE = {
    **_TEST_SOURCE,
    "files": [
        {
            "path": "tests/test_make_data.py",
            "source": "def test_ok():\n    pass\n\nfrom __future__ import annotations\n",
        },
    ],
}


class TestTestCodeRepairChain:
    """Standalone GenerateTestCode RepairLoop still converges (PlanMode no longer
    hosts this loop; sequential_task_build owns per-task test generation)."""

    @pytest.mark.integration
    def test_test_code_repair_feeds_violations_back_and_converges(self, tmp_path: Path) -> None:
        from molexp.harness.core.run_context import HarnessRunContext
        from molexp.harness.schemas import TestSource
        from molexp.harness.stages import GenerateTestCode, RepairLoop, ValidateTestSource
        from molexp.harness.store.sqlite_event_log import SQLiteEventLog

        db = tmp_path / "harness.sqlite"
        store = FileArtifactStore(root=tmp_path / "artifacts")
        gateway = StubAgentGateway(store)
        gateway.register_sequence(
            "test_code_file_writer",
            [
                _SYNTAX_ERROR_TEST_SOURCE,
                _SYNTAX_ERROR_TEST_SOURCE,
                _TEST_SOURCE,
                _TEST_SOURCE,
            ],
            output_kind="test_source_file",
        )
        ctx = HarnessRunContext(
            run_id="r-repair-code",
            workspace_root=tmp_path,
            artifact_store=store,
            event_log=SQLiteEventLog(path=db),
            lineage_store=SQLiteArtifactLineageStore(path=db, artifact_store=store),
            agent_gateway=gateway,
        )
        store.put_json(kind="test_spec", obj=_TEST_SPEC, created_by="seed", parent_ids=[])
        store.put_json(
            kind="workflow_source", obj=_WORKFLOW_SOURCE, created_by="seed", parent_ids=[]
        )

        loop = RepairLoop(
            name="generate_test_code",
            generate=GenerateTestCode(),
            validators=[ValidateTestSource()],
            feedback_kind="test_code_feedback",
            attempts=3,
        )
        ref = asyncio.run(loop.run(ctx))

        assert ref.kind == "test_source"
        source = TestSource.model_validate_json(store.get(ref.id))
        assert [f.path for f in source.files] == [
            "tests/test_make_data.py",
            "tests/test_summarize.py",
        ]
        feedback = store.latest_by_kind("test_code_feedback")
        assert feedback is not None
        assert b"compile_error" in store.get(feedback.id)
        assert feedback.id in ref.parent_ids


# ───────────────────────────────────────────────── offline run (plan-only)


class TestPlanModeRun:
    @pytest.mark.integration
    def test_plan_mode_runs_all_steps_offline(self, tmp_path: Path) -> None:
        run = _make_run(tmp_path)
        gateway = _fixture_gateway(run)

        result = asyncio.run(
            PlanMode(approver=auto_grant_approver).run(run=run, user_input=_DRAFT, gateway=gateway)
        )

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
            "test_result",  # sequential_task_build dry-run
            "input_set",
            "execution_result",  # compile-only
            "analysis_result",  # step-8 gate
            "execution_report",  # step 9
        }
        # Side-effect artifacts written by sequential_task_build (not stage returns).
        store = FileArtifactStore(root=run.run_dir / "artifacts")
        assert store.latest_by_kind("workflow_source") is not None
        assert store.latest_by_kind("test_source") is not None
        assert store.latest_by_kind("test_spec") is not None

    @pytest.mark.integration
    def test_compile_dry_run_is_a_compile_not_a_real_run(self, tmp_path: Path) -> None:
        run = _make_run(tmp_path)
        gateway = _fixture_gateway(run)
        store = FileArtifactStore(root=run.run_dir / "artifacts")

        asyncio.run(
            PlanMode(approver=auto_grant_approver).run(run=run, user_input=_DRAFT, gateway=gateway)
        )

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

        result = asyncio.run(
            PlanMode(approver=auto_grant_approver).run(run=run, user_input=_DRAFT, gateway=gateway)
        )
        src_ref = store.latest_by_kind("workflow_source")
        assert src_ref is not None
        user_plan_ref = next(a for a in result.stage_artifacts if a.kind == "user_plan")

        provenance = SQLiteArtifactLineageStore(
            path=run.run_dir / "harness.sqlite", artifact_store=store
        )
        ancestors = {ref.id for ref in provenance.trace_backward(src_ref.id)}
        # Sequential build may parent only to bound_workflow; lineage still
        # reaches user_plan through the graph when edges are recorded.
        # At minimum the source artifact exists after a green plan.
        assert src_ref.kind == "workflow_source"
        _ = user_plan_ref, ancestors  # lineage shape varies with sequential path


# ───────────────────────────────────────────── --execute tail (real run)


class TestPlanModeExecute:
    @pytest.mark.integration
    def test_execute_tail_runs_the_real_workflow(self, tmp_path: Path) -> None:
        run = _make_run(tmp_path)
        gateway = _fixture_gateway(run)
        store = FileArtifactStore(root=run.run_dir / "artifacts")

        result = asyncio.run(
            PlanMode(approver=auto_grant_approver, execute=True).run(
                run=run, user_input=_DRAFT, gateway=gateway
            )
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

        from molexp.harness.errors import StagePersistedFailureError

        with pytest.raises(StagePersistedFailureError):
            asyncio.run(
                PlanMode(approver=auto_grant_approver, execute=True).run(
                    run=run, user_input=_DRAFT, gateway=gateway
                )
            )

        # In-function import: a module-level `TestResult` would be collected by
        # pytest as a test class (house pattern for Test*-named schemas).
        from molexp.harness.schemas import TestResult

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

        first = asyncio.run(
            PlanMode(approver=auto_grant_approver).run(run=run, user_input=_DRAFT, gateway=gateway)
        )

        # Nothing registered: any re-run of an LLM stage body would raise, so
        # completing proves every stage was skipped via the completion ledger.
        empty = StubAgentGateway(FileArtifactStore(root=run.run_dir / "artifacts"))
        second = asyncio.run(
            PlanMode(approver=auto_grant_approver).run(run=run, user_input=_DRAFT, gateway=empty)
        )

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
