"""Plan pipeline suspension on pending approvals (spec vision-loop-01-approval-inbox).

A bare ``PlanMode()`` (no approver) driven through the shared
``drive_plan_mode`` lifecycle must **suspend** at the ``approve_experiment_spec``
gate, never auto-grant: ``ApprovalPendingError`` propagates, the plan Run lands
``failed`` with ``metadata.error`` naming the approval (frozen status
vocabulary — no new Run status words), and the pending request is recorded in
the run's approval store.

A decision recorded in that store lets a re-drive pass the gate: the stage
ledger skips completed stages (asserted via gateway-call counters — no LLM
stage runs twice) and the pipeline proceeds to the next gate. Granting the
step-8 review too completes the plan; the SUCCEEDED terminal clears
``metadata.error``.

Offline and deterministic: ``StubAgentGateway`` cans every LLM output and
``DryRunExecutor`` skips the step-7 subprocesses.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from pathlib import Path

import pytest

from molexp.harness import DryRunExecutor, PlanMode
from molexp.harness.errors import ApprovalPendingError
from molexp.harness.gateways.stub import StubAgentGateway
from molexp.harness.schemas import AgentCallResult, AgentCallSpec, ApprovalDecision
from molexp.harness.store.approval_store import SQLiteApprovalStore
from molexp.harness.store.file_artifact_store import FileArtifactStore
from molexp.services.plan_runtime import drive_plan_mode
from molexp.workspace import Workspace

_DRAFT = "Simulate NEMD ionic mobility"

# ── canned stage outputs (mirrors tests/test_harness/test_plan_mode.py) ──────

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


class _CountingGateway(StubAgentGateway):
    """StubAgentGateway that counts calls per agent — the stage-call counter."""

    def __init__(self, artifact_store: FileArtifactStore) -> None:
        super().__init__(artifact_store)
        self.calls: dict[str, int] = {}

    async def call(self, spec: AgentCallSpec) -> AgentCallResult:
        self.calls[spec.agent_name] = self.calls.get(spec.agent_name, 0) + 1
        return await super().call(spec)


def _pre_gate_gateway(run) -> _CountingGateway:
    """Only the two agents that run BEFORE the spec-approval gate."""
    gw = _CountingGateway(FileArtifactStore(root=run.run_dir / "artifacts"))
    gw.register("experiment_report_writer", _EXPERIMENT_REPORT, output_kind="experiment_report")
    gw.register("experiment_spec_generator", _EXPERIMENT_SPEC, output_kind="experiment_spec")
    return gw


def _full_gateway(run) -> _CountingGateway:
    gw = _pre_gate_gateway(run)
    gw.register("workflow_ir_extractor", _WORKFLOW_IR, output_kind="workflow_ir")
    gw.register("bound_workflow_binder", _BOUND_WORKFLOW, output_kind="bound_workflow")
    gw.register("workflow_source_writer", _WORKFLOW_SOURCE, output_kind="workflow_source")
    # Per-task workflow codegen fans one call per bound task; return the slice's
    # matching module (the assembly is synthesized by the stage).
    _TASK_MODULE = {"make_data": _TASK_MAKE_DATA, "summarize": _TASK_SUMMARIZE}

    def _wf_file_responder(spec, store):
        from molexp.harness.schemas import BoundWorkflow

        slice_wf = BoundWorkflow.model_validate_json(store.get(spec.input_artifact_ids[0]))
        task = slice_wf.tasks[0]
        src = _TASK_MODULE[task.ir_task_id]
        return gw.make_response(
            {
                "source": src,
                "module_name": task.ir_task_id,
                "bound_workflow_id": slice_wf.id,
                "symbols": [],
                "files": [{"path": f"workflow/{task.ir_task_id}.py", "source": src}],
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
    gw.register("test_code_writer", _TEST_SOURCE, output_kind="test_source")
    # Multi-task suites fan out per file; register the per-file writer too.
    gw.register("test_code_file_writer", _TEST_SOURCE, output_kind="test_source_file")
    gw.register("input_set_generator", _INPUT_SET, output_kind="input_set")
    return gw


def _drive(run, gateway) -> None:
    """One drive of a bare (approver-less) PlanMode through the run lifecycle."""
    asyncio.run(
        drive_plan_mode(
            PlanMode(executor=DryRunExecutor()),
            run=run,
            user_input=_DRAFT,
            gateway=gateway,
        )
    )


def _grant(request_id: str) -> ApprovalDecision:
    return ApprovalDecision(
        request_id=request_id,
        granted=True,
        decided_by="inbox-operator",
        decided_at=datetime.now(tz=UTC),
        reason="approved from the inbox",
    )


@pytest.fixture()
def run(tmp_path: Path):
    ws = Workspace(tmp_path / "lab", name="approval-lab")
    ws.materialize()
    return ws.add_project("demo").add_experiment("nemd").add_run(params={})


def _store(run) -> SQLiteApprovalStore:
    return SQLiteApprovalStore(run.run_dir / "harness.sqlite")


# ─────────────────────────────────────────────────── suspension


class TestPlanModeSuspends:
    def test_bare_plan_mode_suspends_at_the_spec_gate(self, run) -> None:
        gateway = _pre_gate_gateway(run)

        with pytest.raises(ApprovalPendingError) as excinfo:
            _drive(run, gateway)

        assert excinfo.value.run_id == run.id
        assert [r.intent for r in excinfo.value.requests] == ["experiment_spec"]

    def test_suspension_marks_the_run_failed_naming_the_approval(self, run) -> None:
        with pytest.raises(ApprovalPendingError):
            _drive(run, _pre_gate_gateway(run))

        assert run.status == "failed"
        error = run.metadata.error
        assert error is not None
        assert "approval" in f"{error.type}: {error.message}".lower()

    def test_suspension_records_the_pending_request_in_the_run_store(self, run) -> None:
        with pytest.raises(ApprovalPendingError):
            _drive(run, _pre_gate_gateway(run))

        [pending] = _store(run).pending(run.id)
        assert pending.intent == "experiment_spec"


# ─────────────────────────────────── grant → re-drive → resume via ledger


class TestGrantThenRedrive:
    @pytest.mark.integration
    def test_grant_and_redrive_skips_completed_stages_and_proceeds(self, run) -> None:
        """Suspend → grant → re-drive: the ledger skips the pre-gate stages
        (call counters stay at 1), the gate passes on the stored grant, and the
        pipeline proceeds until the NEXT approver-less gate (step-8 review).
        Granting that too completes the plan and clears ``metadata.error``."""
        gateway = _full_gateway(run)
        store = _store(run)

        # 1. First drive suspends at the spec gate.
        with pytest.raises(ApprovalPendingError):
            _drive(run, gateway)
        assert gateway.calls == {
            "experiment_report_writer": 1,
            "experiment_spec_generator": 1,
        }
        [spec_request] = store.pending(run.id)

        # 2. Decision lands in the store; the re-drive passes the gate.
        store.record_decision(_grant(spec_request.id))
        with pytest.raises(ApprovalPendingError) as second:
            _drive(run, gateway)

        # Proceeded past the spec gate: suspended at the step-8 review gate now.
        assert [r.intent for r in second.value.requests] == ["final_report"]
        assert second.value.requests[0].id != spec_request.id
        # Ledger resume: no pre-gate LLM stage ran twice …
        assert gateway.calls["experiment_report_writer"] == 1
        assert gateway.calls["experiment_spec_generator"] == 1
        # … while the post-gate pipeline really ran. Workflow codegen fans one
        # call per bound task (two here: make_data + summarize).
        assert gateway.calls["workflow_ir_extractor"] == 1
        assert gateway.calls["workflow_source_file_writer"] == 2

        # 3. Granting the review too completes the plan; SUCCEEDED clears error.
        [review_request] = store.pending(run.id)
        store.record_decision(_grant(review_request.id))
        _drive(run, gateway)

        assert run.status == "succeeded"
        assert run.metadata.error is None
