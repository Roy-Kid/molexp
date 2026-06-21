"""``molexp plan`` — end-to-end via Typer CliRunner, no LLM.

The production gateway factory (``molexp.cli.plan_cmd._make_gateway``) is
monkeypatched to return a :class:`StubAgentGateway` loaded with canned valid
outputs per planning agent, so the full 9-stage PlanMode pipeline runs
offline against a tmp workspace. Asserts the CLI wiring end-to-end: command
registration, model resolution failure, draft-from-arg / draft-from-file,
stage-artifact reporting, and the on-disk artifact + audit landing spots.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from molexp.cli import app

# A WorkflowSource program that compiles to a real Workflow (public API) —
# ValidateWorkflowSource actually compiles it, so it must be valid.
_VALID_SOURCE = """\
from molexp.workflow import TaskContext, WorkflowCompiler


def build_workflow() -> WorkflowCompiler:
    wf = WorkflowCompiler(name="demo")

    @wf.task
    async def build_system(ctx: TaskContext) -> dict:
        return {"structure": "system.pdb"}

    @wf.task(depends_on=["build_system"])
    async def simulate(ctx: TaskContext) -> dict:
        return {"trajectory": "traj.dcd"}

    return wf
"""

_EXPERIMENT_REPORT = {
    "title": "Water NEMD",
    "objective": "Measure ionic mobility",
    "system_description": "SPC/E water box under an applied field",
    "experimental_design": "Apply field; record current",
}
_WORKFLOW_IR = {
    "id": "wf-water",
    "name": "water_nemd",
    "objective": "Compute mobility",
    "inputs": {},
    "tasks": [
        {
            "id": "build",
            "name": "Pack water",
            "purpose": "Build SPC/E box",
            "task_type": "molecule_builder",
            "inputs": {},
            "outputs": {"structure": "structure.pdb"},
        }
    ],
    "edges": [],
    "expected_outputs": [],
}
_BOUND_WORKFLOW = {
    "id": "bw-water",
    "workflow_ir_id": "wf-water",
    "tasks": [
        {
            "id": "b-build",
            "ir_task_id": "build",
            "capability_id": "molpy.builder.water.SPCEBuilder",
            "package": "molpy",
            "callable": "molpy.builder.water.SPCEBuilder.run",
            "parameters": {},
            "inputs": {},
            "outputs": {"structure": "structure.pdb"},
        }
    ],
    "edges": [],
    "execution_backend": "local",
    "environment": {},
    "resource_policy": {
        "backend": "local",
        "max_runtime_s": 3600,
        "denied_paths": ["/", "~/.ssh"],
    },
}
_WORKFLOW_SOURCE = {
    "source": _VALID_SOURCE,
    "module_name": "water_nemd",
    "bound_workflow_id": "bw-water",
    "symbols": ["WorkflowCompiler", "TaskContext"],
}


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


def _patch_gateway(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make ``molexp plan`` build a StubAgentGateway instead of a live router."""

    def _fake_make_gateway(*, model: str, run: Any) -> Any:
        from molexp.harness.gateways.stub import StubAgentGateway
        from molexp.harness.store.file_artifact_store import FileArtifactStore

        # Share the run's artifact dir with the Mode-built ctx store.
        store = FileArtifactStore(root=run.run_dir / "artifacts")
        gw = StubAgentGateway(store)
        gw.register("experiment_report_writer", _EXPERIMENT_REPORT, output_kind="experiment_report")
        gw.register("workflow_ir_extractor", _WORKFLOW_IR, output_kind="workflow_ir")
        gw.register("bound_workflow_binder", _BOUND_WORKFLOW, output_kind="bound_workflow")
        gw.register("workflow_source_writer", _WORKFLOW_SOURCE, output_kind="workflow_source")
        return gw

    monkeypatch.setattr("molexp.cli.plan_cmd._make_gateway", _fake_make_gateway)


@pytest.mark.integration
def test_plan_command_is_registered(runner: CliRunner) -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "plan" in result.output


@pytest.mark.integration
def test_plan_runs_all_stages_against_a_workspace(
    runner: CliRunner, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_gateway(monkeypatch)

    result = runner.invoke(
        app,
        [
            "plan",
            "Simulate NEMD ionic mobility of an SPC/E water box",
            "--workspace",
            str(tmp_path),
            "--model",
            "stub-model",
        ],
    )

    assert result.exit_code == 0, result.output
    # Stage progress + artifact report rendered.
    assert "save_user_plan" in result.output
    assert "approval_gate" in result.output
    assert "workflow_source" in result.output
    assert "all stages completed" in result.output
    # Artifacts + audit records landed where the CLI says they do.
    run_dirs = list(tmp_path.rglob("harness.sqlite"))
    assert len(run_dirs) == 1, "expected exactly one plan run with an audit db"
    run_dir = run_dirs[0].parent
    artifacts = run_dir / "artifacts"
    assert artifacts.is_dir() and any(artifacts.iterdir())
    # Rich wraps long lines and emits ANSI codes; strip both to compare.
    plain = re.sub(r"\x1b\[[0-9;]*m", "", result.output)
    assert str(artifacts) in "".join(plain.split())
    assert (run_dir / ".mode_ledger").is_dir(), "per-run completion ledger written"


class TestInteractiveApprover:
    """The experiment-report review checkpoint on ``molexp plan``."""

    def test_plan_non_interactive_default_auto_grants(
        self, runner: CliRunner, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """ac-007 — under CliRunner (stdin is not a TTY) the experiment-spec
        review checkpoint auto-grants, with and without --yes; neither blocks."""
        _patch_gateway(monkeypatch)
        base = ["plan", "Simulate NEMD", "--workspace", str(tmp_path), "--model", "stub-model"]

        default_run = runner.invoke(app, base)
        assert default_run.exit_code == 0, default_run.output
        assert "all stages completed" in default_run.output

        yes_run = runner.invoke(app, [*base, "--yes"])
        assert yes_run.exit_code == 0, yes_run.output
        assert "all stages completed" in yes_run.output

    def test_auto_grants_when_assume_yes(self, tmp_path: Path) -> None:
        """ac-006 — InteractiveApprover auto-grants (no prompt) when --yes is set."""
        import asyncio
        from datetime import UTC, datetime

        from molexp.cli.plan_cmd import InteractiveApprover
        from molexp.harness.schemas import ApprovalRequest
        from molexp.workspace import Workspace

        ws = Workspace(tmp_path / "lab", name="lab")
        ws.materialize()
        run = ws.add_project("p").add_experiment("e").add_run(params={})
        approver = InteractiveApprover(run=run, assume_yes=True)
        request = ApprovalRequest(
            id="r",
            intent="experiment_spec",
            reason="x",
            triggered_by_policy="t",
            created_at=datetime.now(tz=UTC),
        )
        decision = asyncio.run(approver(request))
        assert decision.granted is True
        assert decision.decided_by == "cli-non-interactive"


@pytest.mark.integration
def test_plan_reads_draft_from_file(
    runner: CliRunner, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_gateway(monkeypatch)
    draft = tmp_path / "draft.md"
    draft.write_text("Simulate a polymer melt equilibration", encoding="utf-8")

    result = runner.invoke(
        app,
        [
            "plan",
            "--file",
            str(draft),
            "--workspace",
            str(tmp_path / "lab"),
            "--model",
            "stub-model",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "polymer melt" in result.output
    assert "all stages completed" in result.output


@pytest.mark.integration
def test_plan_without_model_exits_with_actionable_error(
    runner: CliRunner, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("molexp.cli.plan_cmd._configured_model", lambda: None)
    result = runner.invoke(app, ["plan", "a draft", "--workspace", str(tmp_path)])
    assert result.exit_code == 1
    assert "No model configured" in result.output
    assert "molexp config set agent.model" in result.output


@pytest.mark.integration
def test_plan_requires_exactly_one_draft_source(
    runner: CliRunner, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("molexp.cli.plan_cmd._configured_model", lambda: "stub-model")
    # Neither argument nor file.
    result = runner.invoke(app, ["plan", "--workspace", str(tmp_path)])
    assert result.exit_code == 1
    assert "exactly one way" in result.output
    # Both at once.
    draft = tmp_path / "d.md"
    draft.write_text("x", encoding="utf-8")
    result = runner.invoke(
        app, ["plan", "inline draft", "--file", str(draft), "--workspace", str(tmp_path)]
    )
    assert result.exit_code == 1
    assert "exactly one way" in result.output


@pytest.mark.integration
def test_plan_rerun_skips_completed_stages_via_ledger(
    runner: CliRunner, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Same draft twice → same run, second invocation resumes from the ledger."""
    _patch_gateway(monkeypatch)
    args = [
        "plan",
        "Simulate NEMD ionic mobility",
        "--workspace",
        str(tmp_path),
        "--model",
        "stub-model",
    ]

    first = runner.invoke(app, args)
    assert first.exit_code == 0, first.output

    # Second run: stage bodies are skipped (ledger hit), so even an
    # unregistered gateway (would raise on any LLM stage) completes.
    def _empty_gateway(*, model: str, run: Any) -> Any:
        from molexp.harness.gateways.stub import StubAgentGateway
        from molexp.harness.store.file_artifact_store import FileArtifactStore

        return StubAgentGateway(FileArtifactStore(root=run.run_dir / "artifacts"))

    monkeypatch.setattr("molexp.cli.plan_cmd._make_gateway", _empty_gateway)
    second = runner.invoke(app, args)
    assert second.exit_code == 0, second.output
    assert "all stages completed" in second.output


# Run-side canned outputs for `molexp plan --execute` (harness-run-mode-02
# T05). The TestSpec targets the canned IR task "build"; the test source
# imports from the canned WorkflowSource module ("water_nemd") — it is never
# executed here because the executor seam returns a DryRunExecutor.
_TEST_SPEC = {
    "id": "ts-water",
    "name": "workflow_compiles",
    "kind": "unit_test",
    "target_task_id": "build",
    "description": "The generated workflow module compiles into a workflow.",
}
_TEST_SOURCE = {
    "source": (
        "from water_nemd import build_workflow\n"
        "\n"
        "\n"
        "def test_build_compiles() -> None:\n"
        "    assert build_workflow().compile() is not None\n"
    ),
    "module_name": "test_water_nemd",
    "test_spec_id": "ts-water",
    "bound_workflow_id": "bw-water",
    "symbols": ["build_workflow"],
}
_FINAL_REPORT = {
    "title": "CannedWaterNemdFinalReport",
    "objective": "Measure ionic mobility from real execution outputs.",
    "methods_summary": "Canned workflow driven through the harness executor seam.",
    "test_summary": "Generated unit test compiled the workflow (dry-run executor).",
    "execution_summary": "Driver dry-run reported exit 0.",
    "results": "No numerical outputs under DryRunExecutor.",
    "conclusions": "Plan-then-run wiring verified end-to-end via the CLI.",
    "limitations": ["dry-run executor"],
    "next_steps": ["re-run with the default LocalExecutor"],
}

# The 10 RunMode stage names the `--execute` output table must show
# (ApprovalGate.name is "approval_gate" per stages/approval_gate.py).
_RUN_STAGE_NAMES = (
    "generate_test_spec",
    "validate_test_spec",
    "generate_test_code",
    "validate_test_source",
    "materialize_execution",
    "execute_tests",
    "execute_workflow",
    "generate_final_report",
    "approval_gate",
    "generate_audit_report",
)


def _patch_full_gateway(monkeypatch: pytest.MonkeyPatch) -> None:
    """Stub gateway with canned outputs for all 7 agents (plan + run)."""

    def _fake_make_gateway(*, model: str, run: Any) -> Any:
        from molexp.harness.gateways.stub import StubAgentGateway
        from molexp.harness.store.file_artifact_store import FileArtifactStore

        # Share the run's artifact dir with the Mode-built ctx store.
        store = FileArtifactStore(root=run.run_dir / "artifacts")
        gw = StubAgentGateway(store)
        gw.register("experiment_report_writer", _EXPERIMENT_REPORT, output_kind="experiment_report")
        gw.register("workflow_ir_extractor", _WORKFLOW_IR, output_kind="workflow_ir")
        gw.register("bound_workflow_binder", _BOUND_WORKFLOW, output_kind="bound_workflow")
        gw.register("workflow_source_writer", _WORKFLOW_SOURCE, output_kind="workflow_source")
        gw.register("test_spec_writer", _TEST_SPEC, output_kind="test_spec")
        gw.register("test_code_writer", _TEST_SOURCE, output_kind="test_source")
        gw.register("final_report_writer", _FINAL_REPORT, output_kind="final_report")
        return gw

    monkeypatch.setattr("molexp.cli.plan_cmd._make_gateway", _fake_make_gateway)


def _patch_executor(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make `--execute` drive RunMode stages through a DryRunExecutor."""

    def _fake_make_executor() -> Any:
        from molexp.harness import DryRunExecutor

        return DryRunExecutor()

    monkeypatch.setattr("molexp.cli.plan_cmd._make_executor", _fake_make_executor)


def test_make_executor_seam_defaults_to_local_executor() -> None:
    from molexp.cli import plan_cmd
    from molexp.harness import LocalExecutor

    assert isinstance(plan_cmd._make_executor(), LocalExecutor)


@pytest.mark.integration
def test_plan_execute_chains_run_mode_on_the_same_run(
    runner: CliRunner, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_full_gateway(monkeypatch)
    _patch_executor(monkeypatch)

    result = runner.invoke(
        app,
        [
            "plan",
            "Simulate NEMD ionic mobility of an SPC/E water box",
            "--workspace",
            str(tmp_path),
            "--model",
            "stub-model",
            "--execute",
        ],
    )

    assert result.exit_code == 0, result.output
    plain = re.sub(r"\x1b\[[0-9;]*m", "", result.output)
    # Every RunMode stage row rendered, plus the plan stages as before.
    assert "save_user_plan" in plain
    for stage_name in _RUN_STAGE_NAMES:
        assert stage_name in plain, f"missing RunMode stage {stage_name!r} in output"
    # The canned FinalReport is surfaced (single token: survives rich wrapping).
    assert "CannedWaterNemdFinalReport" in plain
