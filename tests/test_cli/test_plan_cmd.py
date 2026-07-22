"""``molexp plan`` — end-to-end via Typer CliRunner, no LLM.

The production gateway factory (``molexp.cli.plan_cmd.PlanRuntime.build_gateway``) is
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
_INPUT_SET = {
    "id": "is-water",
    "experiment_spec_id": "spec-water",
    "title": "single-cell sweep",
    "sweep_axes": [],
    "strategy": "grid",
    "total_runs": 1,
}


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


@pytest.fixture(autouse=True)
def _disable_grounding(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep ``molexp plan`` offline: never spawn the molmcp server in CLI tests.

    ``--ground`` is on by default, so without this the StubAgentGateway path
    would reach out to the user-scope ``molmcp`` config. Tests that exercise
    grounding do so against the registry/catalog units directly.
    """
    monkeypatch.setattr("molexp.cli.plan_cmd._resolve_grounding", lambda *_a, **_k: None)


def _patch_gateway(monkeypatch: pytest.MonkeyPatch) -> None:
    """Stub every plan agent + a DryRunExecutor, so ``molexp plan`` runs offline.

    The 9-step pipeline includes the spec, input-set, and per-task test stages
    (plus the optional execute tail), so the stub registers all of them. The
    executor seam is patched to a ``DryRunExecutor`` so step-7 ExecuteTests /
    CompileWorkflow (and the tail) spawn no real subprocesses — the real
    compile + pytest path is covered by the harness integration tests.
    """

    def _fake_preflight(*, model: str) -> Any:
        return None  # stub router: the stub gateway never touches an LLM

    def _fake_make_gateway(*, model: str, run: Any, router: Any = None, **_kwargs: Any) -> Any:
        from molexp.harness.gateways.stub import StubAgentGateway
        from molexp.harness.store.file_artifact_store import FileArtifactStore

        # Share the run's artifact dir with the Mode-built ctx store.
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
        gw.register("test_code_writer", _TEST_SOURCE, output_kind="test_source")
        gw.register("input_set_generator", _INPUT_SET, output_kind="input_set")
        gw.register("final_report_writer", _FINAL_REPORT, output_kind="final_report")
        return gw

    monkeypatch.setattr("molexp.cli.plan_cmd.PlanRuntime.preflight", _fake_preflight)
    monkeypatch.setattr("molexp.cli.plan_cmd.PlanRuntime.build_gateway", _fake_make_gateway)
    _patch_executor(monkeypatch)


class TestPlanCmd:
    @pytest.mark.integration
    def test_plan_command_is_registered(self, runner: CliRunner) -> None:
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "plan" in result.output

    @pytest.mark.integration
    def test_plan_runs_all_stages_against_a_workspace(
        self, runner: CliRunner, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
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
                "--verbose",
                "--yes",
            ],
        )

        assert result.exit_code == 0, result.output
        # Verbose mode unfolds the internal stage names inside the nine steps.
        assert "save_user_plan" in result.output
        assert "generate_experiment_spec" in result.output
        assert "approve_plan" in result.output
        assert "generate_execution_report" in result.output
        assert "9 steps completed" in result.output
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

    @pytest.mark.integration
    def test_plan_reads_draft_from_file(
        self, runner: CliRunner, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
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
                "--yes",
            ],
        )

        assert result.exit_code == 0, result.output
        assert "polymer melt" in result.output
        assert "9 steps completed" in result.output

    @pytest.mark.integration
    def test_plan_requires_exactly_one_draft_source(
        self, runner: CliRunner, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
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

    def test_make_executor_seam_defaults_to_local_executor(self) -> None:
        from molexp.cli import plan_cmd
        from molexp.harness import LocalExecutor

        assert isinstance(plan_cmd.PlanRuntime.build_executor(), LocalExecutor)

    @pytest.mark.integration
    def test_plan_execute_runs_the_real_execution_tail(
        self, runner: CliRunner, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _patch_gateway(monkeypatch)  # registers final_report_writer + DryRunExecutor

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
                "--verbose",
                "--yes",
            ],
        )

        assert result.exit_code == 0, result.output
        plain = re.sub(r"\x1b\[[0-9;]*m", "", result.output)
        # The plan stages plus the opt-in real-execution tail all render.
        assert "save_user_plan" in plain
        for stage_name in _EXECUTE_TAIL_STAGES:
            assert stage_name in plain, f"missing execute-tail stage {stage_name!r} in output"
        # The canned FinalReport is surfaced (single token: survives rich wrapping).
        assert "CannedWaterNemdFinalReport" in plain


class TestPlanUiParity:
    """`molexp plan` (Python) lands the SAME records the UI reads.

    The invariant the user requires: a plan produced from the CLI must be
    indistinguishable, through the server's UI-facing routes, from one generated
    in the web app. Both paths now funnel through
    ``plan_runtime.materialize_plan_records``; this test drives the CLI, then
    queries the very endpoints the UI calls against the same workspace.
    """

    @pytest.mark.integration
    def test_cli_plan_is_visible_through_ui_routes(
        self, runner: CliRunner, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from fastapi.testclient import TestClient

        from molexp.cli._common import deterministic_run_id
        from molexp.server.app import create_app
        from molexp.server.dependencies import get_workspace
        from molexp.workspace import Workspace

        _patch_gateway(monkeypatch)
        draft = "Simulate NEMD ionic mobility of an SPC/E water box"
        result = runner.invoke(
            app,
            [
                "plan",
                draft,
                "--workspace",
                str(tmp_path),
                "--model",
                "stub-model",
                "--project",
                "lab",
                "--experiment",
                "nemd",
                "--yes",
            ],
        )
        assert result.exit_code == 0, result.output

        run_id = deterministic_run_id({"mode": "plan", "draft": draft})
        task_id = f"plan-{run_id}"

        # Serve the SAME workspace the CLI wrote to, exactly as `molexp serve` would.
        ws = Workspace(tmp_path)
        app_ = create_app()
        app_.dependency_overrides[get_workspace] = lambda: ws
        client = TestClient(app_)

        # 1) The plan shows in the Agents hub session list (planMode flag set).
        listed = client.get("/api/agent-tasks").json()
        match = next((t for t in listed["tasks"] if t["taskId"] == task_id), None)
        assert match is not None, listed
        assert match["planMode"] is True

        # 2) The session transcript carries the deliverables locator the
        #    progress rail + Deliverables panel read off `loop_completed.payload.plan`.
        task = client.get(f"/api/agent-tasks/{task_id}").json()
        events = task["events"]
        assert events[-1]["type"] == "loop_completed"
        plan = events[-1]["payload"]["plan"]
        assert plan["run_id"] == run_id
        assert plan["project_id"] == "lab"
        assert plan["experiment_id"] == "nemd"
        # Stage steps are present (drive the rail's progress states): the
        # transcript keys each of the nine steps on its representative artifact.
        kinds = {
            e["payload"]["result"]["artifact"] for e in events if e["type"] == "tool_call_completed"
        }
        assert {
            "experiment_report",
            "experiment_spec",
            "workflow_source",
            "execution_report",
        } <= kinds

        # 3) The structured deliverables the right panel renders are all there —
        #    every 9-step deliverable surfaces through the same route the UI calls.
        detail = client.get(f"/api/projects/lab/experiments/nemd/plans/{run_id}").json()
        assert detail["experimentReport"]["title"] == "Water NEMD"
        assert detail["experimentSpec"] is not None
        assert detail["experimentSpecYaml"]
        assert detail["capabilities"] is not None  # the resolved (or "no registry") catalog
        assert detail["workflowSource"]
        task_ids = {t["id"] for t in detail["tasks"]}
        assert {"build_system", "simulate"} <= task_ids
        assert detail["inputSet"] is not None
        assert detail["executionReport"]["target_name"] == "local"


class TestInteractiveApprover:
    """The experiment-report review checkpoint on ``molexp plan``."""

    def test_plan_non_interactive_suspends_without_yes(
        self, runner: CliRunner, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """vision-loop-01 — off a TTY without --yes the first gate SUSPENDS
        (exit 2, pending printed); an explicit --yes completes unattended."""
        _patch_gateway(monkeypatch)
        base = ["plan", "Simulate NEMD", "--workspace", str(tmp_path), "--model", "stub-model"]

        default_run = runner.invoke(app, base)
        assert default_run.exit_code == 2, default_run.output
        assert "approval pending" in default_run.output.lower()

        yes_run = runner.invoke(app, [*base, "--yes"])
        assert yes_run.exit_code == 0, yes_run.output
        assert "9 steps completed" in yes_run.output


# Canned per-task test outputs (step 5) + the execute-tail final report. The
# TestSpec targets the canned IR task "build"; the test source imports from the
# canned WorkflowSource module ("water_nemd") — never actually executed here
# because the executor seam returns a DryRunExecutor.
_TEST_SPEC = {
    "id": "tsb-water",
    "bound_workflow_id": "bw-water",
    "specs": [
        {
            "id": "ts-water",
            "name": "workflow_compiles",
            "kind": "unit_test",
            "target_task_id": "build",
            "description": "The generated workflow module compiles into a workflow.",
        }
    ],
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

# The four execute-tail stage names the `--execute` output table must show
# (the gate is named "approve_execution" in PlanMode's tail).
_EXECUTE_TAIL_STAGES = (
    "execute_workflow",
    "generate_final_report",
    "approve_execution",
    "generate_audit_report",
)


def _patch_executor(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make step-7 ExecuteTests/CompileWorkflow (+ the tail) use a DryRunExecutor."""

    def _fake_make_executor() -> Any:
        from molexp.harness import DryRunExecutor

        return DryRunExecutor()

    monkeypatch.setattr("molexp.cli.plan_cmd.PlanRuntime.build_executor", _fake_make_executor)


class TestPlanZeroResidue:
    """A failed preflight leaves NOTHING on disk (plan P0).

    All three failure classes — missing model, missing ``molexp[agent]``
    extra, missing/invalid model credentials — must exit non-zero with a
    one-line human-readable message (no traceback) BEFORE any workspace
    write: the target directory must not even be created.
    """

    def _invoke_in_empty_dir(self, runner: CliRunner, tmp_path: Path, args: list[str]) -> Any:
        target = tmp_path / "lab"
        assert not target.exists()
        result = runner.invoke(app, ["plan", "a draft", "--workspace", str(target), "--yes", *args])
        assert result.exit_code == 1, result.output
        assert "Traceback" not in result.output
        assert not target.exists(), "a failed preflight must leave no residue"
        return result

    def test_missing_model_exits_clean_with_no_residue(
        self, runner: CliRunner, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr("molexp.cli.plan_cmd._configured_model", lambda: None)
        result = self._invoke_in_empty_dir(runner, tmp_path, [])
        assert "No model configured" in result.output

    def test_missing_agent_extra_names_install_command_and_no_residue(
        self, runner: CliRunner, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Simulate a base install: pydantic_ai import fails -> one-line hint."""
        import importlib.abc
        import sys

        class _Block(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname: str, path: Any = None, target: Any = None) -> None:
                if fullname == "pydantic_ai" or fullname.startswith("pydantic_ai."):
                    raise ModuleNotFoundError(f"No module named {fullname!r}", name=fullname)
                return

        for mod in list(sys.modules):
            if mod == "pydantic_ai" or mod.startswith(("pydantic_ai.", "molexp.agent._pydanticai")):
                monkeypatch.delitem(sys.modules, mod, raising=False)
        blocker = _Block()
        sys.meta_path.insert(0, blocker)
        try:
            result = self._invoke_in_empty_dir(
                runner, tmp_path, ["--model", "anthropic:claude-sonnet-4-5"]
            )
        finally:
            sys.meta_path.remove(blocker)
        plain = re.sub(r"\x1b\[[0-9;]*m", "", result.output)
        assert 'pip install "molexp[agent]"' in " ".join(plain.split())

    @pytest.mark.integration
    def test_unknown_model_exits_clean_with_no_residue(
        self, runner: CliRunner, tmp_path: Path
    ) -> None:
        """The REAL preflight path: an unknown model id fails before any write."""
        result = self._invoke_in_empty_dir(runner, tmp_path, ["--model", "stub-model"])
        plain = re.sub(r"\x1b\[[0-9;]*m", "", result.output)
        assert "'stub-model'" in plain

    @pytest.mark.integration
    def test_missing_api_key_exits_clean_with_no_residue(
        self, runner: CliRunner, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The REAL preflight path: a missing provider key fails pre-write."""
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        result = self._invoke_in_empty_dir(
            runner, tmp_path, ["--model", "anthropic:claude-sonnet-4-5"]
        )
        plain = re.sub(r"\x1b\[[0-9;]*m", "", result.output)
        assert "ANTHROPIC_API_KEY" in " ".join(plain.split())


class TestPlanNineStepBanner:
    """Banner + progress speak the documented nine-step vocabulary.

    Internal stage names (save_user_plan, bind_molcrafts_tasks, ...) fold
    into ``-v/--verbose``; the default output shows only the nine step
    titles from docs/guide/plan-mode.md.
    """

    STEP_TITLES = (
        "Draft proposal",
        "Draft spec",
        "Resolve capabilities",
        "Workflow IR",
        "Tasks + per-task tests",
        "Input set",
        "Compile / dry-run",
        "Review",
        "Execution report",
    )

    def _normalized(self, output: str) -> str:
        plain = re.sub(r"\x1b\[[0-9;]*m", "", output)
        return " ".join(plain.split())

    @pytest.mark.integration
    def test_default_output_shows_steps_and_hides_stage_names(
        self, runner: CliRunner, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _patch_gateway(monkeypatch)
        result = runner.invoke(
            app,
            [
                "plan",
                "Simulate NEMD",
                "--workspace",
                str(tmp_path),
                "--model",
                "stub-model",
                "--yes",
            ],
        )
        assert result.exit_code == 0, result.output
        text = self._normalized(result.output)
        assert "9 steps on run" in text
        for title in self.STEP_TITLES:
            assert title in text, f"step title {title!r} missing from banner"
        # Internal stage names stay folded away without --verbose.
        for stage_name in (
            "save_user_plan",
            "bind_molcrafts_tasks",
            "materialize_execution",
            "generate_execution_report",
        ):
            assert stage_name not in text, f"stage name {stage_name!r} leaked at default verbosity"
        assert "9 steps completed" in text
