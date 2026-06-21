"""``molexp plan`` — run the harness PlanMode pipeline against a workspace.

The first production call path into :mod:`molexp.harness`: a natural-language
experiment draft flows through the canonical PlanMode stage sequence
(``SaveUserPlan → … → ApprovalGate``) on a ``workspace.Run``, driven by a
:class:`~molexp.harness.gateways.router_backed.RouterBackedAgentGateway`
built from the configured LLM. Stage completion is recorded in the per-run
completion ledger, so re-running the same draft resumes after the last
completed stage instead of re-invoking the LLM.

Model resolution mirrors ``molexp agent``: ``--model`` wins, else the
``agent.model`` key from ``molexp config``; with neither, the command fails
with an actionable message.

Heavy imports (``molexp.harness``, ``molexp.workspace``, the agent router)
are deferred into the command body so plain ``molexp --help`` stays fast.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import TYPE_CHECKING, Annotated

import typer

if TYPE_CHECKING:
    from molexp._typing import JSONValue
    from molexp.harness.executors import Executor
    from molexp.harness.gateways.gateway import AgentGateway
    from molexp.harness.stages import Approver
    from molexp.workspace.run import Run

__all__ = ["plan"]

_DRAFT_PREVIEW_CHARS = 80


def _make_approver(*, assume_yes: bool, run: Run) -> Approver | None:
    """Build the experiment-report review approver, or ``None`` for auto-grant.

    Returns ``None`` — letting the gate use its auto-grant default — when
    ``assume_yes`` is set or stdin is not a TTY (CI, pipes, the test runner),
    so non-interactive pipelines never block. On an interactive terminal it
    returns an async approver that prints the latest ``experiment_report`` and
    prompts ``[y/N]`` before the plan compiles. A seam: tests monkeypatch this.
    """
    import sys

    if assume_yes or not sys.stdin.isatty():
        return None

    from datetime import UTC, datetime

    from molexp.cli._common import rprint
    from molexp.harness.schemas import ApprovalDecision, ApprovalRequest, ExperimentReport
    from molexp.harness.store.file_artifact_store import FileArtifactStore

    store = FileArtifactStore(root=Path(run.run_dir / "artifacts"))

    async def approver(request: ApprovalRequest) -> ApprovalDecision:
        ref = store.latest_by_kind("experiment_report")
        if ref is not None:
            report = ExperimentReport.model_validate_json(store.get(ref.id))
            rprint("\n[bold]Review the experiment report before compiling:[/bold]")
            rprint(f"  title      : {report.title}")
            rprint(f"  objective  : {report.objective}")
            rprint(f"  system     : {report.system_description}")
            rprint(f"  design     : {report.experimental_design}")
            if report.assumptions:
                rprint(f"  assumptions: {'; '.join(report.assumptions)}")
        answer = input(f"Approve this experiment report? [{request.intent}] [y/N] ").strip().lower()
        granted = answer in ("y", "yes")
        return ApprovalDecision(
            request_id=request.id,
            granted=granted,
            decided_by="cli-interactive",
            decided_at=datetime.now(tz=UTC),
            reason=f"operator answered {answer!r}",
        )

    return approver


def _configured_model() -> str | None:
    """Return the ``agent.model`` value from ``molexp config``, if any.

    Delegates to the ``molexp agent`` command's resolver so both commands
    read the same configuration key. A seam: tests monkeypatch this.
    """
    from molexp.cli.agent_cmd import _configured_model as agent_configured_model

    return agent_configured_model()


def _make_gateway(*, model: str, run: Run) -> AgentGateway:
    """Build the production gateway for ``run`` from the resolved ``model``.

    Wires a :class:`~molexp.agent.PydanticAIRouter` (every tier on the same
    model id) into a ``RouterBackedAgentGateway`` whose artifact store shares
    the run's artifact directory with the Mode-built context. A seam: tests
    monkeypatch this to inject a ``StubAgentGateway`` instead.
    """
    from molexp.agent import PydanticAIRouter
    from molexp.agent.router import ModelTier
    from molexp.harness import RouterBackedAgentGateway
    from molexp.harness.prompts import prompts_by_agent
    from molexp.harness.prompts.workflow_source import (
        SYSTEM_PROMPT as WORKFLOW_SOURCE_SYSTEM_PROMPT,
    )
    from molexp.harness.schemas import (
        BoundWorkflow,
        ExperimentReport,
        FinalReport,
        TestSource,
        TestSpecBundle,
        WorkflowIR,
        WorkflowSource,
    )
    from molexp.harness.store.file_artifact_store import FileArtifactStore

    store = FileArtifactStore(root=Path(run.run_dir / "artifacts"))
    router = PydanticAIRouter(models=dict.fromkeys(ModelTier, model))
    return RouterBackedAgentGateway(
        router=router,
        artifact_store=store,
        agent_responses={
            "experiment_report_writer": ExperimentReport,
            "workflow_ir_extractor": WorkflowIR,
            "bound_workflow_binder": BoundWorkflow,
            "workflow_source_writer": WorkflowSource,
            "test_spec_writer": TestSpecBundle,
            "test_code_writer": TestSource,
            "final_report_writer": FinalReport,
        },
        output_kind_by_agent={
            "experiment_report_writer": "experiment_report",
            "workflow_ir_extractor": "workflow_ir",
            "bound_workflow_binder": "bound_workflow",
            "workflow_source_writer": "workflow_source",
            "test_spec_writer": "test_spec",
            "test_code_writer": "test_source",
            "final_report_writer": "final_report",
        },
        system_prompt_by_agent={
            **prompts_by_agent(),
            "workflow_source_writer": WORKFLOW_SOURCE_SYSTEM_PROMPT,
        },
        model=model,
    )


def _make_executor() -> Executor:
    """Build the executor RunMode drives its subprocesses through.

    Defaults to the real :class:`LocalExecutor`. A seam: tests monkeypatch
    this to inject a ``DryRunExecutor`` so ``--execute`` CLI tests spawn no
    subprocesses.
    """
    from molexp.harness import LocalExecutor

    return LocalExecutor()


def _resolve_draft(draft: str | None, file: Path | None) -> str:
    """Return the draft text from exactly one of ``draft`` / ``file``."""
    from molexp.cli._common import rprint

    if (draft is None) == (file is None):
        rprint(
            "[red]Provide the draft exactly one way:[/red] either as the "
            "[bold]DRAFT[/bold] argument or via [bold]--file <path>[/bold]."
        )
        raise typer.Exit(1)
    if file is not None:
        try:
            text = file.read_text(encoding="utf-8")
        except OSError as exc:
            rprint(f"[red]Could not read draft file:[/red] {exc}")
            raise typer.Exit(1) from exc
    else:
        assert draft is not None  # narrowed by the exactly-one check above
        text = draft
    if not text.strip():
        rprint("[red]The experiment draft is empty.[/red]")
        raise typer.Exit(1)
    return text


def plan(
    draft: Annotated[
        str | None,
        typer.Argument(help="Natural-language experiment draft (or use --file)."),
    ] = None,
    file: Annotated[
        Path | None,
        typer.Option("--file", "-f", help="Read the experiment draft from a file."),
    ] = None,
    workspace: Annotated[
        Path | None,
        typer.Option("--workspace", help="Workspace root; defaults to the current directory."),
    ] = None,
    model: Annotated[
        str | None,
        typer.Option("--model", help="Model id; defaults to `molexp config` agent.model."),
    ] = None,
    project: Annotated[
        str,
        typer.Option("--project", help="Project the plan run is filed under."),
    ] = "plans",
    experiment: Annotated[
        str,
        typer.Option("--experiment", help="Experiment the plan run is filed under."),
    ] = "plan",
    execute: Annotated[
        bool,
        typer.Option(
            "--execute",
            help="After planning, chain RunMode on the same run: generate + run "
            "tests, execute the workflow, and produce the final report.",
        ),
    ] = False,
    yes: Annotated[
        bool,
        typer.Option(
            "--yes/--non-interactive",
            "-y",
            help="Auto-approve the experiment-report review checkpoint. The "
            "default already auto-approves when stdin is not a TTY (CI/pipes).",
        ),
    ] = False,
) -> None:
    """Turn an experiment draft into validated molexp.workflow source (PlanMode)."""
    from molexp.cli._common import deterministic_run_id, rprint
    from molexp.harness import PlanMode, StageExecutionError
    from molexp.workspace import Workspace

    draft_text = _resolve_draft(draft, file)

    resolved_model = model or _configured_model()
    if not resolved_model:
        rprint(
            "[red]No model configured.[/red] Pass [bold]--model <id>[/bold] or run "
            "[bold]molexp config set agent.model <id>[/bold]."
        )
        raise typer.Exit(1)

    workspace_root = (workspace or Path.cwd()).resolve()
    ws = Workspace(workspace_root)
    ws.materialize()
    # Content-addressed run id: the same draft maps to the same Run, so the
    # per-run completion ledger resumes instead of starting a fresh pipeline.
    params: dict[str, JSONValue] = {"mode": "plan", "draft": draft_text}
    run = (
        ws.add_project(project)
        .add_experiment(experiment)
        .add_run(params, id=deterministic_run_id(params))
    )

    mode = PlanMode(approver=_make_approver(assume_yes=yes, run=run))
    stage_names = [stage.name for stage in mode.stages(draft_text)]
    preview = draft_text.strip().splitlines()[0][:_DRAFT_PREVIEW_CHARS]
    rprint(f"[bold]molexp plan[/bold] — {len(stage_names)} stages on run [bold]{run.id}[/bold]")
    rprint(f"  model     : {resolved_model}")
    rprint(f"  draft     : {preview}")
    rprint(f"  workspace : {workspace_root}")
    rprint(f"  stages    : {' -> '.join(stage_names)}")

    gateway = _make_gateway(model=resolved_model, run=run)
    try:
        result = asyncio.run(mode.run(run=run, user_input=draft_text, gateway=gateway))
    except StageExecutionError as exc:
        rprint(f"[red]Plan pipeline failed:[/red] {exc}")
        rprint(
            "[dim]Completed stages stay in the run's completion ledger — "
            "re-running the same draft resumes from the failed stage.[/dim]"
        )
        raise typer.Exit(1) from exc

    rprint("\n[green]OK[/green] all stages completed — stage artifacts:")
    for name, ref in zip(stage_names, result.stage_artifacts, strict=True):
        rprint(f"  {name:<26} {ref.kind:<20} {ref.id}")

    if execute:
        _execute_run_mode(run=run, draft_text=draft_text, gateway=gateway)

    rprint(f"\n  artifacts : {run.run_dir / 'artifacts'}")
    rprint(f"  audit db  : {run.run_dir / 'harness.sqlite'}  (events + artifact lineage)")
    rprint(
        "[dim]Re-running the same draft skips completed stages via the "
        "per-run completion ledger.[/dim]"
    )


def _execute_run_mode(*, run: Run, draft_text: str, gateway: AgentGateway) -> None:
    """Chain RunMode after PlanMode on the same content-addressed Run.

    The generated tests gate the real execution; stage failures surface with
    the same ledger-resume hint as the planning half (RunMode keeps its own
    per-run ledger, keyed by mode name + draft).
    """
    from molexp.cli._common import rprint
    from molexp.harness import FinalReport, RunMode, StageExecutionError
    from molexp.harness.store.file_artifact_store import FileArtifactStore

    mode = RunMode(executor=_make_executor())
    stage_names = [stage.name for stage in mode.stages(draft_text)]
    rprint(
        f"\n[bold]molexp plan --execute[/bold] — chaining RunMode "
        f"({len(stage_names)} stages) on run [bold]{run.id}[/bold]"
    )
    rprint(f"  stages    : {' -> '.join(stage_names)}")

    try:
        result = asyncio.run(mode.run(run=run, user_input=draft_text, gateway=gateway))
    except StageExecutionError as exc:
        rprint(f"[red]Run pipeline failed:[/red] {exc}")
        rprint(
            "[dim]Completed stages stay in the run's completion ledger — "
            "re-running the same draft with --execute resumes from the failed stage.[/dim]"
        )
        raise typer.Exit(1) from exc

    rprint("\n[green]OK[/green] run-mode stages completed — stage artifacts:")
    for name, ref in zip(stage_names, result.stage_artifacts, strict=True):
        rprint(f"  {name:<26} {ref.kind:<20} {ref.id}")

    report_ref = next((a for a in result.stage_artifacts if a.kind == "final_report"), None)
    if report_ref is None:
        rprint("[yellow]No final_report artifact found after RunMode.[/yellow]")
        return
    store = FileArtifactStore(root=Path(run.run_dir / "artifacts"))
    report = FinalReport.model_validate_json(store.get(report_ref.id))
    rprint(f"\n[bold]Final report[/bold] — {report.title}")
    rprint(f"  objective   : {report.objective}")
    rprint(f"  tests       : {report.test_summary}")
    rprint(f"  execution   : {report.execution_summary}")
    rprint(f"  results     : {report.results}")
    rprint(f"  conclusions : {report.conclusions}")
