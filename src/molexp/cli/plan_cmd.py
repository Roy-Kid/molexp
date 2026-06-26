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
    from molexp.harness.registry.capability_registry import CapabilityRegistry
    from molexp.harness.schemas import ApprovalDecision, ApprovalRequest, ModeResult
    from molexp.harness.store.file_artifact_store import FileArtifactStore
    from molexp.workspace import Workspace
    from molexp.workspace.models import ComputeTarget
    from molexp.workspace.run import Run

__all__ = ["plan"]

_DRAFT_PREVIEW_CHARS = 80


class InteractiveApprover:
    """``Approver`` for ``molexp plan``'s review checkpoints.

    A callable approver (mirrors the ``Approver`` protocol via
    :meth:`__call__`). It **auto-grants without prompting** when ``assume_yes``
    is set or stdin is not a TTY (CI, pipes, the test runner) — so
    non-interactive pipelines never block. On an interactive terminal it gates
    each checkpoint, branching on the request intent:

    - ``experiment_spec`` — the **pre-compile** gate. Prints the concrete
      ``experiment_spec`` (the resolved variables/conditions + answered
      questions) and prompts ``[y/N]`` BEFORE the spec is fed to the LLM to
      build the workflow. A rejection stops the pipeline before any IR/source.
    - ``final_report`` — the terminal whole-plan review.

    ``PlanMode(approver=InteractiveApprover(...))`` wires it into both gates.
    """

    def __init__(self, *, run: Run, assume_yes: bool = False) -> None:
        self._run = run
        self._assume_yes = assume_yes

    def _interactive(self) -> bool:
        import sys

        return not self._assume_yes and sys.stdin.isatty()

    async def __call__(self, request: ApprovalRequest) -> ApprovalDecision:
        from datetime import UTC, datetime

        from molexp.harness.schemas import ApprovalDecision

        if not self._interactive():
            return ApprovalDecision(
                request_id=request.id,
                granted=True,
                decided_by="cli-non-interactive",
                decided_at=datetime.now(tz=UTC),
                reason="auto-granted (non-interactive: --yes or no TTY)",
            )

        if request.intent == "experiment_spec":
            self._print_spec()
            prompt = "Approve this spec and compile the workflow?"
        else:
            self._print_final_summary()
            prompt = "Approve this plan as final?"
        answer = input(f"{prompt} [{request.intent}] [y/N] ").strip().lower()
        return ApprovalDecision(
            request_id=request.id,
            granted=answer in ("y", "yes"),
            decided_by="cli-interactive",
            decided_at=datetime.now(tz=UTC),
            reason=f"operator answered {answer!r}",
        )

    def _store(self) -> FileArtifactStore:
        from molexp.harness.store.file_artifact_store import FileArtifactStore

        return FileArtifactStore(root=Path(self._run.run_dir / "artifacts"))

    def _print_spec(self) -> None:
        """Print the concrete experiment_spec the operator approves pre-compile."""
        import json

        from molexp.cli._common import rprint

        store = self._store()
        ref = store.latest_by_kind("experiment_spec")
        rprint("\n[bold]Review the concrete spec before compiling the workflow:[/bold]")
        if ref is None:
            rprint("  (no experiment_spec artifact found)")
            return
        spec = json.loads(store.get(ref.id))
        rprint(f"  title     : {spec.get('title')}")
        rprint(f"  objective : {spec.get('objective')}")
        for v in spec.get("variables", []):
            val = (v.get("value") or {}).get("value")
            rprint(f"  variable  : {v.get('name')} = {val} {v.get('unit') or ''}".rstrip())
        for q in spec.get("resolved_questions", []):
            rprint(f"  resolved  : {q.get('question')} -> {q.get('answer')}")

    def _print_final_summary(self) -> None:
        """Print a brief whole-plan summary for the terminal review gate."""
        from molexp.cli._common import rprint

        store = self._store()
        rprint("\n[bold]Review the full verified plan:[/bold]")
        has_source = store.latest_by_kind("workflow_source") is not None
        has_dry = store.latest_by_kind("execution_result") is not None
        rprint(f"  workflow source generated : {has_source}")
        rprint(f"  compiled / dry-ran        : {has_dry}")


def _configured_model() -> str | None:
    """Return the ``agent.model`` value from ``molexp config``, if any.

    Delegates to the ``molexp agent`` command's resolver so both commands
    read the same configuration key. A seam: tests monkeypatch this.
    """
    from molexp.cli.agent_cmd import _configured_model as agent_configured_model

    return agent_configured_model()


def _resolve_grounding(workspace_root: Path, *, ground: bool) -> CapabilityRegistry | None:
    """Build a molmcp-backed ``CapabilityRegistry`` when ``--ground`` is set.

    Returns ``None`` when grounding is off or molmcp is unavailable (the helper
    prints a visible notice in the latter case — never a silent downgrade).
    """
    if not ground:
        return None
    from molexp.cli._common import rprint
    from molexp.mcp_capabilities import resolve_capability_registry

    return resolve_capability_registry(
        workspace_root, notify=lambda message: rprint(f"[dim]{message}[/dim]")
    )


class PlanRuntime:
    @staticmethod
    def build_gateway(*, model: str, run: Run) -> AgentGateway:
        """Build the production gateway for ``run`` from the resolved ``model``.

        Wires a :class:`~molexp.agent.PydanticAIRouter` (every tier on the same
        model id) into a ``RouterBackedAgentGateway`` whose artifact store shares
        the run's artifact directory with the Mode-built context. A seam: tests
        monkeypatch this to inject a ``StubAgentGateway`` instead.
        """
        from molexp.agent import PydanticAIRouter
        from molexp.agent.router import ModelTier
        from molexp.harness import RouterBackedAgentGateway
        from molexp.harness.gateways import (
            plan_agent_responses,
            plan_output_kinds,
            plan_system_prompts,
        )
        from molexp.harness.store.file_artifact_store import FileArtifactStore

        store = FileArtifactStore(root=Path(run.run_dir / "artifacts"))
        router = PydanticAIRouter(models=dict.fromkeys(ModelTier, model))
        return RouterBackedAgentGateway(
            router=router,
            artifact_store=store,
            agent_responses=plan_agent_responses(),
            output_kind_by_agent=plan_output_kinds(),
            system_prompt_by_agent=plan_system_prompts(),
            model=model,
        )

    @staticmethod
    def build_executor() -> Executor:
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
            help="After the plan is reviewed (step 8), run the workflow for real "
            "on the same run: the opt-in execution tail (real pytest + engine) "
            "plus the final report and audit. Off by default — the plan stops at "
            "the execution report.",
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
    ground: Annotated[
        bool,
        typer.Option(
            "--ground/--no-ground",
            help="Ground task binding against the molcrafts toolchain via the "
            "configured `molmcp` MCP server: the binder picks capabilities from "
            "the live catalog and ValidateBoundWorkflow checks each bound "
            "capability exists, its call shape, and its backend. On by default; "
            "skips with a notice when molmcp is not available. Use --no-ground "
            "to disable.",
        ),
    ] = True,
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
    exp = ws.add_project(project).add_experiment(experiment)
    run = exp.add_run(params, id=deterministic_run_id(params))

    mode = PlanMode(
        approver=InteractiveApprover(run=run, assume_yes=yes),
        executor=PlanRuntime.build_executor(),
        execute=execute,
        compute_target=_resolve_compute_target(run, ws),
    )
    stage_names = [stage.name for stage in mode.stages(draft_text)]
    preview = draft_text.strip().splitlines()[0][:_DRAFT_PREVIEW_CHARS]
    rprint(f"[bold]molexp plan[/bold] — {len(stage_names)} stages on run [bold]{run.id}[/bold]")
    rprint(f"  model     : {resolved_model}")
    rprint(f"  draft     : {preview}")
    rprint(f"  workspace : {workspace_root}")
    rprint(f"  stages    : {' -> '.join(stage_names)}")

    gateway = PlanRuntime.build_gateway(model=resolved_model, run=run)
    capability_registry = _resolve_grounding(workspace_root, ground=ground)
    try:
        result = asyncio.run(
            mode.run(
                run=run,
                user_input=draft_text,
                gateway=gateway,
                capability_registry=capability_registry,
            )
        )
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

    # Materialize the SAME UI-facing records the server's `POST /plan-tasks`
    # writes — persist the workflow IR onto the experiment + record the Agents
    # session (with the deliverables locator) and Knowledge note — so a plan
    # produced here is identical, in the UI, to one generated from the web app.
    from molexp.server.plan_runtime.materialize import materialize_plan_records

    task_id = f"plan-{run.id}"
    materialize_plan_records(
        run=run,
        experiment=exp,
        workspace_root=str(workspace_root),
        task_id=task_id,
        draft=draft_text,
        model=resolved_model,
    )
    rprint(f"  ui session: [bold]{task_id}[/bold] (open the Agents tab to see this plan)")

    if execute:
        _print_final_report(run, result)

    rprint(f"\n  artifacts : {run.run_dir / 'artifacts'}")
    rprint(f"  audit db  : {run.run_dir / 'harness.sqlite'}  (events + artifact lineage)")
    rprint(
        "[dim]Re-running the same draft skips completed stages via the "
        "per-run completion ledger.[/dim]"
    )


def _resolve_compute_target(run: Run, ws: Workspace) -> ComputeTarget | None:
    """Resolve the run's intended ``ComputeTarget`` from the workspace registry.

    ``RunMetadata.target`` names a target registered in
    ``WorkspaceMetadata.targets``; the step-9 execution report describes it.
    A fresh workspace with no targets resolves to ``None`` (a local default).
    """
    target_name = getattr(run.metadata, "target", None)
    if not target_name:
        return None
    targets = getattr(getattr(ws, "metadata", None), "targets", []) or []
    return next((t for t in targets if t.name == target_name), None)


def _print_final_report(run: Run, result: ModeResult) -> None:
    """Print the final report produced by the ``--execute`` tail."""
    from molexp.cli._common import rprint
    from molexp.harness import FinalReport
    from molexp.harness.store.file_artifact_store import FileArtifactStore

    report_ref = next((a for a in result.stage_artifacts if a.kind == "final_report"), None)
    if report_ref is None:
        rprint("[yellow]No final_report artifact found after execution.[/yellow]")
        return
    store = FileArtifactStore(root=Path(run.run_dir / "artifacts"))
    report = FinalReport.model_validate_json(store.get(report_ref.id))
    rprint(f"\n[bold]Final report[/bold] — {report.title}")
    rprint(f"  objective   : {report.objective}")
    rprint(f"  tests       : {report.test_summary}")
    rprint(f"  execution   : {report.execution_summary}")
    rprint(f"  results     : {report.results}")
    rprint(f"  conclusions : {report.conclusions}")
