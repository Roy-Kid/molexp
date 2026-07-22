"""``molexp plan`` — run the harness emergent-planning pipeline on a workspace.

The production call path into :mod:`molexp.harness`: a natural-language
experiment draft is handed to :class:`~molexp.harness.EmergentPlanOrchestrator`
on a ``workspace.Run``, driven by a
:class:`~molexp.harness.gateways.router_backed.RouterBackedAgentGateway`
built from the configured LLM. The orchestrator runs two phases —
**phase 1** emergent planning (draft → task board → hard review gate → frozen
experiment plan) and **phase 2** deterministic realization (a separate phase,
not driven by this command yet). With no approver the review gate suspends
store-first (exit 2); a re-run with a stored grant replays through the gate.

Model resolution mirrors ``molexp agent``: ``--model`` wins, else the
``agent.model`` key from ``molexp config``; with neither, the command fails
with an actionable message.

Zero-residue ordering: draft + model resolution and the agent-stack
preflight (:meth:`PlanRuntime.preflight` → shared
``services.plan_runtime.preflight_plan_router``) all run BEFORE the first
workspace write, so a missing ``molexp[agent]`` extra, an unconfigured
model, or a missing API key exits with a one-line message and leaves the
target directory untouched.

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
    from molexp.agent.router import Router
    from molexp.harness.gateways.gateway import AgentGateway
    from molexp.harness.registry.capability_registry import CapabilityRegistry
    from molexp.harness.schemas import ApprovalDecision, ApprovalRequest
    from molexp.services.plan_runtime import PlanRecordOutcome
    from molexp.workspace.run import Run

__all__ = ["plan"]

_DRAFT_PREVIEW_CHARS = 80


class InteractiveApprover:
    """``Approver`` for ``molexp plan``'s emergent review gate.

    A callable approver (satisfies the harness ``Approver`` type —
    ``async (ApprovalRequest) -> ApprovalDecision`` — via :meth:`__call__`). It
    auto-grants **only under an explicit ``--yes``**; the command constructs it
    solely on a TTY or with ``--yes``, and passes ``approve=None`` otherwise —
    the gate then suspends pending instead of granting. On an interactive
    terminal it renders the review pack for the request and prompts
    ``[a]pprove / [r]eject / [v]revise``.

    :class:`~molexp.harness.EmergentPlanOrchestrator` receives it as its
    ``approve`` seam and asks it at the plan-tool side-effect gate and the hard
    ``approve_experiment_plan`` review gate before the plan is frozen.
    """

    def __init__(self, *, run: Run, assume_yes: bool = False) -> None:
        self._run = run
        self._assume_yes = assume_yes

    def _interactive(self) -> bool:
        import sys

        return not self._assume_yes and sys.stdin.isatty()

    async def __call__(self, request: ApprovalRequest) -> ApprovalDecision:
        from datetime import UTC, datetime
        from pathlib import Path

        from molexp.harness.schemas import ApprovalDecision, ReviewDecision
        from molexp.harness.store.file_artifact_store import FileArtifactStore
        from molexp.services.plan_runtime.preview import (
            build_review_pack,
            render_review_pack,
        )

        if not self._interactive():
            # Only reachable via --yes: the command constructs this approver
            # solely on a TTY or with --yes, so a non-interactive call IS the
            # operator's explicit blanket consent — named in the audit trail.
            return ApprovalDecision(
                request_id=request.id,
                granted=True,
                decided_by="cli---yes",
                decided_at=datetime.now(tz=UTC),
                reason="auto-granted (--yes)",
            )

        pack = build_review_pack(self._run, request.intent)
        from molexp.cli._common import rprint

        rprint("\n[bold]Review pack:[/bold]")
        for line in render_review_pack(pack).splitlines():
            rprint(f"  {line}")
        prompt = (
            "Approve this spec and compile the workflow?"
            if request.intent == "experiment_spec"
            else "Approve this plan as final?"
        )
        answer = (
            input(f"{prompt} [{request.intent}] [a]pprove / [r]eject / [v]revise: ").strip().lower()
        )
        if answer in ("a", "approve", "y", "yes"):
            action = "approve"
            granted = True
        elif answer in ("v", "revise"):
            action = "revise"
            granted = False
        else:
            # fail-closed: unknown / empty / r / reject → reject
            action = "reject"
            granted = False

        decision = ReviewDecision(
            pack_id=pack.pack_id,
            action=action,  # type: ignore[arg-type]
            decided_by="cli-interactive",
            decided_at=datetime.now(tz=UTC),
            reason=f"operator answered {answer!r}",
        )
        FileArtifactStore(root=Path(str(self._run.run_dir)) / "artifacts").put_json(
            kind="review_decision",
            obj=decision.model_dump(mode="json"),
            created_by="cli.InteractiveApprover",
            parent_ids=[],
        )
        return ApprovalDecision(
            request_id=request.id,
            granted=granted,
            decided_by="cli-interactive",
            decided_at=decision.decided_at,
            reason=decision.reason,
        )


def _print_record_errors(outcome: PlanRecordOutcome) -> None:
    """Surface record-materialization failures loudly (never exit-code-changing).

    The science and its artifacts are already safely on disk; the record layer
    is a projection — so a broken record prints a red block naming each failed
    record instead of silently vanishing into a log (no-silent-fallback law).
    """
    if not outcome.errors:
        return
    from molexp.cli._common import rprint

    rprint("[red]record materialization failed for:[/red]")
    for error in outcome.errors:
        rprint(f"  - {error.record}: {error.error}")


def _configured_model() -> str | None:
    """Return the ``agent.model`` value from ``molexp config``, if any.

    Delegates to the ``molexp agent`` command's resolver so both commands
    read the same configuration key. A seam: tests monkeypatch this.
    """
    from molexp.cli.agent_cmd import _configured_model as agent_configured_model

    return agent_configured_model()


def _resolve_grounding(
    workspace_root: Path,
    *,
    ground: bool,
    task: str | None = None,
) -> CapabilityRegistry | None:
    """Build a molmcp-backed ``CapabilityRegistry`` when ``--ground`` is set.

    Returns ``None`` when grounding is off or molmcp is unavailable (the helper
    prints a visible notice in the latter case — never a silent downgrade).
    ``task`` is the experiment draft so discovery follows the user request
    (auto-discovery — no fixed polymer query table).
    """
    if not ground:
        return None
    from molexp.cli._common import rprint
    from molexp.mcp_capabilities import resolve_capability_registry

    return resolve_capability_registry(
        workspace_root,
        task=task,
        notify=lambda message: rprint(f"[dim]{message}[/dim]"),
    )


class PlanRuntime:
    @staticmethod
    def preflight(*, model: str) -> Router | None:
        """Validate the agent stack + credentials BEFORE any disk write.

        Delegates to :func:`molexp.services.plan_runtime.preflight_plan_router`
        (shared with the server path): imports the agent stack, constructs the
        router, and forces credential resolution — no network, no disk. Raises
        :class:`~molexp.services.plan_runtime.PlanPreflightError` with a
        one-line human-readable reason. A seam: tests monkeypatch this to a
        no-op returning ``None`` alongside a stubbed :meth:`build_gateway`.
        """
        from molexp.services.plan_runtime import preflight_plan_router

        return preflight_plan_router(model=model)

    @staticmethod
    def build_gateway(
        *,
        model: str,
        run: Run,
        router: Router | None = None,
        workspace_root: str | Path | None = None,
        task_id: str | None = None,
        draft: str | None = None,
        turn_id: str | None = None,
    ) -> AgentGateway:
        """Build the production gateway for ``run`` from the resolved ``model``.

        Delegates to the shared service builder
        (:func:`molexp.services.plan_runtime.build_plan_gateway`) so the CLI
        and the server construct the exact same gateway; ``router`` reuses the
        instance :meth:`preflight` already validated. A seam: tests
        monkeypatch this to inject a ``StubAgentGateway`` instead.

        Pass ``workspace_root`` + ``task_id`` (+ optional ``draft``) so each
        LLM call is projected into the Agents-tab session cache.
        """
        from molexp.services.plan_runtime import build_plan_gateway

        return build_plan_gateway(
            model=model,
            run=run,
            router=router,
            workspace_root=workspace_root,
            task_id=task_id,
            draft=draft,
            turn_id=turn_id,
        )


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
            help="Request the phase-2 deterministic realization tail after the "
            "plan is approved. Realization is a separate phase not yet driven by "
            "this command — `molexp plan` stops at the frozen plan + report; "
            "passing --execute prints a notice.",
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
    verbose: Annotated[
        bool,
        typer.Option(
            "--verbose",
            "-v",
            help="Show the produced artifacts of the two-phase plan flow "
            "(emergent planning → deterministic realization) on completion.",
        ),
    ] = False,
) -> None:
    """Turn an experiment draft into a frozen experiment plan (emergent planning)."""
    from molexp.cli._common import deterministic_run_id, rprint
    from molexp.harness import ApprovalPendingError, EmergentPlanOrchestrator, StageExecutionError
    from molexp.services.plan_runtime import PlanPreflightError
    from molexp.workspace import Workspace

    draft_text = _resolve_draft(draft, file)

    resolved_model = model or _configured_model()
    if not resolved_model:
        rprint(
            "[red]No model configured.[/red] Pass [bold]--model <id>[/bold] or run "
            "[bold]molexp config set agent.model <id>[/bold]."
        )
        raise typer.Exit(1)

    # Preflight — BEFORE any disk write. A missing agent extra, an unknown
    # model, or a missing API key exits here with a one-line reason and
    # leaves the workspace directory untouched (zero residue).
    try:
        router = PlanRuntime.preflight(model=resolved_model)
    except PlanPreflightError as exc:
        from rich.markup import escape

        # escape(): the message may contain rich-markup-lookalikes ("[agent]").
        rprint(f"[red]{escape(str(exc))}[/red]")
        raise typer.Exit(1) from exc

    workspace_root = (workspace or Path.cwd()).resolve()
    ws = Workspace(workspace_root)
    ws.materialize()
    # Content-addressed run id: the same draft maps to the same Run, so a
    # re-run replays store-first through the review gate on that Run.
    params: dict[str, JSONValue] = {"mode": "plan", "draft": draft_text}
    exp = ws.add_project(project).add_experiment(experiment)
    run = exp.add_run(params, id=deterministic_run_id(params))

    # Explicit or suspended, never implicit: an interactive approver exists
    # only on a TTY or with --yes; otherwise approve=None means the review gate
    # resolves store-first and SUSPENDS pending (exit 2) instead of granting.
    import sys

    approver = InteractiveApprover(run=run, assume_yes=yes) if (yes or sys.stdin.isatty()) else None
    mode = EmergentPlanOrchestrator(approve=approver)
    preview = draft_text.strip().splitlines()[0][:_DRAFT_PREVIEW_CHARS]
    rprint(f"[bold]molexp plan[/bold] — emergent planning on run [bold]{run.id}[/bold]")
    rprint(f"  model     : {resolved_model}")
    rprint(f"  draft     : {preview}")
    rprint(f"  workspace : {workspace_root}")
    rprint("  phase 1   : emergent planning (draft -> task board -> review gate -> frozen plan)")
    rprint("  phase 2   : deterministic realization (separate phase)")

    plan_task_id = f"plan-{run.id}"
    gateway = PlanRuntime.build_gateway(
        model=resolved_model,
        run=run,
        router=router,
        workspace_root=str(workspace_root),
        task_id=plan_task_id,
        draft=draft_text,
    )
    capability_registry = _resolve_grounding(workspace_root, ground=ground, task=draft_text)
    from molexp.services.plan_runtime import drive_plan_mode

    try:
        # drive_plan_mode wraps the pipeline in the run lifecycle so the plan
        # Run's status is honest (running -> succeeded | failed) — the same
        # shared path the server's plan-tasks use.
        result = asyncio.run(
            drive_plan_mode(
                mode,
                run=run,
                user_input=draft_text,
                gateway=gateway,
                capability_registry=capability_registry,
            )
        )
    except ApprovalPendingError as exc:
        # Suspended, not failed: the pending request is persisted in the run's
        # approval store; a decision lets a re-run resume past the gate.
        rprint(f"[yellow]Plan suspended — approval pending:[/yellow] {exc}")
        for request in exc.requests:
            rprint(f"  - [{request.intent}] {request.reason}")
        rprint(
            "[dim]To proceed: decide in the UI approvals inbox, re-run this "
            "command on a TTY to answer interactively, or re-run with --yes. "
            "A granted plan review replays store-first through the gate on the "
            "same run.[/dim]"
        )
        raise typer.Exit(2) from exc
    except StageExecutionError as exc:
        rprint(f"[red]Plan pipeline failed:[/red] {exc}")
        rprint(
            "[dim]The plan board was rejected before the review gate opened — "
            "re-running the same draft regenerates it.[/dim]"
        )
        # A terminally-failed plan still materializes (Agents-tab entry with
        # status failed + a FailureAnalysis knowledge record). The suspension
        # path above never reaches here — ApprovalPendingError is not a failure.
        from molexp.services.plan_runtime import PlanFailure, materialize_plan_records

        outcome = materialize_plan_records(
            run=run,
            experiment=exp,
            workspace_root=str(workspace_root),
            task_id=plan_task_id,
            draft=draft_text,
            model=resolved_model,
            failure=PlanFailure(stage=None, error=str(exc)),
        )
        _print_record_errors(outcome)
        raise typer.Exit(1) from exc

    rprint("\n[green]OK[/green] emergent plan completed")
    rprint(f"  artifacts : {len(result.stage_artifacts)} produced")
    if verbose:
        for ref in result.stage_artifacts:
            rprint(f"    {ref.kind:<24} {ref.id}")
    if result.final_artifact is not None:
        rprint(f"  final     : {result.final_artifact.kind}  {result.final_artifact.id}")

    # Materialize the SAME UI-facing records the server's `POST /plan-tasks`
    # writes — persist the workflow IR onto the experiment + record the Agents
    # session (with the deliverables locator) and Knowledge note — so a plan
    # produced here is identical, in the UI, to one generated from the web app.
    from molexp.services.plan_runtime import materialize_plan_records

    outcome = materialize_plan_records(
        run=run,
        experiment=exp,
        workspace_root=str(workspace_root),
        task_id=plan_task_id,
        draft=draft_text,
        model=resolved_model,
    )
    _print_record_errors(outcome)
    rprint(f"  ui session: [bold]{plan_task_id}[/bold] (open the Agents tab to see this plan)")

    if execute:
        # Honest notice: the emergent orchestrator is planning-only. Real
        # execution / realization (phase 2) is a separate phase this command
        # does not drive yet — never a silent success claim.
        rprint(
            "[yellow]--execute[/yellow]: deterministic realization (phase 2) is a "
            "separate phase not driven by this command yet — the plan stops at the "
            "frozen plan + report."
        )

    rprint(f"\n  artifacts : {run.run_dir / 'artifacts'}")
    rprint(f"  audit db  : {run.run_dir / 'harness.sqlite'}  (events + artifact lineage)")
    rprint(
        "[dim]Re-running the same draft replays store-first through the review "
        "gate on the same content-addressed run.[/dim]"
    )
