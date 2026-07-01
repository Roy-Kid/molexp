"""``molexp curate`` — workspace curation, LLM-free ops + a natural-language ``ask``.

A Typer group with two front-ends onto the **one** curation execution stack
(the §8 ChangeProposal gate):

- ``molexp curate ask "<NL>"`` — the LLM-driven flow: a natural-language request
  is planned by an agent and (for a destructive op) routed through the gate.
  Mirrors ``molexp plan``'s model resolution (``--model`` → ``molexp config``
  ``agent.model``).
- ``molexp curate move-run|delete-folder|rehome-asset`` — deterministic, LLM-free:
  the operator names the op + targets with flags; a ``ChangeProposal`` is built
  directly and gated.

Both share the ``run_curation_proposal`` backend (Python ≡ UI). Destructive ops
are gated by an :class:`InteractiveApprover` (``[y/N]`` on a TTY; auto-granted on
``--yes`` / non-interactive). Heavy imports are deferred into command bodies so
plain ``molexp --help`` stays fast.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import TYPE_CHECKING, Annotated

import typer

if TYPE_CHECKING:
    from collections.abc import Callable

    from molexp.harness.gateways.gateway import AgentGateway
    from molexp.harness.schemas import ApprovalDecision, ApprovalRequest, ChangeProposal
    from molexp.workspace.run import Run

__all__ = ["curate_app"]

_REQUEST_PREVIEW_CHARS = 80

curate_app = typer.Typer(
    name="curate",
    help="Workspace curation — deterministic ops (move-run/delete-folder/rehome-asset) + NL `ask`.",
    no_args_is_help=True,
)


class InteractiveApprover:
    """``Approver`` for the curation ChangeProposal gate.

    A callable approver (mirrors the ``Approver`` protocol via :meth:`__call__`).
    It **auto-grants without prompting** when ``assume_yes`` is set or stdin is
    not a TTY (CI, pipes, the test runner). On an interactive terminal it prints
    the proposed high-risk op and prompts ``[y/N]`` before the mutation runs; a
    rejection is *recorded* by the gate (``status="rejected"``) — no mutation.
    """

    def __init__(self, *, assume_yes: bool = False) -> None:
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

        from molexp.cli._common import rprint

        proposal_id = request.metadata.get("proposal_id", "?")
        op = request.metadata.get("high_risk_op", "?")
        rprint("\n[bold]This curation step mutates high-risk state:[/bold]")
        rprint(f"  proposal : {proposal_id}")
        rprint(f"  op       : {op}")
        answer = input(f"Approve and run? [{request.intent}] [y/N] ").strip().lower()
        return ApprovalDecision(
            request_id=request.id,
            granted=answer in ("y", "yes"),
            decided_by="cli-interactive",
            decided_at=datetime.now(tz=UTC),
            reason=f"operator answered {answer!r}",
        )


def _configured_model() -> str | None:
    """Return the ``agent.model`` value from ``molexp config``, if any."""
    from molexp.cli.agent_cmd import _configured_model as agent_configured_model

    return agent_configured_model()


def _build_gateway(*, model: str, run: Run) -> AgentGateway:
    """Build the production curate gateway (a seam mirroring ``PlanRuntime``)."""
    from molexp.server.curate_runtime.gateway import build_curate_gateway

    return build_curate_gateway(model=model, run=run)


# ── curate ask (natural-language) ─────────────────────────────────────────────


@curate_app.command("ask")
def curate_ask(
    request: Annotated[
        str | None,
        typer.Argument(help="Natural-language workspace-curation request (or use --file)."),
    ] = None,
    file: Annotated[
        Path | None,
        typer.Option("--file", "-f", help="Read the curation request from a file."),
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
        typer.Option("--project", help="Project the curate run is filed under."),
    ] = "curations",
    experiment: Annotated[
        str,
        typer.Option("--experiment", help="Experiment the curate run is filed under."),
    ] = "curate",
    yes: Annotated[
        bool,
        typer.Option(
            "--yes/--non-interactive",
            "-y",
            help="Auto-approve destructive curation steps (default auto-approves off a TTY).",
        ),
    ] = False,
) -> None:
    """Plan + run one curation capability from a natural-language request (LLM)."""
    from molexp._typing import JSONValue
    from molexp.cli._common import deterministic_run_id, rprint
    from molexp.server.curate_runtime.flow import CurationArgumentError, run_curation_flow
    from molexp.workspace import Workspace

    request_text = _resolve_request(request, file)

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
    params: dict[str, JSONValue] = {"mode": "curate", "request": request_text}
    exp = ws.add_project(project).add_experiment(experiment)
    run = exp.add_run(params, id=deterministic_run_id(params))

    preview = request_text.strip().splitlines()[0][:_REQUEST_PREVIEW_CHARS]
    rprint(f"[bold]molexp curate ask[/bold] — run [bold]{run.id}[/bold]")
    rprint(f"  model     : {resolved_model}")
    rprint(f"  request   : {preview}")
    rprint(f"  workspace : {workspace_root}")

    gateway = _build_gateway(model=resolved_model, run=run)
    try:
        result = asyncio.run(
            run_curation_flow(
                request_text,
                workspace=ws,
                experiment=exp,
                run=run,
                gateway=gateway,
                approve=InteractiveApprover(assume_yes=yes),
            )
        )
    except CurationArgumentError as exc:
        rprint(f"[red]Could not build the curation action:[/red] {exc}")
        raise typer.Exit(1) from exc

    rprint("\n[green]OK[/green] curation complete:")
    rprint(f"  capability : {result.capability_id}")
    rprint(f"  summary    : {result.mutation_summary}")
    rprint(f"  granted    : {result.granted}")
    rprint(f"\n  artifacts : {run.run_dir / 'artifacts'}")
    rprint(f"  audit db  : {run.run_dir / 'harness.sqlite'}  (events + artifact lineage)")


# ── deterministic (LLM-free) ops ──────────────────────────────────────────────

_WorkspaceOpt = Annotated[
    Path | None, typer.Option("--workspace", help="Workspace root; defaults to cwd.")
]
_ProjectOpt = Annotated[str, typer.Option("--project", help="Audit project.")]
_ExperimentOpt = Annotated[str, typer.Option("--experiment", help="Audit experiment.")]
_YesOpt = Annotated[
    bool,
    typer.Option("--yes/--non-interactive", "-y", help="Auto-approve (default off a TTY)."),
]


def _execute_proposal(
    proposal: ChangeProposal,
    *,
    label: str,
    workspace: Path | None,
    project: str,
    experiment: str,
    yes: bool,
) -> None:
    """Mint an audit run, gate + execute *proposal*, print the outcome (exit 1 if not executed)."""
    from molexp._typing import JSONValue
    from molexp.cli._common import deterministic_run_id, rprint
    from molexp.server.curate_runtime import run_curation_proposal
    from molexp.workspace import Workspace

    ws_root = (workspace or Path.cwd()).resolve()
    ws = Workspace(ws_root)
    ws.materialize()

    params: dict[str, JSONValue] = {
        "mode": "curate-propose",
        "op": proposal.proposed_change.op,
        "proposal": proposal.id,
    }
    run = (
        ws.add_project(project)
        .add_experiment(experiment)
        .add_run(params, id=deterministic_run_id(params))
    )
    rprint(f"[bold]molexp curate {label}[/bold] — proposal [bold]{proposal.id}[/bold]")

    result = asyncio.run(
        run_curation_proposal(
            proposal, workspace=ws, run=run, approve=InteractiveApprover(assume_yes=yes)
        )
    )
    outcome = result.execution_result
    status = outcome.status if outcome is not None else "failed"
    color = "green" if status == "executed" else "yellow"
    # Print the chemist-facing curation verb (move_run/…), not the engine op.
    curation_op = proposal.proposed_change.payload.get("curation_op", proposal.proposed_change.op)
    rprint(f"  op     : {curation_op}")
    rprint(f"  status : [{color}]{status}[/{color}]")
    if status != "executed":
        reason = outcome.reason if outcome is not None else None
        rprint(f"  reason : [red]{reason or 'no reason recorded'}[/red]")
        raise typer.Exit(1)


def _build_or_exit(thunk: Callable[[], ChangeProposal]) -> ChangeProposal:
    """Run *thunk* (a ``build_curation_proposal`` call); a ``ValueError`` exits 1."""
    from molexp.cli._common import rprint

    try:
        return thunk()
    except ValueError as exc:
        rprint(f"[red]{exc}[/red]")
        raise typer.Exit(1) from exc


@curate_app.command("move-run")
def curate_move_run(
    run: Annotated[str, typer.Option("--run", help="Run id to relocate.")],
    to: Annotated[str, typer.Option("--to", help="Target experiment id.")],
    workspace: _WorkspaceOpt = None,
    project: _ProjectOpt = "curations",
    experiment: _ExperimentOpt = "curate",
    yes: _YesOpt = False,
) -> None:
    """Relocate a run to another experiment (gated)."""
    from molexp.server.curate_runtime import build_curation_proposal

    proposal = _build_or_exit(
        lambda: build_curation_proposal("move_run", run=run, target_experiment=to)
    )
    _execute_proposal(
        proposal,
        label="move-run",
        workspace=workspace,
        project=project,
        experiment=experiment,
        yes=yes,
    )


@curate_app.command("delete-folder")
def curate_delete_folder(
    folder: Annotated[str, typer.Option("--folder", help="Run/folder id to delete.")],
    workspace: _WorkspaceOpt = None,
    project: _ProjectOpt = "curations",
    experiment: _ExperimentOpt = "curate",
    yes: _YesOpt = False,
) -> None:
    """Delete a folder / run (gated, irreversible)."""
    from molexp.server.curate_runtime import build_curation_proposal

    proposal = _build_or_exit(lambda: build_curation_proposal("delete_folder", folder=folder))
    _execute_proposal(
        proposal,
        label="delete-folder",
        workspace=workspace,
        project=project,
        experiment=experiment,
        yes=yes,
    )


@curate_app.command("rehome-asset")
def curate_rehome_asset(
    asset: Annotated[str, typer.Option("--asset", help="Data-asset id.")],
    from_: Annotated[str, typer.Option("--from", help="Source experiment id.")],
    to: Annotated[str, typer.Option("--to", help="Target experiment id.")],
    action: Annotated[str, typer.Option("--action", help="'copy' (default) or 'move'.")] = "copy",
    workspace: _WorkspaceOpt = None,
    project: _ProjectOpt = "curations",
    experiment: _ExperimentOpt = "curate",
    yes: _YesOpt = False,
) -> None:
    """Re-home a data asset into another experiment scope (gated)."""
    from molexp.server.curate_runtime import build_curation_proposal

    proposal = _build_or_exit(
        lambda: build_curation_proposal(
            "rehome_asset",
            asset=asset,
            source={"kind": "experiment", "id": from_},
            target={"kind": "experiment", "id": to},
            action=action,
        )
    )
    _execute_proposal(
        proposal,
        label="rehome-asset",
        workspace=workspace,
        project=project,
        experiment=experiment,
        yes=yes,
    )


def _resolve_request(request: str | None, file: Path | None) -> str:
    """Return the request text from exactly one of ``request`` / ``file``."""
    from molexp.cli._common import rprint

    if (request is None) == (file is None):
        rprint(
            "[red]Provide the request exactly one way:[/red] either as the "
            "[bold]REQUEST[/bold] argument or via [bold]--file <path>[/bold]."
        )
        raise typer.Exit(1)
    if file is not None:
        try:
            text = file.read_text(encoding="utf-8")
        except OSError as exc:
            rprint(f"[red]Could not read request file:[/red] {exc}")
            raise typer.Exit(1) from exc
    else:
        assert request is not None  # narrowed by the exactly-one check above
        text = request
    if not text.strip():
        rprint("[red]The curation request is empty.[/red]")
        raise typer.Exit(1)
    return text
