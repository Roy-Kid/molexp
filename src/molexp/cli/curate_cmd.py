"""``molexp curate`` — run the shared workspace-curation flow from a TTY.

A thin adapter over :func:`~molexp.server.curate_runtime.flow.run_curation_flow`
— the ONE backend code path the ``curate-tasks`` route also delegates to
(Python ≡ UI). A natural-language request flows through discover → plan → (gate)
→ in-process invoke on a content-addressed ``workspace.Run``, driven by a
:class:`~molexp.harness.gateways.router_backed.RouterBackedAgentGateway` built
from the configured LLM.

Destructive curation capabilities (those declaring ``side_effects``) are gated:
on an interactive terminal the operator is prompted ``[y/N]`` before the
mutation runs; non-interactively (``--yes`` or no TTY — CI, pipes, the test
runner) they auto-grant so pipelines never block.

Model resolution mirrors ``molexp plan``: ``--model`` wins, else the
``agent.model`` key from ``molexp config``; with neither, the command fails with
an actionable message. Heavy imports are deferred into the command body so plain
``molexp --help`` stays fast.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import TYPE_CHECKING, Annotated

import typer

if TYPE_CHECKING:
    from molexp.harness.gateways.gateway import AgentGateway
    from molexp.harness.schemas import ApprovalDecision, ApprovalRequest
    from molexp.workspace.run import Run

__all__ = ["curate"]

_REQUEST_PREVIEW_CHARS = 80


class InteractiveApprover:
    """``Approver`` for ``molexp curate``'s destructive-capability gate.

    A callable approver (mirrors the ``Approver`` protocol via :meth:`__call__`).
    It **auto-grants without prompting** when ``assume_yes`` is set or stdin is
    not a TTY (CI, pipes, the test runner). On an interactive terminal it prints
    the side-effecting capability and prompts ``[y/N]`` before the mutation runs;
    a rejection raises ``StageExecutionError`` inside the gate and no mutation
    occurs.
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

        cap_id = request.metadata.get("capability_id", "?")
        effects = request.metadata.get("side_effects", [])
        rprint("\n[bold]This curation step has side effects:[/bold]")
        rprint(f"  capability   : {cap_id}")
        rprint(f"  side effects : {', '.join(effects) if effects else '(none)'}")
        answer = input(f"Approve and run? [{request.intent}] [y/N] ").strip().lower()
        return ApprovalDecision(
            request_id=request.id,
            granted=answer in ("y", "yes"),
            decided_by="cli-interactive",
            decided_at=datetime.now(tz=UTC),
            reason=f"operator answered {answer!r}",
        )


def _configured_model() -> str | None:
    """Return the ``agent.model`` value from ``molexp config``, if any.

    Delegates to the ``molexp agent`` command's resolver so both commands read
    the same configuration key. A seam: tests monkeypatch this.
    """
    from molexp.cli.agent_cmd import _configured_model as agent_configured_model

    return agent_configured_model()


def _build_gateway(*, model: str, run: Run) -> AgentGateway:
    """Build the production curate gateway (a seam mirroring ``PlanRuntime``).

    Delegates to :func:`~molexp.server.curate_runtime.gateway.build_curate_gateway`
    so the CLI and route construct the gateway identically.
    """
    from molexp.server.curate_runtime.gateway import build_curate_gateway

    return build_curate_gateway(model=model, run=run)


def curate(
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
            help="Auto-approve destructive curation steps. The default already "
            "auto-approves when stdin is not a TTY (CI/pipes).",
        ),
    ] = False,
) -> None:
    """Plan + invoke one workspace-curation capability in-process (shared flow)."""
    from molexp.cli._common import deterministic_run_id, rprint
    from molexp.harness import StageExecutionError
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
    # Content-addressed run id: the same request maps to the same Run (parity
    # with the curate-tasks route's content addressing).
    from molexp._typing import JSONValue

    params: dict[str, JSONValue] = {"mode": "curate", "request": request_text}
    exp = ws.add_project(project).add_experiment(experiment)
    run = exp.add_run(params, id=deterministic_run_id(params))

    preview = request_text.strip().splitlines()[0][:_REQUEST_PREVIEW_CHARS]
    rprint(f"[bold]molexp curate[/bold] — run [bold]{run.id}[/bold]")
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
    except StageExecutionError as exc:
        rprint(f"[red]Curation denied / failed at the approval gate:[/red] {exc}")
        raise typer.Exit(1) from exc
    except CurationArgumentError as exc:
        rprint(f"[red]Could not reconstruct curation arguments:[/red] {exc}")
        raise typer.Exit(1) from exc

    rprint("\n[green]OK[/green] curation complete:")
    rprint(f"  capability : {result.capability_id}")
    rprint(f"  summary    : {result.mutation_summary}")
    rprint(f"  granted    : {result.granted}")
    rprint(f"\n  artifacts : {run.run_dir / 'artifacts'}")
    rprint(f"  audit db  : {run.run_dir / 'harness.sqlite'}  (events + artifact lineage)")


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
