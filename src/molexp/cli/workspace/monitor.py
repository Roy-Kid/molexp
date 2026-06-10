"""``molexp monitor`` — job monitoring dashboard (TUI)."""

from __future__ import annotations

from typing import Annotated, Any

import typer

from molexp.cli._app import app
from molexp.cli._common import rprint
from molexp.cli._target import TargetOption, resolve_workspace_target
from molexp.workspace.target import RemoteTarget


@app.command()
def monitor(
    project: Annotated[
        str | None,
        typer.Option("--project", "-p", help="Filter by project name or ID."),
    ] = None,
    experiment: Annotated[
        str | None,
        typer.Option("--experiment", "-e", help="Filter by experiment name or ID."),
    ] = None,
    refresh: Annotated[
        float,
        typer.Option("--refresh", "-r", help="Refresh interval in seconds."),
    ] = 2.0,
    target_spec: TargetOption = ".",
) -> None:
    """Open the run dashboard for every run in the workspace.

    Press ``q`` to close; jobs keep running in the background.
    """
    target, _transport, _fs = resolve_workspace_target(target_spec)

    if isinstance(target, RemoteTarget):
        rprint("[yellow]Remote job monitoring via workspace is not yet supported.[/yellow]")
        rprint("Use [bold]molq monitor[/bold] directly for remote job dashboards.")
        raise typer.Exit(1)

    try:
        from molexp.workspace import Workspace as _Workspace

        ws = _Workspace.load(target.path)
    except FileNotFoundError:
        rprint(f"[red]Error:[/red] No workspace found at {target.path}")
        rprint("  Run [bold]molexp init[/bold] to create one.")
        raise typer.Exit(1)  # noqa: B904

    runs: list[Any] = []
    for proj in ws.list_projects():
        if project and proj.id != project and proj.name != project:
            continue
        for exp in proj.list_experiments():
            if experiment and exp.id != experiment and exp.name != experiment:
                continue
            runs.extend(exp.list_runs())

    if not runs:
        rprint("[yellow]No runs found[/yellow] — check --project / --experiment filters.")
        raise typer.Exit(0)

    from molexp.cli.tui import RunMonitor

    title = ws.name
    if experiment:
        title = f"{ws.name} / {experiment}"
    elif project:
        title = f"{ws.name} / {project}"

    rprint(f"[dim]Monitoring {len(runs)} runs. Press q to close.[/dim]")
    RunMonitor(title=title, refresh_interval=refresh).watch(runs)
    rprint("\n[dim]Monitor closed. Runs are still executing (if any).[/dim]")
