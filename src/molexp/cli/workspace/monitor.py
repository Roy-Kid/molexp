"""``molexp workspace monitor`` — job monitoring dashboard (TUI)."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Annotated, Any

import typer

from molexp.cli._common import rprint
from molexp.cli.workspace import _get_ctx_target, workspace_app
from molexp.workspace.target import RemoteTarget


def _logical_cwd() -> Path:
    pwd = os.environ.get("PWD")
    if pwd and os.path.samefile(pwd, os.getcwd()):  # noqa: PTH109, PTH121
        return Path(pwd)
    return Path(os.getcwd())  # noqa: PTH109


@workspace_app.command()
def monitor(
    ctx: typer.Context,
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
) -> None:
    """Open the run dashboard for every run in the workspace.

    Press ``q`` to close; jobs keep running in the background.
    """
    target = _get_ctx_target(ctx)

    if isinstance(target, RemoteTarget):
        rprint("[yellow]Remote job monitoring via workspace is not yet supported.[/yellow]")
        rprint("Use [bold]molq monitor[/bold] directly for remote job dashboards.")
        raise typer.Exit(1)

    try:
        from molexp.workspace import Workspace as _Workspace

        ws = _Workspace.load(target.path)
    except FileNotFoundError:
        rprint(f"[red]Error:[/red] No workspace found at {target.path}")
        rprint("  Run [bold]molexp workspace . init[/bold] to create one.")
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

    from molexp.monitor import RunMonitor

    title = ws.name
    if experiment:
        title = f"{ws.name} / {experiment}"
    elif project:
        title = f"{ws.name} / {project}"

    rprint(f"[dim]Monitoring {len(runs)} runs. Press q to close.[/dim]")
    RunMonitor(title=title, refresh_interval=refresh).watch(runs)
    rprint("\n[dim]Monitor closed. Runs are still executing (if any).[/dim]")
