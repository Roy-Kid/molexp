"""``molexp watch`` — reopen the interactive run monitor."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Any, Optional

import typer

from . import app
from ._common import rprint


@app.command()
def watch(
    workspace: Annotated[
        Optional[Path],
        typer.Option("--workspace", "-w", help="Workspace root (default: current directory)."),
    ] = None,
    project: Annotated[
        Optional[str],
        typer.Option("--project", "-p", help="Filter by project name or ID."),
    ] = None,
    experiment: Annotated[
        Optional[str],
        typer.Option("--experiment", "-e", help="Filter by experiment name or ID."),
    ] = None,
    refresh: Annotated[
        float,
        typer.Option("--refresh", "-r", help="Refresh interval in seconds."),
    ] = 2.0,
) -> None:
    """Reopen the full-screen run monitor for an existing workspace."""
    ws_root = Path(workspace).resolve() if workspace else Path.cwd()

    try:
        from molexp.workspace import Workspace as _Workspace
        ws = _Workspace.load(ws_root)
    except FileNotFoundError:
        rprint(f"[red]Error:[/red] No workspace found at {ws_root}")
        rprint("  Run [bold]molexp init[/bold] to create one, or pass [bold]--workspace[/bold].")
        raise typer.Exit(1)

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

    try:
        from molexp.monitor import RunMonitor
    except ImportError:
        rprint("[red]Error:[/red] molq is not installed. Install it to use the monitor.")
        raise typer.Exit(1)

    title = ws.name
    if experiment:
        title = f"{ws.name} / {experiment}"
    elif project:
        title = f"{ws.name} / {project}"

    rprint(f"[dim]Watching {len(runs)} runs. Press q to close.[/dim]")
    RunMonitor(title=title, refresh_interval=refresh).watch(runs)
    rprint("\n[dim]Monitor closed. Runs are still executing (if any).[/dim]")
