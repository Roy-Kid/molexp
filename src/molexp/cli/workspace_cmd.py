"""``molexp init`` / ``molexp info`` — workspace top-level commands."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Optional

import typer

from molexp.workspace import Workspace

from . import app
from ._common import get_workspace, rprint, status_color


@app.command()
def init(
    path: Annotated[
        Optional[Path],
        typer.Argument(help="Workspace path (default: current directory)"),
    ] = None,
) -> None:
    """Initialize a new workspace."""
    workspace_path = path or Path.cwd()
    ws = Workspace.from_path(workspace_path)

    rprint(f"[green]OK[/green] Initialized workspace at: {ws.root}")
    rprint(f"  - Projects directory: {ws.root / 'projects'}")
    rprint(f"  - Assets directory: {ws.root / 'assets'}")


@app.command()
def info(
    path: Annotated[
        Optional[Path],
        typer.Option("--path", "-p", help="Workspace path"),
    ] = None,
) -> None:
    """Show workspace information."""
    ws = get_workspace(path)

    projects = ws.list_projects()

    total_experiments = 0
    total_runs = 0
    run_status_counts: dict[str, int] = {
        "pending": 0,
        "running": 0,
        "succeeded": 0,
        "failed": 0,
        "cancelled": 0,
    }
    profile_counts: dict[str, int] = {}

    for project in projects:
        experiments = project.list_experiments()
        total_experiments += len(experiments)

        for experiment in experiments:
            runs = experiment.list_runs()
            total_runs += len(runs)

            for r in runs:
                status = str(r.status).lower()
                if status in run_status_counts:
                    run_status_counts[status] += 1
                pname = r.metadata.profile
                if pname:
                    profile_counts[pname] = profile_counts.get(pname, 0) + 1

    rprint(f"[bold]Workspace:[/bold] {ws.root}")
    rprint("\n[bold]Statistics:[/bold]")
    rprint(f"  Projects: {len(projects)}")
    rprint(f"  Experiments: {total_experiments}")
    rprint(f"  Runs: {total_runs}")

    if total_runs > 0:
        rprint("\n[bold]Run Status:[/bold]")
        for status, count in run_status_counts.items():
            if count > 0:
                color = status_color(status)
                rprint(f"  [{color}]{status.capitalize()}[/{color}]: {count}")

    if profile_counts:
        rprint("\n[bold]Profiles:[/bold]")
        for pname, count in sorted(profile_counts.items()):
            rprint(f"  [cyan]{pname}[/cyan]: {count}")
