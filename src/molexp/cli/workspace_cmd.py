"""``molexp init`` / ``molexp info`` — workspace top-level commands."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from molexp.workspace import Workspace

from . import app
from ._common import get_workspace, rprint, status_color


@app.command()
def init(
    path: Annotated[
        Path | None,
        typer.Argument(help="Workspace path (default: current directory)"),
    ] = None,
) -> None:
    """Initialize a new workspace.

    Constructs a :class:`~molexp.workspace.Workspace` and *materializes*
    it so the directory and ``workspace.json`` actually appear on disk.
    The ``Workspace(...)`` constructor itself is side-effect-free by
    design (see CLAUDE.md), so this command must call ``materialize()``
    explicitly — otherwise ``molexp serve`` later complains the
    workspace doesn't exist.
    """
    workspace_path = path or Path.cwd()
    ws = Workspace(workspace_path)
    ws.materialize()

    rprint(f"[green]OK[/green] Initialized workspace at: {ws.root}")
    rprint(f"  - Projects directory: {ws.root / 'projects'}")
    rprint(f"  - Assets directory: {ws.root / 'assets'}")


@app.command()
def info(
    path: Annotated[
        Path | None,
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
