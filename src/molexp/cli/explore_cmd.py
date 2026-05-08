"""``molexp explore`` — interactive workspace explorer (tree monitor)."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Annotated

import typer

from . import app
from ._common import rprint


def _logical_cwd() -> Path:
    """Return cwd preserving symlinks (uses $PWD if it points to cwd)."""
    pwd = os.environ.get("PWD")
    if pwd and os.path.samefile(pwd, os.getcwd()):
        return Path(pwd)
    return Path(os.getcwd())


@app.command()
def explore(
    workspace: Annotated[
        Path | None,
        typer.Argument(help="Workspace root (default: current directory)."),
    ] = None,
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
    """Open the full-screen workspace explorer.

    Navigate: arrows / Enter expand, Space select, a/A select all / clear,
    d opens the delete confirmation dialog (running items show their cancel
    plan; uncancellable ones are listed and skipped, never force-deleted).
    """
    if workspace is None:
        ws_root = _logical_cwd()
    elif workspace.is_absolute():
        ws_root = workspace
    else:
        ws_root = _logical_cwd() / workspace

    try:
        from molexp.workspace import Workspace as _Workspace

        ws = _Workspace.load(ws_root)
    except FileNotFoundError:
        rprint(f"[red]Error:[/red] No workspace found at {ws_root}")
        rprint("  Run [bold]molexp init[/bold] to create one, or pass a workspace path.")
        raise typer.Exit(1)

    from molexp.tree_monitor import TreeMonitor

    monitor = TreeMonitor(
        project_filter=project,
        experiment_filter=experiment,
        refresh_interval=refresh,
    )
    warnings = monitor.watch(ws)

    rprint("\n[dim]Explorer closed.[/dim]")
    for msg in warnings:
        rprint(f"[yellow]warning:[/yellow] {msg}")
