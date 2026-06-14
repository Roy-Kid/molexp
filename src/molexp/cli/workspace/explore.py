"""``molexp explore`` — interactive workspace explorer (TUI)."""

from __future__ import annotations

from typing import Annotated

import typer

from molexp.cli._app import app
from molexp.cli._common import rprint
from molexp.cli._target import TargetOption, resolve_workspace_target
from molexp.workspace.target import RemoteTarget


@app.command()
def explore(
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
    """Open the full-screen workspace explorer.

    Navigate with arrows / Enter to expand, Space to select,
    a/A to select all/clear, d to open delete confirmation.
    """
    target, _transport, _fs = resolve_workspace_target(target_spec)

    if isinstance(target, RemoteTarget):
        rprint("[yellow]Remote workspace explore is not yet supported.[/yellow]")
        rprint("Use [bold]molexp exec[/bold] or [bold]shell[/bold] instead.")
        raise typer.Exit(1)

    try:
        from molexp.workspace import Workspace as _Workspace

        ws = _Workspace.load(target.path)
    except FileNotFoundError:
        rprint(f"[red]Error:[/red] No workspace found at {target.path}")
        rprint("  Run [bold]molexp init[/bold] to create one.")
        raise typer.Exit(1)  # noqa: B904

    from molexp.cli.tui import TreeMonitor

    monitor = TreeMonitor(
        project_filter=project,
        experiment_filter=experiment,
        refresh_interval=refresh,
    )
    warnings = monitor.watch(ws)

    rprint("\n[dim]Explorer closed.[/dim]")
    for msg in warnings:
        rprint(f"[yellow]warning:[/yellow] {msg}")
