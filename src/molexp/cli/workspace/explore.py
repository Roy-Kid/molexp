"""``molexp workspace explore`` — interactive workspace explorer (TUI)."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Annotated

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
def explore(
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
    """Open the full-screen workspace explorer.

    Navigate with arrows / Enter to expand, Space to select,
    a/A to select all/clear, d to open delete confirmation.
    """
    target = _get_ctx_target(ctx)

    if isinstance(target, RemoteTarget):
        rprint("[yellow]Remote workspace explore is not yet supported.[/yellow]")
        rprint("Use [bold]molexp workspace <target> exec[/bold] or [bold]shell[/bold] instead.")
        raise typer.Exit(1)

    try:
        from molexp.workspace import Workspace as _Workspace

        ws = _Workspace.load(target.path)
    except FileNotFoundError:
        rprint(f"[red]Error:[/red] No workspace found at {target.path}")
        rprint("  Run [bold]molexp workspace . init[/bold] to create one.")
        raise typer.Exit(1)  # noqa: B904

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
