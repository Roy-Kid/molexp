"""``molexp experiment ...`` — experiment management sub-commands."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
from rich.table import Table

from . import app
from ._common import console, get_workspace, rprint

experiment_app = typer.Typer(help="Experiment management commands")
app.add_typer(experiment_app, name="experiment")


@experiment_app.command("create")
def experiment_create(
    project_id: Annotated[str, typer.Argument(help="Project ID")],
    name: Annotated[str, typer.Option("--name", "-n", help="Experiment name")],
    path: Annotated[Path | None, typer.Option("--path", "-p", help="Workspace path")] = None,
) -> None:
    """Create a new experiment."""
    from molexp.workspace import ProjectNotFoundError

    ws = get_workspace(path)

    try:
        try:
            project = ws.get_project(project_id)
        except ProjectNotFoundError:
            rprint(f"[red]Error:[/red] Project not found: {project_id}")
            raise typer.Exit(1) from None

        experiment = project.add_experiment(name)
        rprint(f"[green]OK[/green] Created experiment: {experiment.id}")
        rprint(f"  Name: {experiment.name}")
        rprint(f"  Project: {project_id}")
    except typer.Exit:
        raise
    except Exception as e:
        rprint(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)


@experiment_app.command("list")
def experiment_list(
    project_id: Annotated[str, typer.Argument(help="Project ID")],
    path: Annotated[Path | None, typer.Option("--path", "-p", help="Workspace path")] = None,
) -> None:
    """List all experiments in a project."""
    from molexp.workspace import ProjectNotFoundError

    ws = get_workspace(path)
    try:
        project = ws.get_project(project_id)
    except ProjectNotFoundError:
        rprint(f"[red]Error:[/red] Project not found: {project_id}")
        raise typer.Exit(1) from None

    experiments = project.list_experiments()

    if not experiments:
        rprint(f"[yellow]No experiments found in project: {project_id}[/yellow]")
        return

    table = Table(title=f"Experiments in {project_id}")
    table.add_column("ID", style="cyan")
    table.add_column("Name", style="green")
    table.add_column("Created")

    for exp in experiments:
        table.add_row(
            exp.id,
            exp.name,
            exp.created_at.strftime("%Y-%m-%d %H:%M"),
        )

    console.print(table)
