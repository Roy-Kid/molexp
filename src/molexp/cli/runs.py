"""``molexp runs ...`` — run-entity management sub-commands."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, Any

import typer
from rich.table import Table

from . import app
from ._common import (
    _TERMINAL_STATUSES,
    console,
    get_workspace,
    rprint,
    run_executor_info,
    status_color,
)

run_app = typer.Typer(help="Run management commands")
app.add_typer(run_app, name="runs")

from . import prune as _prune

_prune.register(run_app)


@run_app.command("create")
def run_create(
    project_id: Annotated[str, typer.Argument(help="Project ID")],
    experiment_id: Annotated[str, typer.Argument(help="Experiment ID")],
    params: Annotated[
        str | None,
        typer.Option("--params", help="Parameters JSON string or file path"),
    ] = None,
    path: Annotated[
        Path | None,
        typer.Option("--path", "-p", help="Workspace path"),
    ] = None,
) -> None:
    """Create a new run."""
    ws = get_workspace(path)

    parameters: dict = {}
    if params:
        params_path = Path(params)
        if params_path.exists():
            parameters = json.loads(params_path.read_text())
        else:
            try:
                parameters = json.loads(params)
            except json.JSONDecodeError:
                rprint(f"[red]Error:[/red] Invalid JSON in parameters: {params}")
                raise typer.Exit(1)

    from molexp.workspace import (
        ExperimentNotFoundError as _ExpNotFound,
    )
    from molexp.workspace import (
        ProjectNotFoundError as _ProjNotFound,
    )

    try:
        try:
            project = ws.project(project_id)
        except _ProjNotFound:
            rprint(f"[red]Error:[/red] Project not found: {project_id}")
            raise typer.Exit(1) from None

        try:
            experiment = project.experiment(experiment_id)
        except _ExpNotFound:
            rprint(f"[red]Error:[/red] Experiment not found: {experiment_id}")
            raise typer.Exit(1) from None

        r = experiment.Run(parameters=parameters)
        rprint(f"[green]OK[/green] Created run: {r.id}")
        rprint(f"  Project: {project_id}")
        rprint(f"  Experiment: {experiment_id}")
        rprint(f"  Status: {r.status}")
        if parameters:
            rprint(f"  Parameters: {json.dumps(parameters, indent=2)}")

    except typer.Exit:
        raise
    except Exception as e:
        rprint(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)


@run_app.command("list")
def run_list(
    project_id: Annotated[str, typer.Argument(help="Project ID")],
    experiment_id: Annotated[str, typer.Argument(help="Experiment ID")],
    path: Annotated[Path | None, typer.Option("--path", "-p", help="Workspace path")] = None,
) -> None:
    """List all runs in an experiment."""
    ws = get_workspace(path)

    from molexp.workspace import (
        ExperimentNotFoundError as _ExpNotFound,
    )
    from molexp.workspace import (
        ProjectNotFoundError as _ProjNotFound,
    )

    try:
        project = ws.project(project_id)
    except _ProjNotFound:
        rprint(f"[red]Error:[/red] Project not found: {project_id}")
        raise typer.Exit(1) from None

    try:
        experiment = project.experiment(experiment_id)
    except _ExpNotFound:
        rprint(f"[red]Error:[/red] Experiment not found: {experiment_id}")
        raise typer.Exit(1) from None

    runs = experiment.list_runs()

    if not runs:
        rprint(f"[yellow]No runs found in {project_id}/{experiment_id}[/yellow]")
        return

    table = Table(title=f"Runs in {project_id}/{experiment_id}")
    table.add_column("Run ID", style="cyan")
    table.add_column("Status", style="green")
    table.add_column("Profile", style="cyan")
    table.add_column("Created")

    for r in runs:
        status = str(r.status).lower()
        color = status_color(status)
        profile_display = r.metadata.profile or "—"

        table.add_row(
            r.id,
            f"[{color}]{status}[/{color}]",
            profile_display,
            r.metadata.created_at.strftime("%Y-%m-%d %H:%M:%S"),
        )

    console.print(table)


@run_app.command("cancel")
def run_cancel(
    run_ids: Annotated[
        list[str] | None,
        typer.Argument(
            help="Run IDs to cancel. Omit to use --project/--experiment with --all or --status."
        ),
    ] = None,
    project_id: Annotated[
        str | None,
        typer.Option("--project", "-p", help="Project ID (required in experiment-scope mode)."),
    ] = None,
    experiment_id: Annotated[
        str | None,
        typer.Option(
            "--experiment", "-e", help="Experiment ID (required in experiment-scope mode)."
        ),
    ] = None,
    all_runs: Annotated[
        bool,
        typer.Option("--all", help="Cancel all non-terminal runs in the experiment."),
    ] = False,
    status_filter: Annotated[
        str | None,
        typer.Option(
            "--status", help="Comma-separated statuses to filter (e.g. 'pending,running')."
        ),
    ] = None,
    scheduler: Annotated[
        str,
        typer.Option(
            "--scheduler",
            help="Fallback molq scheduler backend when run metadata lacks executor info.",
        ),
    ] = "slurm",
    cluster: Annotated[
        str | None,
        typer.Option("--cluster", help="molq cluster name (default: 'default')."),
    ] = None,
    yes: Annotated[
        bool,
        typer.Option("--yes", "-y", help="Skip confirmation prompt."),
    ] = False,
    path: Annotated[
        Path | None,
        typer.Option("--path", help="Workspace path."),
    ] = None,
) -> None:
    """Cancel one or more scheduled runs."""
    ws = get_workspace(path)

    target_runs: list[Any] = []

    from molexp.workspace import (
        ExperimentNotFoundError as _ExpNotFound,
    )
    from molexp.workspace import (
        ProjectNotFoundError as _ProjNotFound,
    )
    from molexp.workspace import (
        RunNotFoundError as _RunNotFound,
    )

    if run_ids:
        for rid in run_ids:
            found = None
            for proj in ws.list_projects():
                for exp in proj.list_experiments():
                    try:
                        found = exp.run(rid)
                    except _RunNotFound:
                        continue
                    break
                if found:
                    break
            if found is None:
                rprint(f"[yellow]Warning:[/yellow] Run {rid!r} not found — skipping.")
            else:
                target_runs.append(found)
    else:
        if not project_id or not experiment_id:
            rprint(
                "[red]Error:[/red] Provide run IDs as arguments, or both "
                "--project and --experiment options."
            )
            raise typer.Exit(1)

        try:
            project = ws.project(project_id)
        except _ProjNotFound:
            rprint(f"[red]Error:[/red] Project not found: {project_id}")
            raise typer.Exit(1) from None

        try:
            experiment = project.experiment(experiment_id)
        except _ExpNotFound:
            rprint(f"[red]Error:[/red] Experiment not found: {experiment_id}")
            raise typer.Exit(1) from None

        candidates = experiment.list_runs()

        if all_runs:
            target_runs = [r for r in candidates if r.status not in _TERMINAL_STATUSES]
        elif status_filter:
            allowed = {s.strip().lower() for s in status_filter.split(",")}
            target_runs = [r for r in candidates if r.status in allowed]
        else:
            rprint(
                "[red]Error:[/red] Specify --all or --status when using "
                "--project/--experiment mode."
            )
            raise typer.Exit(1)

    if not target_runs:
        rprint("[yellow]No runs matched the criteria — nothing to cancel.[/yellow]")
        raise typer.Exit(0)

    already_terminal = [r for r in target_runs if r.status in _TERMINAL_STATUSES]
    target_runs = [r for r in target_runs if r.status not in _TERMINAL_STATUSES]

    for r in already_terminal:
        rprint(f"[yellow]Skipping[/yellow] {r.id} — already terminal: {r.status}")

    if not target_runs:
        rprint("[yellow]All matched runs are already in a terminal state.[/yellow]")
        raise typer.Exit(0)

    table = Table(title=f"Runs to cancel ({len(target_runs)})")
    table.add_column("Run ID", style="cyan")
    table.add_column("Status", style="yellow")
    table.add_column("Scheduler", style="magenta")
    table.add_column("job_id", style="dim")
    table.add_column("scheduler_job_id", style="dim")

    for r in target_runs:
        executor_info = run_executor_info(r)
        table.add_row(
            r.id,
            r.status,
            executor_info.get("scheduler") or scheduler,
            executor_info.get("job_id") or "—",
            executor_info.get("scheduler_job_id") or "—",
        )

    console.print(table)

    if not yes:
        confirm = typer.prompt(f"\nCancel {len(target_runs)} job(s)? [y/N]", default="N")
        if confirm.strip().lower() not in ("y", "yes"):
            rprint("[dim]Aborted.[/dim]")
            raise typer.Exit(0)

    from molq import Cluster, Submitor

    submitor_cache: dict[tuple[str, str], Any] = {}

    cancelled = 0
    errors = 0

    try:
        for r in target_runs:
            executor_info = run_executor_info(r)
            molq_id = executor_info.get("job_id")
            scheduler_job_id = executor_info.get("scheduler_job_id")
            run_scheduler = executor_info.get("scheduler") or scheduler
            run_cluster = executor_info.get("cluster_name") or cluster or "default"

            if run_scheduler != "local":
                if molq_id and run_scheduler:
                    cache_key = (run_scheduler, run_cluster)
                    submitor = submitor_cache.get(cache_key)
                    if submitor is None:
                        submitor = Submitor(Cluster(name=run_cluster, scheduler=run_scheduler))
                        submitor_cache[cache_key] = submitor
                    try:
                        submitor.cancel_job(molq_id)
                    except Exception as exc:
                        rprint(
                            f"  [yellow]Warning:[/yellow] scheduler cancel failed for "
                            f"{r.id} (job_id={molq_id}, scheduler={run_scheduler}): {exc}"
                        )
                        errors += 1
                elif scheduler_job_id and run_scheduler:
                    cache_key = (run_scheduler, run_cluster)
                    submitor = submitor_cache.get(cache_key)
                    if submitor is None:
                        submitor = Submitor(Cluster(name=run_cluster, scheduler=run_scheduler))
                        submitor_cache[cache_key] = submitor
                    try:
                        submitor._scheduler_impl.cancel(scheduler_job_id)
                    except Exception as exc:
                        rprint(
                            f"  [yellow]Warning:[/yellow] scheduler cancel failed for "
                            f"{r.id} (scheduler_job_id={scheduler_job_id}, scheduler={run_scheduler}): {exc}"
                        )
                        errors += 1
                else:
                    rprint(
                        f"  [yellow]Warning:[/yellow] {r.id} has no molq job metadata — "
                        "updating workspace state only."
                    )

            r.cancel()
            rprint(f"  [green]OK[/green] Cancelled {r.id}")
            cancelled += 1
    finally:
        for submitor in submitor_cache.values():
            submitor.close()

    rprint(f"\n[green]Done.[/green] {cancelled} run(s) cancelled", end="")
    if errors:
        rprint(f", [yellow]{errors} scheduler error(s)[/yellow] (workspace state updated).")
    else:
        rprint(".")


@run_app.command("info")
def run_info(
    project_id: Annotated[str, typer.Argument(help="Project ID")],
    experiment_id: Annotated[str, typer.Argument(help="Experiment ID")],
    run_id: Annotated[str, typer.Argument(help="Run ID")],
    path: Annotated[Path | None, typer.Option("--path", "-p", help="Workspace path")] = None,
) -> None:
    """Show run information."""
    from molexp.workspace import (
        ExperimentNotFoundError as _ExpNotFound,
    )
    from molexp.workspace import (
        ProjectNotFoundError as _ProjNotFound,
    )
    from molexp.workspace import (
        RunNotFoundError as _RunNotFound,
    )

    ws = get_workspace(path)

    try:
        project = ws.project(project_id)
    except _ProjNotFound:
        rprint(f"[red]Error:[/red] Project not found: {project_id}")
        raise typer.Exit(1) from None

    try:
        experiment = project.experiment(experiment_id)
    except _ExpNotFound:
        rprint(f"[red]Error:[/red] Experiment not found: {experiment_id}")
        raise typer.Exit(1) from None

    try:
        r = experiment.run(run_id)
    except _RunNotFound:
        rprint(f"[red]Error:[/red] Run not found: {run_id}")
        raise typer.Exit(1) from None

    rprint(f"[bold]Run:[/bold] {r.id}")
    rprint(f"  Status: {r.status}")
    rprint(f"  Created: {r.metadata.created_at}")
    if r.metadata.finished_at:
        rprint(f"  Finished: {r.metadata.finished_at}")
    if r.metadata.profile:
        rprint(f"  Profile: [cyan]{r.metadata.profile}[/cyan]")
        if r.metadata.config_hash:
            rprint(f"  Config hash: {r.metadata.config_hash[:12]}…")
        if r.metadata.config:
            rprint(f"  Config: {json.dumps(r.metadata.config, indent=2, default=str)}")
    rprint(f"  Parameters: {json.dumps(r.parameters, indent=2, default=str)}")
