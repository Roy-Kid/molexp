"""``molexp workspace {project,experiment,runs,target,asset}`` — resource CRUD.

The ``mcp`` config group lives in the sibling :mod:`.mcp_config` module.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, Any, NoReturn

import typer
from rich.console import Console
from rich.table import Table

from molexp.cli._common import (
    _TERMINAL_STATUSES,
    get_workspace,
    rprint,
    run_executor_info,
    status_color,
)
from molexp.cli._target import TargetOption, resolve_workspace_target
from molexp.workspace.target import LocalTarget

_console = Console()

# ---------------------------------------------------------------------------
# project
# ---------------------------------------------------------------------------

project_app = typer.Typer(help="Project management commands", no_args_is_help=True)


@project_app.command("create")
def project_create(
    name: Annotated[str, typer.Argument(help="Project name")],
    target_spec: TargetOption = ".",
) -> None:
    """Create a new project."""
    target, _transport, _fs = resolve_workspace_target(target_spec)
    if not isinstance(target, LocalTarget):
        _remote_only("project create")
    ws = get_workspace(target.path if target.path != Path.cwd() else None)
    try:
        project = ws.add_project(name)
        rprint(f"[green]OK[/green] Created project: {project.id}")
        rprint(f"  Name: {project.name}")
    except Exception as e:
        rprint(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)  # noqa: B904


@project_app.command("list")
def project_list(target_spec: TargetOption = ".") -> None:
    """List all projects."""
    target, _transport, _fs = resolve_workspace_target(target_spec)
    if not isinstance(target, LocalTarget):
        _remote_only("project list")
    ws = get_workspace(target.path if target.path != Path.cwd() else None)
    projects = ws.list_projects()
    if not projects:
        rprint("[yellow]No projects found[/yellow]")
        return
    table = Table(title="Projects")
    table.add_column("ID", style="cyan")
    table.add_column("Name", style="green")
    table.add_column("Owner")
    table.add_column("Tags")
    table.add_column("Created")
    for project in projects:
        table.add_row(
            project.id,
            project.name,
            project.owner,
            ", ".join(project.tags),
            project.created_at.strftime("%Y-%m-%d %H:%M"),
        )
    _console.print(table)


@project_app.command("info")
def project_info(
    project_id: Annotated[str, typer.Argument(help="Project ID")],
    target_spec: TargetOption = ".",
) -> None:
    """Show project information."""
    from molexp.workspace import ProjectNotFoundError

    target, _transport, _fs = resolve_workspace_target(target_spec)
    if not isinstance(target, LocalTarget):
        _remote_only("project info")
    ws = get_workspace(target.path if target.path != Path.cwd() else None)
    try:
        project = ws.get_project(project_id)
    except ProjectNotFoundError:
        rprint(f"[red]Error:[/red] Project not found: {project_id}")
        raise typer.Exit(1) from None
    rprint(f"[bold]Project:[/bold] {project.id}")
    rprint(f"  Name: {project.name}")
    rprint(f"  Description: {project.description}")
    rprint(f"  Owner: {project.owner}")
    rprint(f"  Tags: {', '.join(project.tags)}")
    rprint(f"  Created: {project.created_at}")
    experiments = project.list_experiments()
    rprint(f"  Experiments: {len(experiments)}")


# ---------------------------------------------------------------------------
# experiment
# ---------------------------------------------------------------------------

experiment_app = typer.Typer(help="Experiment management commands", no_args_is_help=True)


@experiment_app.command("create")
def experiment_create(
    project_id: Annotated[str, typer.Argument(help="Project ID")],
    name: Annotated[str, typer.Option("--name", "-n", help="Experiment name")],
    target_spec: TargetOption = ".",
) -> None:
    """Create a new experiment."""
    from molexp.workspace import ProjectNotFoundError

    target, _transport, _fs = resolve_workspace_target(target_spec)
    if not isinstance(target, LocalTarget):
        _remote_only("experiment create")
    ws = get_workspace(target.path if target.path != Path.cwd() else None)
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
        raise typer.Exit(1)  # noqa: B904


@experiment_app.command("list")
def experiment_list(
    project_id: Annotated[str, typer.Argument(help="Project ID")],
    target_spec: TargetOption = ".",
) -> None:
    """List all experiments in a project."""
    from molexp.workspace import ProjectNotFoundError

    target, _transport, _fs = resolve_workspace_target(target_spec)
    if not isinstance(target, LocalTarget):
        _remote_only("experiment list")
    ws = get_workspace(target.path if target.path != Path.cwd() else None)
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
        table.add_row(exp.id, exp.name, exp.created_at.strftime("%Y-%m-%d %H:%M"))
    _console.print(table)


# ---------------------------------------------------------------------------
# runs
# ---------------------------------------------------------------------------

run_app = typer.Typer(help="Run management commands", no_args_is_help=True)


@run_app.command("create")
def run_create(
    project_id: Annotated[str, typer.Argument(help="Project ID")],
    experiment_id: Annotated[str, typer.Argument(help="Experiment ID")],
    params: Annotated[
        str | None, typer.Option("--params", help="Parameters JSON string or file path")
    ] = None,
    target_spec: TargetOption = ".",
) -> None:
    """Create a new run."""
    target, _transport, _fs = resolve_workspace_target(target_spec)
    if not isinstance(target, LocalTarget):
        _remote_only("runs create")
    ws = get_workspace(target.path if target.path != Path.cwd() else None)
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
                raise typer.Exit(1)  # noqa: B904

    from molexp.workspace import ExperimentNotFoundError as _ExpNotFound
    from molexp.workspace import ProjectNotFoundError as _ProjNotFound

    try:
        try:
            project = ws.get_project(project_id)
        except _ProjNotFound:
            rprint(f"[red]Error:[/red] Project not found: {project_id}")
            raise typer.Exit(1) from None
        try:
            experiment = project.get_experiment(experiment_id)
        except _ExpNotFound:
            rprint(f"[red]Error:[/red] Experiment not found: {experiment_id}")
            raise typer.Exit(1) from None
        r = experiment.add_run(params=parameters)
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
        raise typer.Exit(1)  # noqa: B904


@run_app.command("list")
def run_list(
    project_id: Annotated[str, typer.Argument(help="Project ID")],
    experiment_id: Annotated[str, typer.Argument(help="Experiment ID")],
    target_spec: TargetOption = ".",
) -> None:
    """List all runs in an experiment."""
    target, _transport, _fs = resolve_workspace_target(target_spec)
    if not isinstance(target, LocalTarget):
        _remote_only("runs list")
    ws = get_workspace(target.path if target.path != Path.cwd() else None)
    from molexp.workspace import ExperimentNotFoundError as _ExpNotFound
    from molexp.workspace import ProjectNotFoundError as _ProjNotFound

    try:
        project = ws.get_project(project_id)
    except _ProjNotFound:
        rprint(f"[red]Error:[/red] Project not found: {project_id}")
        raise typer.Exit(1) from None
    try:
        experiment = project.get_experiment(experiment_id)
    except _ExpNotFound:
        rprint(f"[red]Error:[/red] Experiment not found: {experiment_id}")
        raise typer.Exit(1) from None
    runs = experiment.list_runs()
    if not runs:
        rprint(f"[yellow]No runs found in {project_id}/{experiment_id}[/yellow]")
        return
    from molexp._run_display import elapsed

    table = Table(title=f"Runs in {project_id}/{experiment_id}")
    table.add_column("Run ID", style="cyan")
    table.add_column("Status", style="green")
    table.add_column("Profile", style="cyan")
    table.add_column("Created")
    table.add_column("Duration")
    for r in runs:
        status = str(r.status).lower()
        color = status_color(status)
        profile_display = r.metadata.profile or "—"
        finished = r.finished_at.isoformat() if r.finished_at else None
        duration = elapsed(r.metadata.created_at.isoformat(), finished)
        table.add_row(
            r.id,
            f"[{color}]{status}[/{color}]",
            profile_display,
            r.metadata.created_at.strftime("%Y-%m-%d %H:%M:%S"),
            duration or "—",
        )
    _console.print(table)


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
        typer.Option("--project", "-p", help="Project ID (required in experiment-scope mode)"),
    ] = None,
    experiment_id: Annotated[
        str | None,
        typer.Option(
            "--experiment", "-e", help="Experiment ID (required in experiment-scope mode)"
        ),
    ] = None,
    all_runs: Annotated[
        bool, typer.Option("--all", help="Cancel all non-terminal runs in the experiment.")
    ] = False,
    status_filter: Annotated[
        str | None,
        typer.Option(
            "--status", help="Comma-separated statuses to filter (e.g. 'pending,running')."
        ),
    ] = None,
    scheduler: Annotated[
        str, typer.Option("--scheduler", help="Fallback molq scheduler backend.")
    ] = "slurm",
    cluster: Annotated[
        str | None, typer.Option("--cluster", help="molq cluster name (default: 'default').")
    ] = None,
    yes: Annotated[bool, typer.Option("--yes", "-y", help="Skip confirmation prompt.")] = False,
    target_spec: TargetOption = ".",
) -> None:
    """Cancel one or more scheduled runs."""
    target, _transport, _fs = resolve_workspace_target(target_spec)
    if not isinstance(target, LocalTarget):
        _remote_only("runs cancel")
    ws = get_workspace(target.path if target.path != Path.cwd() else None)
    target_runs: list[Any] = []

    from molexp.workspace import ExperimentNotFoundError as _ExpNotFound
    from molexp.workspace import ProjectNotFoundError as _ProjNotFound
    from molexp.workspace import RunNotFoundError as _RunNotFound

    if run_ids:
        for rid in run_ids:
            found = None
            for proj in ws.list_projects():
                for exp in proj.list_experiments():
                    try:
                        found = exp.get_run(rid)
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
            rprint("[red]Error:[/red] Provide run IDs, or both --project and --experiment.")
            raise typer.Exit(1)
        try:
            project = ws.get_project(project_id)
        except _ProjNotFound:
            rprint(f"[red]Error:[/red] Project not found: {project_id}")
            raise typer.Exit(1) from None
        try:
            experiment = project.get_experiment(experiment_id)
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
                "[red]Error:[/red] Specify --all or --status when using --project/--experiment mode."
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
    _console.print(table)

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
                            f"  [yellow]Warning:[/yellow] scheduler cancel failed for {r.id}: {exc}"
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
                            f"  [yellow]Warning:[/yellow] scheduler cancel failed for {r.id}: {exc}"
                        )
                        errors += 1
                else:
                    rprint(f"  [yellow]Warning:[/yellow] {r.id} has no molq job metadata.")
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
    target_spec: TargetOption = ".",
) -> None:
    """Show run information."""
    target, _transport, _fs = resolve_workspace_target(target_spec)
    if not isinstance(target, LocalTarget):
        _remote_only("runs info")
    ws = get_workspace(target.path if target.path != Path.cwd() else None)
    from molexp.workspace import ExperimentNotFoundError as _ExpNotFound
    from molexp.workspace import ProjectNotFoundError as _ProjNotFound
    from molexp.workspace import RunNotFoundError as _RunNotFound

    try:
        project = ws.get_project(project_id)
    except _ProjNotFound:
        rprint(f"[red]Error:[/red] Project not found: {project_id}")
        raise typer.Exit(1) from None
    try:
        experiment = project.get_experiment(experiment_id)
    except _ExpNotFound:
        rprint(f"[red]Error:[/red] Experiment not found: {experiment_id}")
        raise typer.Exit(1) from None
    try:
        r = experiment.get_run(run_id)
    except _RunNotFound:
        rprint(f"[red]Error:[/red] Run not found: {run_id}")
        raise typer.Exit(1) from None

    rprint(f"[bold]Run:[/bold] {r.id}")
    rprint(f"  Status: {r.status}")
    rprint(f"  Created: {r.metadata.created_at}")
    if r.finished_at:
        rprint(f"  Finished: {r.finished_at}")
    if r.metadata.profile:
        rprint(f"  Profile: [cyan]{r.metadata.profile}[/cyan]")
        if r.metadata.config_hash:
            rprint(f"  Config hash: {r.metadata.config_hash[:12]}…")
        if r.metadata.config:
            rprint(f"  Config: {json.dumps(r.metadata.config, indent=2, default=str)}")
    rprint(f"  Parameters: {json.dumps(r.parameters, indent=2, default=str)}")


# Attach prune subcommand from the prune module.
from molexp.cli import prune as _prune  # noqa: E402

_prune.register(run_app)


# ---------------------------------------------------------------------------
# target
# ---------------------------------------------------------------------------

asset_app = typer.Typer(help="Asset management commands", no_args_is_help=True)


@asset_app.command("list")
def asset_list(
    limit: Annotated[int, typer.Option("--limit", "-l", help="Limit results")] = 50,
    target_spec: TargetOption = ".",
) -> None:
    """List workspace-level assets."""
    target, _transport, _fs = resolve_workspace_target(target_spec)
    if not isinstance(target, LocalTarget):
        _remote_only("asset list")
    ws = get_workspace(target.path if target.path != Path.cwd() else None)
    assets = ws.assets.list_assets()[:limit]
    if not assets:
        rprint("[yellow]No assets found[/yellow]")
        return
    table = Table(title="Workspace Assets")
    table.add_column("Asset ID", style="cyan")
    table.add_column("Name", style="green")
    table.add_column("Created")
    for a in assets:
        table.add_row(a.asset_id[:12] + "...", a.name, a.created_at.strftime("%Y-%m-%d %H:%M"))
    _console.print(table)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _remote_only(cmd_name: str) -> NoReturn:  # noqa: ARG001
    """Raise an error for commands not yet supported on remote targets."""
    from molexp.cli.workspace import RemoteWorkspaceError

    raise RemoteWorkspaceError(None)
