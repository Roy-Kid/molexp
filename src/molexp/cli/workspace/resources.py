"""``molexp workspace {project,experiment,runs,target,asset,mcp}`` — resource CRUD."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, Any

import typer
from molcfg import Config, ConfigError
from rich.console import Console
from rich.table import Table

from molexp.cli._common import (
    _TERMINAL_STATUSES,
    get_workspace,
    rprint,
    run_executor_info,
    status_color,
)
from molexp.cli.workspace import _get_ctx_target, workspace_app
from molexp.workspace.target import LocalTarget

_console = Console()

# ---------------------------------------------------------------------------
# project
# ---------------------------------------------------------------------------

project_app = typer.Typer(help="Project management commands", no_args_is_help=True)
workspace_app.add_typer(project_app, name="project")


@project_app.command("create")
def project_create(
    ctx: typer.Context,
    name: Annotated[str, typer.Argument(help="Project name")],
) -> None:
    """Create a new project."""
    target = _get_ctx_target(ctx)
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
def project_list(ctx: typer.Context) -> None:
    """List all projects."""
    target = _get_ctx_target(ctx)
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
    ctx: typer.Context,
    project_id: Annotated[str, typer.Argument(help="Project ID")],
) -> None:
    """Show project information."""
    from molexp.workspace import ProjectNotFoundError

    target = _get_ctx_target(ctx)
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
workspace_app.add_typer(experiment_app, name="experiment")


@experiment_app.command("create")
def experiment_create(
    ctx: typer.Context,
    project_id: Annotated[str, typer.Argument(help="Project ID")],
    name: Annotated[str, typer.Option("--name", "-n", help="Experiment name")],
) -> None:
    """Create a new experiment."""
    from molexp.workspace import ProjectNotFoundError

    target = _get_ctx_target(ctx)
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
    ctx: typer.Context,
    project_id: Annotated[str, typer.Argument(help="Project ID")],
) -> None:
    """List all experiments in a project."""
    from molexp.workspace import ProjectNotFoundError

    target = _get_ctx_target(ctx)
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
workspace_app.add_typer(run_app, name="runs")


@run_app.command("create")
def run_create(
    ctx: typer.Context,
    project_id: Annotated[str, typer.Argument(help="Project ID")],
    experiment_id: Annotated[str, typer.Argument(help="Experiment ID")],
    params: Annotated[
        str | None, typer.Option("--params", help="Parameters JSON string or file path")
    ] = None,
) -> None:
    """Create a new run."""
    target = _get_ctx_target(ctx)
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
        r = experiment.add_run(parameters=parameters)
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
    ctx: typer.Context,
    project_id: Annotated[str, typer.Argument(help="Project ID")],
    experiment_id: Annotated[str, typer.Argument(help="Experiment ID")],
) -> None:
    """List all runs in an experiment."""
    target = _get_ctx_target(ctx)
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
    _console.print(table)


@run_app.command("cancel")
def run_cancel(
    ctx: typer.Context,
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
) -> None:
    """Cancel one or more scheduled runs."""
    target = _get_ctx_target(ctx)
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
    ctx: typer.Context,
    project_id: Annotated[str, typer.Argument(help="Project ID")],
    experiment_id: Annotated[str, typer.Argument(help="Experiment ID")],
    run_id: Annotated[str, typer.Argument(help="Run ID")],
) -> None:
    """Show run information."""
    target = _get_ctx_target(ctx)
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
    if r.metadata.finished_at:
        rprint(f"  Finished: {r.metadata.finished_at}")
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

target_app = typer.Typer(help="Manage compute targets", no_args_is_help=True)
workspace_app.add_typer(target_app, name="target")


@target_app.command("list")
def target_list(ctx: typer.Context) -> None:
    """List all compute targets registered on the workspace."""
    from molexp.workspace.targets import list_targets as _list_targets

    target = _get_ctx_target(ctx)
    if not isinstance(target, LocalTarget):
        _remote_only("target list")
    ws = get_workspace(target.path if target.path != Path.cwd() else None)
    targets = _list_targets(ws)
    if not targets:
        rprint("[yellow]No compute targets registered.[/yellow]")
        rprint(
            "  Add one with: [cyan]molexp workspace . target add NAME --scratch /path [--host user@host] [--scheduler slurm|pbs|lsf][/cyan]"
        )
        return
    for t in targets:
        location = f"[cyan]{t.host}[/cyan]" if t.host else "[dim]local[/dim]"
        rprint(
            f"[bold]{t.name}[/bold]  location={location}  scheduler=[magenta]{t.scheduler}[/magenta]  scratch={t.scratch_root}"
        )


@target_app.command("add")
def target_add(
    ctx: typer.Context,
    name: Annotated[str, typer.Argument(help="Target name (used by --target)")],
    scratch: Annotated[
        str, typer.Option("--scratch", help="Absolute scratch root on the target's filesystem")
    ],
    scheduler: Annotated[
        str, typer.Option("--scheduler", help="Dispatch axis: local | slurm | pbs | lsf")
    ] = "local",
    host: Annotated[str | None, typer.Option("--host", help="user@host for SSH transport")] = None,
    port: Annotated[int | None, typer.Option("--port", help="SSH port")] = None,
    identity: Annotated[str | None, typer.Option("--identity", help="SSH identity file")] = None,
    ssh_opt: Annotated[
        list[str] | None, typer.Option("--ssh-opt", help="Extra ssh argv (repeatable)")
    ] = None,
) -> None:
    """Add a compute target to the workspace."""
    from typing import Literal

    from molexp.workspace import ComputeTarget
    from molexp.workspace import add_target as _add_target

    target = _get_ctx_target(ctx)
    if not isinstance(target, LocalTarget):
        _remote_only("target add")
    ws = get_workspace(target.path if target.path != Path.cwd() else None)
    narrowed_scheduler: Literal["local", "slurm", "pbs", "lsf"]
    if scheduler == "local":
        narrowed_scheduler = "local"
    elif scheduler == "slurm":
        narrowed_scheduler = "slurm"
    elif scheduler == "pbs":
        narrowed_scheduler = "pbs"
    elif scheduler == "lsf":
        narrowed_scheduler = "lsf"
    else:
        rprint(
            f"[red]Invalid scheduler {scheduler!r}[/red] — must be one of: local, slurm, pbs, lsf"
        )
        raise typer.Exit(2)

    try:
        ct = ComputeTarget(
            name=name,
            host=host,
            port=port,
            identity_file=identity,
            ssh_opts=list(ssh_opt or []),
            scheduler=narrowed_scheduler,
            scratch_root=scratch,
        )
    except ValueError as exc:
        rprint(f"[red]Invalid target:[/red] {exc}")
        raise typer.Exit(2) from exc

    try:
        _add_target(ws, ct)
    except ValueError as exc:
        rprint(f"[red]{exc}[/red]")
        raise typer.Exit(1) from exc

    location = host or "local"
    rprint(
        f"[green]OK[/green] Added target [bold]{name}[/bold] ({location}, scheduler={scheduler}, scratch={scratch})"
    )


@target_app.command("remove")
def target_remove(
    ctx: typer.Context,
    name: Annotated[str, typer.Argument(help="Target name to remove")],
) -> None:
    """Remove a compute target from the workspace."""
    from molexp.workspace.targets import remove_target as _remove_target

    target = _get_ctx_target(ctx)
    if not isinstance(target, LocalTarget):
        _remote_only("target remove")
    ws = get_workspace(target.path if target.path != Path.cwd() else None)
    try:
        _remove_target(ws, name)
    except KeyError as exc:
        rprint(f"[red]{exc}[/red]")
        raise typer.Exit(1) from exc
    rprint(f"[green]OK[/green] Removed target [bold]{name}[/bold]")


@target_app.command("test")
def target_test(
    ctx: typer.Context,
    name: Annotated[str, typer.Argument(help="Target name to verify")],
) -> None:
    """Verify connectivity to a target."""
    import shutil

    from molexp.workspace.targets import get_target as _get_target
    from molexp.workspace.targets import to_transport as _to_transport

    target = _get_ctx_target(ctx)
    if not isinstance(target, LocalTarget):
        _remote_only("target test")
    ws = get_workspace(target.path if target.path != Path.cwd() else None)
    try:
        ct = _get_target(ws, name)
    except KeyError as exc:
        rprint(f"[red]{exc}[/red]")
        raise typer.Exit(1) from exc

    if ct.is_remote:
        if shutil.which("ssh") is None:
            rprint("[red]ssh binary not found in PATH[/red]")
            raise typer.Exit(2)
        if shutil.which("rsync") is None:
            rprint("[red]rsync binary not found in PATH[/red]")
            raise typer.Exit(2)

    transport = _to_transport(ct)
    rprint(f"[bold]Testing target {name!r}...[/bold]")
    try:
        result = transport.run(["true"], timeout=15)
        if result.returncode != 0:
            rprint(f"[red]Command 'true' returned {result.returncode}: {result.stderr}[/red]")
            raise typer.Exit(1)
    except Exception as exc:
        rprint(f"[red]Transport.run() failed:[/red] {exc}")
        raise typer.Exit(1) from exc
    rprint("  [green]ok[/green] command execution")
    try:
        transport.mkdir(ct.scratch_root, parents=True, exist_ok=True)
    except Exception as exc:
        rprint(f"[red]mkdir {ct.scratch_root!r} failed:[/red] {exc}")
        raise typer.Exit(1) from exc
    rprint(f"  [green]ok[/green] mkdir {ct.scratch_root}")
    probe = f"{ct.scratch_root.rstrip('/')}/.molexp-target-test"
    try:
        transport.write_text(probe, "x")
        if transport.read_text(probe) != "x":
            rprint("[red]Round-trip mismatch on probe file[/red]")
            raise typer.Exit(1)
        transport.remove(probe)
    except Exception as exc:
        rprint(f"[red]Probe round-trip failed:[/red] {exc}")
        raise typer.Exit(1) from exc
    rprint("  [green]ok[/green] file round-trip")
    rprint(f"[green]Target {name!r} is reachable.[/green]")


# ---------------------------------------------------------------------------
# asset
# ---------------------------------------------------------------------------

asset_app = typer.Typer(help="Asset management commands", no_args_is_help=True)
workspace_app.add_typer(asset_app, name="asset")


@asset_app.command("list")
def asset_list(
    ctx: typer.Context,
    limit: Annotated[int, typer.Option("--limit", "-l", help="Limit results")] = 50,
) -> None:
    """List workspace-level assets."""
    target = _get_ctx_target(ctx)
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
# mcp
# ---------------------------------------------------------------------------

mcp_app = typer.Typer(
    name="mcp", help="Configure MCP servers (mirrors `claude mcp`).", no_args_is_help=True
)
workspace_app.add_typer(mcp_app, name="mcp")

_USER_SCOPE_PATH = Path.home() / ".claude.json"
_PROJECT_SCOPE_FILENAME = ".mcp.json"
_VALID_TRANSPORTS = ("stdio", "http", "sse")
_VALID_SCOPES = ("user", "project")

_ScopeOpt = Annotated[str, typer.Option("--scope", "-s", help="user or project")]
_ConfigOpt = Annotated[Path | None, typer.Option("--config", help="Override path from --scope")]


def _resolve_mcp_path(scope: str, override: Path | None) -> Path:
    if override is not None:
        return override.expanduser().resolve()
    if scope == "user":
        return _USER_SCOPE_PATH
    if scope == "project":
        return Path.cwd() / _PROJECT_SCOPE_FILENAME
    raise typer.BadParameter(f"Unknown scope {scope!r}. Valid: {', '.join(_VALID_SCOPES)}.")


def _load_mcp_cfg(path: Path) -> Config:
    if not path.exists():
        return Config({"mcpServers": {}})
    try:
        cfg = Config.load_json(path)
    except (json.JSONDecodeError, ConfigError) as exc:
        raise typer.BadParameter(
            f"Config file at {path} is not a valid JSON object: {exc}"
        ) from exc
    if "mcpServers" not in cfg:
        cfg["mcpServers"] = {}
    elif not isinstance(cfg["mcpServers"], Config):
        raise typer.BadParameter(f"Config file at {path} has a non-object 'mcpServers' field.")
    return cfg


def _save_mcp_cfg(cfg: Config, path: Path) -> None:
    import os as _os

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    cfg.save_json(tmp, indent=2)
    _os.chmod(tmp, 0o600)  # noqa: PTH101
    tmp.replace(path)


def _server_entry(cfg: Config, name: str) -> Config | None:
    return cfg.get(f"mcpServers.{name}")


def _server_names(cfg: Config) -> list[str]:
    servers = cfg["mcpServers"]
    return list(servers.keys()) if isinstance(servers, Config) else []


def _parse_kv_pair(raw: str, *, sep: str, kind: str) -> tuple[str, str]:
    if sep not in raw:
        raise typer.BadParameter(f"Bad {kind} {raw!r}: expected '<key>{sep}<value>'.")
    key, value = raw.split(sep, 1)
    key = key.strip()
    if kind == "--header":
        value = value.strip()
    if not key:
        raise typer.BadParameter(f"Bad {kind} {raw!r}: empty key.")
    return key, value


def _build_server_entry(
    transport: str,
    command_or_url: str,
    args: list[str],
    env_pairs: list[str],
    header_pairs: list[str],
) -> dict[str, Any]:
    if transport not in _VALID_TRANSPORTS:
        raise typer.BadParameter(
            f"Unknown transport {transport!r}. Valid: {', '.join(_VALID_TRANSPORTS)}."
        )
    if transport == "stdio":
        if header_pairs:
            raise typer.BadParameter("--header is only valid for http/sse transports.")
        env_dict = dict(_parse_kv_pair(e, sep="=", kind="--env") for e in env_pairs)
        return {"type": "stdio", "command": command_or_url, "args": list(args), "env": env_dict}
    if env_pairs:
        raise typer.BadParameter("--env is only valid for stdio transports.")
    if args:
        raise typer.BadParameter("Positional args are only valid for stdio transports.")
    headers = dict(_parse_kv_pair(h, sep=":", kind="--header") for h in header_pairs)
    return {"type": transport, "url": command_or_url, "headers": headers}


def _copy_servers(
    src_cfg: Config, dst_cfg: Config, *, force: bool
) -> tuple[list[str], list[str], list[str]]:
    added, overwritten, skipped = [], [], []
    for name in _server_names(src_cfg):
        entry = src_cfg[f"mcpServers.{name}"]
        if isinstance(entry, Config):
            entry = entry.to_dict()
        target_exists = _server_entry(dst_cfg, name) is not None
        if target_exists and not force:
            skipped.append(name)
            continue
        dst_cfg[f"mcpServers.{name}"] = entry
        (overwritten if target_exists else added).append(name)
    return added, overwritten, skipped


@mcp_app.command("add")
def mcp_add(
    name: Annotated[str, typer.Argument(help="Server name")],
    command_or_url: Annotated[
        str, typer.Argument(help="Command (stdio) or URL (http/sse)", metavar="COMMAND_OR_URL")
    ],
    args: Annotated[
        list[str] | None, typer.Argument(help="Trailing args (after -- separator)")
    ] = None,
    transport: Annotated[
        str, typer.Option("--transport", "-t", help="stdio | http | sse")
    ] = "stdio",
    env: Annotated[
        list[str] | None,
        typer.Option("--env", "-e", help="Env var (stdio only, repeatable, KEY=VALUE)"),
    ] = None,
    header: Annotated[
        list[str] | None,
        typer.Option("--header", "-H", help="HTTP header (http/sse only, repeatable, NAME:VALUE)"),
    ] = None,
    scope: _ScopeOpt = "user",
    config: _ConfigOpt = None,
    force: Annotated[bool, typer.Option("--force", "-f", help="Overwrite if entry exists")] = False,
) -> None:
    """Add an MCP server to the registry."""
    cfg_path = _resolve_mcp_path(scope, config)
    cfg = _load_mcp_cfg(cfg_path)
    if _server_entry(cfg, name) is not None and not force:
        rprint(
            f"[red]Error:[/red] Server [bold]{name}[/bold] already exists in {cfg_path}. Pass --force to overwrite."
        )
        raise typer.Exit(1)
    entry = _build_server_entry(
        transport=transport,
        command_or_url=command_or_url,
        args=list(args or []),
        env_pairs=list(env or []),
        header_pairs=list(header or []),
    )
    cfg[f"mcpServers.{name}"] = entry
    _save_mcp_cfg(cfg, cfg_path)
    rprint(f"[green]OK[/green] Added [bold]{name}[/bold] ({transport}) to {cfg_path}")


@mcp_app.command("add-json")
def mcp_add_json(
    name: Annotated[str, typer.Argument(help="Server name")],
    json_str: Annotated[
        str, typer.Argument(help="JSON object describing the server entry", metavar="JSON")
    ],
    scope: _ScopeOpt = "user",
    config: _ConfigOpt = None,
    force: Annotated[bool, typer.Option("--force", "-f", help="Overwrite if entry exists")] = False,
) -> None:
    """Add an MCP server from a raw JSON string."""
    try:
        entry = json.loads(json_str)
    except json.JSONDecodeError as exc:
        rprint(f"[red]Error:[/red] Invalid JSON: {exc}")
        raise typer.Exit(1) from None
    if not isinstance(entry, dict):
        rprint(f"[red]Error:[/red] JSON must be an object, got {type(entry).__name__}.")
        raise typer.Exit(1)
    cfg_path = _resolve_mcp_path(scope, config)
    cfg = _load_mcp_cfg(cfg_path)
    if _server_entry(cfg, name) is not None and not force:
        rprint(
            f"[red]Error:[/red] Server [bold]{name}[/bold] already exists in {cfg_path}. Pass --force to overwrite."
        )
        raise typer.Exit(1)
    cfg[f"mcpServers.{name}"] = entry
    _save_mcp_cfg(cfg, cfg_path)
    rprint(f"[green]OK[/green] Added [bold]{name}[/bold] from JSON to {cfg_path}")


@mcp_app.command("get")
def mcp_get(
    name: Annotated[str, typer.Argument(help="Server name")],
    scope: _ScopeOpt = "user",
    config: _ConfigOpt = None,
) -> None:
    """Print the JSON entry for a single server."""
    cfg_path = _resolve_mcp_path(scope, config)
    cfg = _load_mcp_cfg(cfg_path)
    entry = _server_entry(cfg, name)
    if entry is None:
        rprint(f"[red]Error:[/red] Server [bold]{name}[/bold] not found in {cfg_path}.")
        raise typer.Exit(1)
    rprint(f"[bold]{name}[/bold]  ([dim]{cfg_path}[/dim])")
    payload = entry.to_dict() if isinstance(entry, Config) else entry
    _console.print_json(json.dumps(payload))


@mcp_app.command("list")
def mcp_list(
    scope: _ScopeOpt = "user",
    config: _ConfigOpt = None,
) -> None:
    """List configured MCP servers."""
    cfg_path = _resolve_mcp_path(scope, config)
    cfg = _load_mcp_cfg(cfg_path)
    names = sorted(_server_names(cfg))
    if not names:
        rprint(f"[yellow]No MCP servers configured[/yellow] ({cfg_path})")
        return
    table = Table(title=f"MCP servers ({cfg_path})")
    table.add_column("Name", style="cyan")
    table.add_column("Transport", style="green")
    table.add_column("Command / URL")
    table.add_column("Args / Headers", overflow="fold")
    for name in names:
        entry = _server_entry(cfg, name)
        if isinstance(entry, Config):
            entry = entry.to_dict()
        if not isinstance(entry, dict):
            entry = {}
        transport = entry.get("type") or ("http" if "url" in entry else "stdio")
        if "url" in entry:
            target = entry["url"]
            extra = ", ".join(f"{k}={v!r}" for k, v in (entry.get("headers") or {}).items())
        else:
            target = entry.get("command", "")
            extra = " ".join(entry.get("args") or [])
        table.add_row(name, transport, target, extra)
    _console.print(table)


@mcp_app.command("remove")
def mcp_remove(
    name: Annotated[str, typer.Argument(help="Server name")],
    scope: _ScopeOpt = "user",
    config: _ConfigOpt = None,
) -> None:
    """Remove an MCP server from the registry."""
    cfg_path = _resolve_mcp_path(scope, config)
    cfg = _load_mcp_cfg(cfg_path)
    if _server_entry(cfg, name) is None:
        rprint(f"[red]Error:[/red] Server [bold]{name}[/bold] not found in {cfg_path}.")
        raise typer.Exit(1)
    del cfg[f"mcpServers.{name}"]
    _save_mcp_cfg(cfg, cfg_path)
    rprint(f"[green]OK[/green] Removed [bold]{name}[/bold] from {cfg_path}")


@mcp_app.command("import")
def mcp_import(
    from_path: Annotated[Path, typer.Argument(help="Source JSON file with `mcpServers` envelope")],
    scope: _ScopeOpt = "user",
    config: _ConfigOpt = None,
    force: Annotated[
        bool, typer.Option("--force", "-f", help="Overwrite entries that already exist")
    ] = False,
) -> None:
    """Copy `mcpServers` from FROM_PATH into the active registry."""
    src_path = from_path.expanduser().resolve()
    if not src_path.exists():
        rprint(f"[red]Error:[/red] Source file not found: {src_path}")
        raise typer.Exit(1)
    src_cfg = _load_mcp_cfg(src_path)
    if not _server_names(src_cfg):
        rprint(f"[yellow]Source has no `mcpServers` to import:[/yellow] {src_path}")
        return
    dst_path = _resolve_mcp_path(scope, config)
    dst_cfg = _load_mcp_cfg(dst_path)
    added, overwritten, skipped = _copy_servers(src_cfg, dst_cfg, force=force)
    _save_mcp_cfg(dst_cfg, dst_path)
    rprint(
        f"[green]OK[/green] Imported {len(added)} new, {len(overwritten)} overwritten, {len(skipped)} skipped ({src_path} → {dst_path})"
    )
    if added:
        rprint(f"  added:       {', '.join(added)}")
    if overwritten:
        rprint(f"  overwritten: {', '.join(overwritten)}")
    if skipped:
        rprint(f"  skipped:     {', '.join(skipped)}  [dim](pass --force to overwrite)[/dim]")


@mcp_app.command("export")
def mcp_export(
    to_path: Annotated[Path, typer.Argument(help="Destination JSON file")],
    scope: _ScopeOpt = "user",
    config: _ConfigOpt = None,
    force: Annotated[
        bool, typer.Option("--force", "-f", help="Overwrite entries that already exist")
    ] = False,
) -> None:
    """Copy the active `mcpServers` registry into TO_PATH."""
    src_path = _resolve_mcp_path(scope, config)
    src_cfg = _load_mcp_cfg(src_path)
    if not _server_names(src_cfg):
        rprint(f"[yellow]Active registry is empty:[/yellow] {src_path}")
        return
    dst_path = to_path.expanduser().resolve()
    dst_cfg = _load_mcp_cfg(dst_path)
    added, overwritten, skipped = _copy_servers(src_cfg, dst_cfg, force=force)
    _save_mcp_cfg(dst_cfg, dst_path)
    rprint(
        f"[green]OK[/green] Exported {len(added)} new, {len(overwritten)} overwritten, {len(skipped)} skipped ({src_path} → {dst_path})"
    )
    if added:
        rprint(f"  added:       {', '.join(added)}")
    if overwritten:
        rprint(f"  overwritten: {', '.join(overwritten)}")
    if skipped:
        rprint(f"  skipped:     {', '.join(skipped)}  [dim](pass --force to overwrite)[/dim]")


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _remote_only(cmd_name: str) -> None:  # noqa: ARG001
    """Raise an error for commands not yet supported on remote targets."""
    from molexp.cli.workspace import RemoteWorkspaceError

    raise RemoteWorkspaceError(None)  # type: ignore[arg-type]
