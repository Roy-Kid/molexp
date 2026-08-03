"""``molexp workspace {project,experiment,runs,target,asset}`` — resource CRUD.

The ``mcp`` config group lives in the sibling :mod:`.mcp_config` module.
"""

from __future__ import annotations

import json
import sys
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
from molexp.workspace.run import RETRYABLE_STATUSES
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


def _stdin_is_interactive() -> bool:
    """Whether stdin is a real terminal (a seam so tests can fake a TTY)."""
    return sys.stdin.isatty()


def _classify_container_id(ws, candidate: str) -> str | None:  # noqa: ANN001
    """Return ``"project"`` / ``"experiment"`` when *candidate* names one.

    Used by ``runs cancel`` to catch the classic mix-up: ``runs list`` takes
    ``PROJECT EXPERIMENT`` positionals while ``runs cancel`` takes bare
    ``RUN_IDS`` — a container name passed as a run id must error with the
    correct usage instead of a silent warn-and-skip.
    """
    from molexp.workspace import ExperimentNotFoundError, ProjectNotFoundError

    try:
        ws.get_project(candidate)
    except ProjectNotFoundError:
        pass
    else:
        return "project"
    for proj in ws.list_projects():
        try:
            proj.get_experiment(candidate)
        except ExperimentNotFoundError:
            continue
        else:
            return "experiment"
    return None


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
        misclassified: list[tuple[str, str]] = []
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
            if found is not None:
                target_runs.append(found)
                continue
            kind = _classify_container_id(ws, rid)
            if kind is not None:
                misclassified.append((rid, kind))
            else:
                rprint(f"[yellow]Warning:[/yellow] Run {rid!r} not found — skipping.")
        if misclassified:
            for rid, kind in misclassified:
                article = "an" if kind == "experiment" else "a"
                rprint(
                    f"[red]Error:[/red] {rid!r} looks like {article} {kind} id — "
                    "`molexp runs cancel` takes RUN_IDS."
                )
            proj_hint = next((rid for rid, kind in misclassified if kind == "project"), None)
            exp_hint = next((rid for rid, kind in misclassified if kind == "experiment"), None)
            list_cmd = f"molexp runs list {proj_hint or '<project>'} {exp_hint or '<experiment>'}"
            rprint(f"To target a run, use its id from: [bold]{list_cmd}[/bold]")
            raise typer.Exit(1)
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
        if not _stdin_is_interactive():
            # Never block a pipe/CI invocation on the y/N prompt.
            rprint(
                "[red]Error:[/red] non-interactive session: pass --yes to confirm "
                f"cancelling {len(target_runs)} run(s)."
            )
            raise typer.Exit(1)
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


@run_app.command("harvest")
def run_harvest(
    project_id: Annotated[str, typer.Argument(help="Project ID")],
    experiment_id: Annotated[str, typer.Argument(help="Experiment ID")],
    run_id: Annotated[str, typer.Argument(help="Run ID")],
    narrative: Annotated[str, typer.Argument(help="Interpretation of the run outcome")],
    kind: Annotated[str, typer.Option("--kind", help="Knowledge kind")] = "Finding",
    created_by: Annotated[str, typer.Option("--created-by")] = "cli",
    target_spec: TargetOption = ".",
) -> None:
    """Harvest a terminal run into a KnowledgeItem (workspace.harvest_run)."""
    target, _transport, _fs = resolve_workspace_target(target_spec)
    if not isinstance(target, LocalTarget):
        _remote_only("runs harvest")
    ws = get_workspace(target.path if target.path != Path.cwd() else None)
    from molexp.workspace import ExperimentNotFoundError as _ExpNotFound
    from molexp.workspace import ProjectNotFoundError as _ProjNotFound
    from molexp.workspace import RunNotFoundError as _RunNotFound
    from molexp.workspace import harvest_run, parse_knowledge_kind

    try:
        project = ws.get_project(project_id)
        experiment = project.get_experiment(experiment_id)
        run = experiment.get_run(run_id)
    except (_ProjNotFound, _ExpNotFound, _RunNotFound) as exc:
        rprint(f"[red]Error:[/red] {exc}")
        raise typer.Exit(1) from None
    try:
        item = harvest_run(
            run,
            kind=parse_knowledge_kind(kind),
            narrative=narrative,
            created_by=created_by,
        )
    except ValueError as exc:
        rprint(f"[red]Error:[/red] {exc}")
        raise typer.Exit(1) from None
    rprint(f"[green]OK[/green] Harvested KnowledgeItem: {item.name}")


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
    # A failed run must say WHY, right under the status, with the one command to
    # retry — the reason is captured in the canonical record (run.json). The
    # error block is status-gated (run-recovery bug 2): a run that has since
    # succeeded must not keep advertising a stale error + retry hint (the
    # lifecycle also clears metadata.error on success; this is the display-side
    # defense for records written before that fix).
    err = r.metadata.error
    if err is not None and r.status in RETRYABLE_STATUSES:
        rprint(f"  [red]Error:[/red] {err.type}: {err.message}")
        script = r.metadata.script if hasattr(r.metadata, "script") else None
        hint = f"molexp run {script} --resume" if script else "molexp run <script> --resume"
        rprint(f"  [dim]Retry with:[/dim] {hint}")
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
    # Recent workspace-timeline events for this run (default-on event spine;
    # the same read path the server's /events endpoint uses). Silent only when
    # the workspace has no timeline yet (nothing has emitted).
    from molexp.workspace.events import read_workspace_events

    events = read_workspace_events(ws.root, ref=r.id, limit=5)
    if events:
        rprint("  Recent events:")
        for ev in events:
            ts = ev.created_at.strftime("%Y-%m-%d %H:%M:%S")
            rprint(f"    {ts}  {ev.type}  [dim]({ev.actor})[/dim]")


# Attach prune subcommand from the prune module.
from molexp.cli import prune as _prune  # noqa: E402

_prune.register(run_app)


# ---------------------------------------------------------------------------
# target
# ---------------------------------------------------------------------------

asset_app = typer.Typer(help="Asset management commands", no_args_is_help=True)


_ASSET_SCOPE_KINDS = ("workspace", "project", "experiment", "run")


def _format_asset_scope(scope) -> str:  # noqa: ANN001
    """Render an ``AssetScope`` as ``kind:id/id/…`` (bare kind at workspace)."""
    if not scope.ids:
        return scope.kind
    return f"{scope.kind}:{'/'.join(scope.ids)}"


@asset_app.command("list")
def asset_list(
    scope: Annotated[
        str | None,
        typer.Option(
            "--scope",
            help="Filter by scope kind: workspace | project | experiment | run.",
        ),
    ] = None,
    limit: Annotated[int, typer.Option("--limit", "-l", help="Limit results")] = 50,
    target_spec: TargetOption = ".",
) -> None:
    """List assets across ALL scopes (workspace, project, experiment, run).

    The default view scans every scope's authoritative ``assets.json`` manifest
    — the same count ``molexp context`` reports — with a Scope column locating
    each asset. Use ``--scope`` to restrict to one scope kind.
    """
    target, _transport, _fs = resolve_workspace_target(target_spec)
    if not isinstance(target, LocalTarget):
        _remote_only("asset list")
    ws = get_workspace(target.path if target.path != Path.cwd() else None)

    if scope is not None and scope not in _ASSET_SCOPE_KINDS:
        rprint(
            f"[red]Error:[/red] unknown scope {scope!r} — "
            f"choose one of: {', '.join(_ASSET_SCOPE_KINDS)}."
        )
        raise typer.Exit(1)

    from molexp.workspace.assets import scan

    assets = scan.scan_assets(ws.root)
    total = len(assets)
    if scope is not None:
        assets = [a for a in assets if a.scope.kind == scope]
    shown = assets[:limit]

    if not shown:
        if scope is not None and total:
            rprint(
                f"[yellow]No {scope}-scope assets found[/yellow] "
                f"— {total} asset(s) exist in other scopes (drop --scope to see them)."
            )
        else:
            rprint("[yellow]No assets found[/yellow]")
        return
    title = "Assets (all scopes)" if scope is None else f"Assets ({scope} scope)"
    table = Table(title=title)
    table.add_column("Asset ID", style="cyan")
    table.add_column("Name", style="green")
    table.add_column("Kind")
    table.add_column("Scope", style="magenta")
    table.add_column("Created")
    for a in shown:
        table.add_row(
            a.asset_id[:12] + "...",
            a.name,
            a.kind if hasattr(a, "kind") else "-",
            _format_asset_scope(a.scope),
            a.created_at.strftime("%Y-%m-%d %H:%M"),
        )
    _console.print(table)
    if len(assets) > len(shown):
        rprint(f"[dim]Showing {len(shown)} of {len(assets)} — raise --limit to see more.[/dim]")


@asset_app.command("info")
def asset_info(
    asset_id: Annotated[str, typer.Argument(help="The asset id to inspect.")],
    target_spec: TargetOption = ".",
) -> None:
    """Show one asset's full record — wraps ``assets.scan.get_asset``."""
    from molexp.workspace.assets import scan as asset_scan

    target, _transport, _fs = resolve_workspace_target(target_spec)
    if not isinstance(target, LocalTarget):
        _remote_only("asset info")
    ws = get_workspace(target.path if target.path != Path.cwd() else None)

    asset = asset_scan.get_asset(ws.root, asset_id)
    if asset is None:
        rprint(f"[red]Error:[/red] no asset with id {asset_id!r} in this workspace.")
        raise typer.Exit(1)
    rprint(f"[bold]{asset.name}[/bold]  ({asset.asset_id})")
    # ``kind`` is the subclass-declared discriminator (base Asset omits it).
    rprint(f"  kind         : {asset.kind if hasattr(asset, 'kind') else type(asset).__name__}")
    rprint(f"  scope        : {_format_asset_scope(asset.scope)}")
    rprint(f"  content_hash : {asset.content_hash or '(none)'}")
    producer = asset.producer
    if producer is not None:
        rprint(f"  producer     : run={producer.run_id or '-'} task={producer.task_id or '-'}")
        if producer.inputs:
            rprint(f"  inputs       : {', '.join(producer.inputs)}")
    rprint(f"  path         : {asset.path}")
    rprint(f"  created      : {asset.created_at.isoformat()}")


@asset_app.command("lineage")
def asset_lineage(
    asset_id: Annotated[str, typer.Argument(help="The asset id to trace.")],
    direction: Annotated[
        str,
        typer.Option("--direction", help="ancestors | descendants | both."),
    ] = "both",
    target_spec: TargetOption = ".",
) -> None:
    """Trace an asset's provenance — wraps ``assets.lineage.ancestors/descendants``."""
    from molexp.workspace.assets import lineage as asset_lineage_mod
    from molexp.workspace.assets import scan as asset_scan

    if direction not in ("ancestors", "descendants", "both"):
        rprint(
            f"[red]Error:[/red] --direction must be ancestors|descendants|both, got {direction!r}."
        )
        raise typer.Exit(1)
    target, _transport, _fs = resolve_workspace_target(target_spec)
    if not isinstance(target, LocalTarget):
        _remote_only("asset lineage")
    ws = get_workspace(target.path if target.path != Path.cwd() else None)

    if asset_scan.get_asset(ws.root, asset_id) is None:
        rprint(f"[red]Error:[/red] no asset with id {asset_id!r} in this workspace.")
        raise typer.Exit(1)

    def _render(label: str, arrow: str, ids: set[str]) -> None:
        rprint(f"[bold]{label}[/bold] ({len(ids)}):")
        if not ids:
            rprint("  (none)")
            return
        for related_id in sorted(ids):
            related = asset_scan.get_asset(ws.root, related_id)
            suffix = (
                f"  {related.name} ({related.kind if hasattr(related, 'kind') else '?'})"
                if related is not None
                else ""
            )
            rprint(f"  {arrow} {related_id}{suffix}")

    if direction in ("ancestors", "both"):
        _render("ancestors", "<-", asset_lineage_mod.ancestors(ws, asset_id))
    if direction in ("descendants", "both"):
        _render("descendants", "->", asset_lineage_mod.descendants(ws, asset_id))


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _remote_only(cmd_name: str) -> NoReturn:  # noqa: ARG001
    """Raise an error for commands not yet supported on remote targets."""
    from molexp.cli.workspace import RemoteWorkspaceError

    raise RemoteWorkspaceError(None)
